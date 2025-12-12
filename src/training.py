"""
Training functions for SRGAN and Attentive ESRGAN.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_srgan_baseline(generator,
                         discriminator,
                         srgan,
                         vgg,
                         train_loader,
                         val_loader=None,
                         epochs=30,
                         steps_per_epoch=50):
    """
    Classical SRGAN training:
      - D: BCE on real/fake HR images
      - G: combined model with BCE (fool D) + VGG MSE (content)

    Returns:
        history: dict with per-epoch averages:
            - 'epoch'
            - 'd_loss'
            - 'g_loss'
            - 'val_psnr' (if val_loader given)
            - 'val_ssim' (if val_loader given)
    """
    batch_size = config.BATCH_SIZE
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    history = {
        "epoch": [],
        "d_loss": [],
        "g_loss": [],
    }
    if val_loader is not None:
        history["val_psnr"] = []
        history["val_ssim"] = []

    print("[SRGAN] Starting training...")

    for epoch in range(epochs):
        d_losses, g_losses = [], []
        start = time.time()

        for step in range(steps_per_epoch):
            try:
                lr_imgs, hr_imgs = train_loader.__getitem__(step % len(train_loader))
            except Exception:
                continue

            if len(lr_imgs) != batch_size:
                continue

            # ---------- Train Discriminator ----------
            fake_imgs = generator.predict(lr_imgs, verbose=0)

            d_loss_real = discriminator.train_on_batch(hr_imgs, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

            d_real_val = d_loss_real[0] if isinstance(d_loss_real, list) else d_loss_real
            d_fake_val = d_loss_fake[0] if isinstance(d_loss_fake, list) else d_loss_fake
            d_loss = 0.5 * (d_real_val + d_fake_val)
            d_losses.append(d_loss)

            # ---------- Train Generator ----------
            hr_features = vgg.predict(hr_imgs, verbose=0)
            g_loss = srgan.train_on_batch(lr_imgs, [real_labels, hr_features])
            g_total = g_loss[0] if isinstance(g_loss, list) else g_loss
            g_losses.append(g_total)

            if step % 10 == 0:
                print(f"[SRGAN] Epoch {epoch+1}/{epochs} "
                      f"Step {step}/{steps_per_epoch} | D: {d_loss:.4f} | G: {g_total:.4f}")

        # ---- End of epoch: aggregate ----
        avg_d = float(np.mean(d_losses)) if d_losses else np.nan
        avg_g = float(np.mean(g_losses)) if g_losses else np.nan

        history["epoch"].append(epoch + 1)
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)

        # ---- Optional: validation PSNR / SSIM on a few batches ----
        if val_loader is not None and len(val_loader) > 0:
            val_psnrs, val_ssims = [], []
            # only a few batches to keep it cheap
            num_val_batches = min(config.NUM_VAL_BATCHES, len(val_loader))
            for i in range(num_val_batches):
                try:
                    lr_val, hr_val = val_loader.__getitem__(i)
                except Exception:
                    continue
                if len(lr_val) == 0:
                    continue

                # G expects [-1,1], so lr_val is already [-1,1] from generator
                sr_val = generator.predict(lr_val, verbose=0)
                # Convert to [0,1]
                hr_01 = (hr_val + 1.0) / 2.0
                sr_01 = (sr_val + 1.0) / 2.0
                hr_01 = np.clip(hr_01, 0.0, 1.0)
                sr_01 = np.clip(sr_01, 0.0, 1.0)

                # Compute PSNR / SSIM per image, then average
                for h_im, s_im in zip(hr_01, sr_01):
                    tf_hr = tf.convert_to_tensor(h_im, tf.float32)
                    tf_sr = tf.convert_to_tensor(s_im, tf.float32)
                    val_psnrs.append(tf.image.psnr(tf_hr, tf_sr, max_val=1.0).numpy())
                    val_ssims.append(tf.image.ssim(tf_hr, tf_sr, max_val=1.0).numpy())

            if val_psnrs:
                history["val_psnr"].append(float(np.mean(val_psnrs)))
                history["val_ssim"].append(float(np.mean(val_ssims)))
                print(f"[SRGAN] Epoch {epoch+1} VAL | "
                      f"PSNR: {history['val_psnr'][-1]:.2f} dB | "
                      f"SSIM: {history['val_ssim'][-1]:.4f}")
            else:
                history["val_psnr"].append(np.nan)
                history["val_ssim"].append(np.nan)

        print(f"[SRGAN] Epoch {epoch+1} done in {time.time()-start:.1f}s "
              f"| D: {avg_d:.4f} | G: {avg_g:.4f}")
        
        if config.SAVE_MODELS:
            import os
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            generator.save(f"{config.MODEL_SAVE_DIR}/srgan_generator_epoch_{epoch+1}.keras")

    print("[SRGAN] Training complete.")
    return history


def train_attentive_esrgan(generator,
                           discriminator,
                           vgg,
                           train_loader,
                           val_loader=None,
                           epochs=30,
                           steps_per_epoch=50):
    """
    ESRGAN-style training:
      - RaLSGAN adversarial loss
      - VGG19 perceptual loss
      - L1 pixel loss

    Returns:
        history dict with:
            - 'epoch'
            - 'd_loss'
            - 'g_loss'
            - 'val_psnr' (optional)
            - 'val_ssim' (optional)
    """
    g_opt = keras.optimizers.Adam(
        learning_rate=config.ESRGAN_GEN_LEARNING_RATE,
        beta_1=config.ESRGAN_BETA_1,
        beta_2=config.ESRGAN_BETA_2,
        clipnorm=config.ESRGAN_CLIPNORM
    )
    d_opt = keras.optimizers.Adam(
        learning_rate=config.ESRGAN_DISC_LEARNING_RATE,
        beta_1=config.ESRGAN_BETA_1,
        beta_2=config.ESRGAN_BETA_2,
        clipnorm=config.ESRGAN_CLIPNORM
    )
    mse = keras.losses.MeanSquaredError()

    history = {
        "epoch": [],
        "d_loss": [],
        "g_loss": [],
    }
    if val_loader is not None:
        history["val_psnr"] = []
        history["val_ssim"] = []

    print("[Attentive ESRGAN] Starting training (RaLSGAN)...")

    for epoch in range(epochs):
        start = time.time()
        epoch_d_losses, epoch_g_losses = [], []

        for step in range(steps_per_epoch):
            try:
                lr_imgs, hr_imgs = train_loader.__getitem__(step % len(train_loader))
            except Exception:
                continue

            if len(lr_imgs) != config.BATCH_SIZE:
                continue

            # -------- Train Discriminator --------
            with tf.GradientTape() as tape_d:
                fake_imgs = generator(lr_imgs, training=True)

                real_logits = discriminator(hr_imgs, training=True)
                fake_logits = discriminator(fake_imgs, training=True)

                mean_fake = tf.reduce_mean(fake_logits, axis=0, keepdims=True)
                mean_real = tf.reduce_mean(real_logits, axis=0, keepdims=True)

                real_rel = real_logits - mean_fake
                fake_rel = fake_logits - mean_real

                d_loss_real = tf.reduce_mean((real_rel - 1.0) ** 2)
                d_loss_fake = tf.reduce_mean((fake_rel + 1.0) ** 2)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_grads = tape_d.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            epoch_d_losses.append(d_loss.numpy())

            # -------- Train Generator --------
            with tf.GradientTape() as tape_g:
                fake_imgs = generator(lr_imgs, training=True)

                real_logits = discriminator(hr_imgs, training=False)
                fake_logits = discriminator(fake_imgs, training=False)

                mean_fake = tf.reduce_mean(fake_logits, axis=0, keepdims=True)
                mean_real = tf.reduce_mean(real_logits, axis=0, keepdims=True)

                real_rel = real_logits - mean_fake
                fake_rel = fake_logits - mean_real

                g_loss_real = tf.reduce_mean((real_rel + 1.0) ** 2)
                g_loss_fake = tf.reduce_mean((fake_rel - 1.0) ** 2)
                adv_loss = 0.5 * (g_loss_real + g_loss_fake)

                # VGG perceptual loss
                hr_vgg = preprocess_input((hr_imgs + 1.0) * 127.5)
                fake_vgg = preprocess_input((fake_imgs + 1.0) * 127.5)
                img_features = vgg(hr_vgg, training=False)
                gen_features = vgg(fake_vgg, training=False)
                content_loss = mse(img_features, gen_features)

                # L1 pixel loss
                pixel_loss = tf.reduce_mean(tf.abs(hr_imgs - fake_imgs))

                total_g_loss = (
                    config.ESRGAN_CONTENT_LOSS_WEIGHT * content_loss +
                    config.ESRGAN_ADV_LOSS_WEIGHT * adv_loss +
                    config.ESRGAN_PIXEL_LOSS_WEIGHT * pixel_loss
                )

            g_grads = tape_g.gradient(total_g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
            epoch_g_losses.append(total_g_loss.numpy())

            if step % 10 == 0:
                print(f"[Attentive ESRGAN] Epoch {epoch+1}/{epochs} "
                      f"Step {step}/{steps_per_epoch} | "
                      f"D: {d_loss.numpy():.4f} | G: {total_g_loss.numpy():.4f}")

        # ---- end epoch: aggregate ----
        avg_d = float(np.mean(epoch_d_losses)) if epoch_d_losses else np.nan
        avg_g = float(np.mean(epoch_g_losses)) if epoch_g_losses else np.nan

        history["epoch"].append(epoch + 1)
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)

        # ---- optional validation metrics ----
        if val_loader is not None and len(val_loader) > 0:
            val_psnrs, val_ssims = [], []
            num_val_batches = min(config.NUM_VAL_BATCHES, len(val_loader))
            for i in range(num_val_batches):
                try:
                    lr_val, hr_val = val_loader.__getitem__(i)
                except Exception:
                    continue
                if len(lr_val) == 0:
                    continue

                sr_val = generator.predict(lr_val, verbose=0)
                hr_01 = (hr_val + 1.0) / 2.0
                sr_01 = (sr_val + 1.0) / 2.0
                hr_01 = np.clip(hr_01, 0.0, 1.0)
                sr_01 = np.clip(sr_01, 0.0, 1.0)

                for h_im, s_im in zip(hr_01, sr_01):
                    tf_hr = tf.convert_to_tensor(h_im, tf.float32)
                    tf_sr = tf.convert_to_tensor(s_im, tf.float32)
                    val_psnrs.append(tf.image.psnr(tf_hr, tf_sr, max_val=1.0).numpy())
                    val_ssims.append(tf.image.ssim(tf_hr, tf_sr, max_val=1.0).numpy())

            if val_psnrs:
                history["val_psnr"].append(float(np.mean(val_psnrs)))
                history["val_ssim"].append(float(np.mean(val_ssims)))
                print(f"[Attentive ESRGAN] Epoch {epoch+1} VAL | "
                      f"PSNR: {history['val_psnr'][-1]:.2f} dB | "
                      f"SSIM: {history['val_ssim'][-1]:.4f}")
            else:
                history["val_psnr"].append(np.nan)
                history["val_ssim"].append(np.nan)

        if config.SAVE_MODELS:
            import os
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            generator.save(f"{config.MODEL_SAVE_DIR}/attentive_esrgan_epoch_{epoch+1}.keras")
        
        print(f"[Attentive ESRGAN] Epoch {epoch+1} done in {time.time()-start:.1f}s | "
              f"D: {avg_d:.4f} | G: {avg_g:.4f}")

    print("[Attentive ESRGAN] Training complete.")
    return history


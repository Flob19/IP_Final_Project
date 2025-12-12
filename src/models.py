"""
Model architectures for Super-Resolution:
- SRCNN (baseline PSNR-oriented)
- SRGAN (SRResNet + BCE + VGG)
- Attentive ESRGAN (no BN, channel attention, RaLSGAN)
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG19

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_srcnn():
    """
    SRCNN architecture (3 conv layers):
    - Conv 64 @ 9x9
    - Conv 32 @ 1x1
    - Conv 3  @ 5x5
    Input/Output: RGB in [0,1].
    """
    model = models.Sequential([
        layers.Conv2D(64, (9, 9), activation='relu', padding='same',
                      input_shape=(None, None, 3)),
        layers.Conv2D(32, (1, 1), activation='relu', padding='same'),
        layers.Conv2D(3, (5, 5), activation='linear', padding='same'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.SRCNN_LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return model


# ============================================================
# SRGAN Baseline Models
# ============================================================

def residual_block_bn(x):
    """
    Residual block with BatchNorm (original SRGAN).
    Conv -> BN -> PReLU -> Conv -> BN -> Add.
    """
    shortcut = x
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    return layers.Add()([x, shortcut])


def upsample_block_bn(x):
    """
    Upsample block with BN-based generator (PixelShuffle x2).
    """
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    return x


def build_srgan_generator(scale=4, num_res_blocks=16):
    """
    Baseline SRGAN generator (SRResNet) with BN in residual blocks.
    Output: tanh in [-1,1].
    """
    lr_input = Input(shape=(None, None, 3))

    x1 = layers.Conv2D(64, 9, padding='same')(lr_input)
    x1 = layers.PReLU(shared_axes=[1, 2])(x1)

    x = x1
    for _ in range(num_res_blocks):
        x = residual_block_bn(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Add()([x, x1])

    if scale >= 2:
        x = upsample_block_bn(x)
    if scale >= 4:
        x = upsample_block_bn(x)

    out = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
    return models.Model(lr_input, out, name="SRGAN_Generator")


def discriminator_block(x, filters, strides=1, batch_norm=True):
    x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def build_srgan_discriminator(input_shape):
    img_input = Input(shape=input_shape)

    x = discriminator_block(img_input, 64, strides=1, batch_norm=False)
    x = discriminator_block(x, 64, strides=2)
    x = discriminator_block(x, 128, strides=1)
    x = discriminator_block(x, 128, strides=2)
    x = discriminator_block(x, 256, strides=1)
    x = discriminator_block(x, 256, strides=2)
    x = discriminator_block(x, 512, strides=1)
    x = discriminator_block(x, 512, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    validity = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(img_input, validity, name="SRGAN_Discriminator")


def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    model = models.Model(
        inputs=vgg.inputs,
        outputs=vgg.get_layer("block5_conv4").output
    )
    model.trainable = False
    return model


def build_srgan_combined(generator, discriminator, vgg, lr_shape):
    discriminator.trainable = False
    vgg.trainable = False

    lr_input = Input(shape=lr_shape)
    sr = generator(lr_input)
    sr_features = vgg(sr)
    validity = discriminator(sr)

    model = models.Model(lr_input, [validity, sr_features])
    model.compile(
        loss=['binary_crossentropy', 'mse'],
        loss_weights=[config.SRGAN_ADV_LOSS_WEIGHT, config.SRGAN_CONTENT_LOSS_WEIGHT],
        optimizer=keras.optimizers.Adam(learning_rate=config.SRGAN_GEN_LEARNING_RATE),
    )
    return model


# ============================================================
# Attentive ESRGAN Models
# ============================================================

def channel_attention_block(x, ratio=16):
    channels = x.shape[-1]
    if channels is None:
        channels = 64  # safe fallback for this architecture

    se = layers.GlobalAveragePooling2D(keepdims=True)(x)
    reduced_channels = max(int(channels) // ratio, 1)
    se = layers.Dense(reduced_channels, activation='relu', use_bias=False)(se)
    se = layers.Dense(int(channels), activation='sigmoid', use_bias=False)(se)
    return layers.Multiply()([x, se])


class PixelShuffle(layers.Layer):
    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self.scale})
        return cfg


def attentive_residual_block(x):
    shortcut = x

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = channel_attention_block(x)

    # residual scaling 0.2
    x = x * 0.2

    return layers.Add()([x, shortcut])


def upsample_block_no_bn(x, scale=2):
    # 64 * scale^2 filters -> after pixel shuffle we still have 64 channels
    x = layers.Conv2D(64 * (scale ** 2), 3, padding='same')(x)
    x = PixelShuffle(scale)(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    return x


def build_attentive_generator(scale=4, num_res_blocks=16):
    lr_input = Input(shape=(None, None, 3))

    x1 = layers.Conv2D(64, 9, padding='same')(lr_input)
    x1 = layers.PReLU(shared_axes=[1, 2])(x1)

    x = x1
    for _ in range(num_res_blocks):
        x = attentive_residual_block(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.Add()([x, x1])

    if scale >= 2:
        x = upsample_block_no_bn(x, scale=2)
    if scale >= 4:
        x = upsample_block_no_bn(x, scale=2)

    out = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
    return models.Model(lr_input, out, name="Attentive_Generator")


def build_relativistic_discriminator(input_shape):
    img_input = Input(shape=input_shape)

    def d_block(x, filters, strides=1, bn=True):
        x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x

    x = d_block(img_input, 64, strides=1, bn=False)
    x = d_block(x, 64, strides=2)
    x = d_block(x, 128, strides=1)
    x = d_block(x, 128, strides=2)
    x = d_block(x, 256, strides=1)
    x = d_block(x, 256, strides=2)
    x = d_block(x, 512, strides=1)
    x = d_block(x, 512, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    validity = layers.Dense(1)(x)  # logits
    return models.Model(img_input, validity, name="Relativistic_Discriminator")


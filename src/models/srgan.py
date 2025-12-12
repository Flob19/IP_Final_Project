"""
SRGAN Baseline Models
- Generator: SRResNet with BatchNorm
- Discriminator: CNN with BatchNorm
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG19
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class SRGANGenerator:
    """
    Baseline SRGAN generator (SRResNet) with BN in residual blocks.
    Output: tanh in [-1,1].
    """
    
    def __init__(self, scale=4, num_res_blocks=16):
        self.scale = scale
        self.num_res_blocks = num_res_blocks
        self.model = self._build_model()
    
    def _residual_block_bn(self, x):
        """Residual block with BatchNorm (original SRGAN)."""
        shortcut = x
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        return layers.Add()([x, shortcut])
    
    def _upsample_block_bn(self, x):
        """Upsample block with BN-based generator (PixelShuffle x2)."""
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        return x
    
    def _build_model(self):
        """Build SRGAN generator model."""
        lr_input = Input(shape=(None, None, 3))

        x1 = layers.Conv2D(64, 9, padding='same')(lr_input)
        x1 = layers.PReLU(shared_axes=[1, 2])(x1)

        x = x1
        for _ in range(self.num_res_blocks):
            x = self._residual_block_bn(x)

        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Add()([x, x1])

        if self.scale >= 2:
            x = self._upsample_block_bn(x)
        if self.scale >= 4:
            x = self._upsample_block_bn(x)

        out = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
        return models.Model(lr_input, out, name="SRGAN_Generator")
    
    def get_model(self):
        """Get the model."""
        return self.model
    
    def load_weights(self, path):
        """Load pre-trained weights."""
        self.model.load_weights(path)
    
    def save(self, path):
        """Save the model."""
        self.model.save(path)


class SRGANDiscriminator:
    """
    SRGAN Discriminator: CNN with BatchNorm.
    """
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _discriminator_block(self, x, filters, strides=1, batch_norm=True):
        """Discriminator block."""
        x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x
    
    def _build_model(self):
        """Build SRGAN discriminator model."""
        img_input = Input(shape=self.input_shape)

        x = self._discriminator_block(img_input, 64, strides=1, batch_norm=False)
        x = self._discriminator_block(x, 64, strides=2)
        x = self._discriminator_block(x, 128, strides=1)
        x = self._discriminator_block(x, 128, strides=2)
        x = self._discriminator_block(x, 256, strides=1)
        x = self._discriminator_block(x, 256, strides=2)
        x = self._discriminator_block(x, 512, strides=1)
        x = self._discriminator_block(x, 512, strides=2)

        x = layers.Flatten()(x)
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        validity = layers.Dense(1, activation='sigmoid')(x)
        return models.Model(img_input, validity, name="SRGAN_Discriminator")
    
    def get_model(self):
        """Get the model."""
        return self.model
    
    def compile_model(self):
        """Compile the discriminator."""
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=config.SRGAN_DISC_LEARNING_RATE),
            metrics=['accuracy'],
        )
    
    def load_weights(self, path):
        """Load pre-trained weights."""
        self.model.load_weights(path)
    
    def save(self, path):
        """Save the model."""
        self.model.save(path)


def build_vgg(hr_shape):
    """Build VGG19 feature extractor for perceptual loss."""
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    model = models.Model(
        inputs=vgg.inputs,
        outputs=vgg.get_layer("block5_conv4").output
    )
    model.trainable = False
    return model


def build_srgan_combined(generator, discriminator, vgg, lr_shape):
    """Build combined SRGAN model for training generator."""
    # Freeze D + VGG for combined model
    discriminator.model.trainable = False
    for l in discriminator.model.layers: 
        l.trainable = False

    vgg.trainable = False
    for l in vgg.layers:
        l.trainable = False

    lr_input = Input(shape=lr_shape)
    sr = generator.model(lr_input)
    sr_features = vgg(sr)
    validity = discriminator.model(sr)

    model = models.Model(lr_input, [validity, sr_features])
    model.compile(
        loss=['binary_crossentropy', 'mse'],
        loss_weights=[config.SRGAN_ADV_LOSS_WEIGHT, config.SRGAN_CONTENT_LOSS_WEIGHT],
        optimizer=keras.optimizers.Adam(learning_rate=config.SRGAN_GEN_LEARNING_RATE),
    )

    # IMPORTANT: Re-enable D so you can still train D separately
    discriminator.model.trainable = True
    for l in discriminator.model.layers:
        l.trainable = True

    return model


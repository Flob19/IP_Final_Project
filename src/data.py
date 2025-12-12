"""
Data pipeline for Super-Resolution GAN training.
"""

import os
import math
import random
import cv2
import numpy as np
from tensorflow import keras


class SRGANDataGenerator(keras.utils.Sequence):
    """
    Custom Data Generator for Super Resolution (SRGAN / ESRGAN).
    - Loads HR images from a folder
    - Random crops HR patches of size HR_CROP_SIZE x HR_CROP_SIZE
    - Generates LR patches via bicubic downsampling
    - Normalizes both LR & HR to [-1, 1]
    """

    def __init__(self, hr_dir, batch_size=16, crop_size=128, scale_factor=4, shuffle=True):
        self.hr_dir = hr_dir
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.shuffle = shuffle

        try:
            self.image_files = sorted(
                f for f in os.listdir(hr_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            print(f"[SRGANDataGenerator] {len(self.image_files)} images found in: {hr_dir}")
        except FileNotFoundError:
            print(f"[SRGANDataGenerator] ERROR: Directory not found: {hr_dir}")
            self.image_files = []

        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        if not self.image_files:
            return 0
        return math.ceil(len(self.image_files) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start:end]
        batch_files = [self.image_files[i] for i in batch_indexes]

        lr_batch, hr_batch = [], []

        for fname in batch_files:
            img_path = os.path.join(self.hr_dir, fname)
            try:
                hr_img = cv2.imread(img_path)
                if hr_img is None:
                    continue
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

                h, w, _ = hr_img.shape
                if h < self.crop_size or w < self.crop_size:
                    continue

                # Random HR crop
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
                hr_patch = hr_img[y:y + self.crop_size, x:x + self.crop_size]

                # LR via bicubic downsampling
                lr_size = (self.crop_size // self.scale_factor,
                           self.crop_size // self.scale_factor)
                lr_patch = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_CUBIC)

                # Normalize to [-1, 1]
                lr_batch.append(lr_patch / 127.5 - 1.0)
                hr_batch.append(hr_patch / 127.5 - 1.0)

            except Exception as e:
                print(f"[SRGANDataGenerator] Error processing {fname}: {e}")
                continue

        if len(lr_batch) == 0:
            # fallback to next batch if something went wrong
            return self.__getitem__((index + 1) % self.__len__())

        return np.array(lr_batch), np.array(hr_batch)


def resolve_div2k_paths(root):
    """
    Handle possible nested structures in Kaggle DIV2K dataset.
    Returns (train_hr_dir, valid_hr_dir).
    """
    train = os.path.join(root, "DIV2K_train_HR", "DIV2K_train_HR")
    valid = os.path.join(root, "DIV2K_valid_HR", "DIV2K_valid_HR")
    if not os.path.exists(train):
        train = os.path.join(root, "DIV2K_train_HR")
        valid = os.path.join(root, "DIV2K_valid_HR")
    return train, valid


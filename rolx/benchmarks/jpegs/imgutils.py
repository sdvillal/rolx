from __future__ import print_function, division
import cv2
import numpy as np


def phash_compare(img1, img2):
    # Image similarity via hashing; opencv seems to have charged batteries
    # http://qtandopencv.blogspot.de/2016/06/introduction-to-image-hash-module-of.html
    hasher = cv2.img_hash_PHash.create()
    return hasher.compare(hasher.compute(img1), hasher.compute(img2))


def psnr(img1, img2, pixel_max=255):
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(pixel_max ** 2 / mse)


# Also go for other stuff like SSIM (remember I implemented that somewhere in fisherman)
# First result in google, a few variants: https://github.com/aizvorski/video-quality

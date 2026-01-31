"""IQA measures for blur analysis"""

from skimage.measure import blur_effect
from classiqa.base import BaseModel
import cv2
import numpy as np


class BlurEffect(BaseModel):
    """A wrapper for Scikit-image's blur_effect

    See: https://scikit-image.org/docs/stable/auto_examples/filters/plot_blur_effect.html
    """

    def __init__(
        self,
        img_size=512,
        h_size=11,
    ):
        super().__init__(img_size, 1)
        self.h_size = h_size

    def extract_features(self, x_gray):
        """Computes the Blur Effect measure"""
        features = [blur_effect(x_gray, h_size=self.h_size)]
        return features


class VarianceOfLaplacian(BaseModel):
    """
    See:
        https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        https://ieeexplore.ieee.org/document/903548
    """

    def __init__(
        self,
        img_size=512,
    ):
        super().__init__(img_size, 1)

    def extract_features(self, x_gray):
        """Computes the Blur Effect measure"""
        features = [cv2.Laplacian(x_gray, cv2.CV_64F).var()]
        return features


class NandaCutlerContrast(BaseModel):
    """
    From "Practical calibrations for a real-time digital omnidirectional camera" (Nanda and Cutler, 2001)
    https://www.researchgate.net/profile/Ross-Cutler/publication/228952354_Practical_calibrations_for_a_real-time_digital_omnidirectional_camera/links/09e4150bc3a55d3861000000/Practical-calibrations-for-a-real-time-digital-omnidirectional-camera.pdf

    It is also explained here:
    "Analysis of focus measure operators for shape-from-focus"
    https://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf
    """

    def __init__(
        self,
        img_size=512,
    ):
        super().__init__(img_size, 1)

        self.kernels = []
        for i in range(9):
            k = np.zeros(9, dtype=float)
            if i != 4:
                k[i] = -1
                k[4] = 1
                k = k.reshape((3, 3))
                self.kernels.append(k)

    def extract_features(self, x_gray):
        """Computes the contrast measure"""
        diffs = np.stack([cv2.filter2D(x_gray, -1, k) for k in self.kernels])
        diffs = np.abs(diffs).sum(axis=0)
        features = [diffs.sum()]
        return features


# TODO: Implement the S3 measure:
# Use this repo: https://github.com/Xiaoming-Zhao/s3_sharpness_measure

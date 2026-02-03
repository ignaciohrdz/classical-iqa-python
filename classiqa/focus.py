"""IQA measures for blur analysis"""

from skimage.measure import blur_effect
from classiqa.model import BaseModel, PatchModel
import cv2
import numpy as np
import torch


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


class BrennerFocus(PatchModel):
    def __init__(self, img_size, n_features, patch_size=11):
        super().__init__(img_size, n_features, patch_size)
        self.unfold = torch.nn.Unfold(patch_size, stride=patch_size)
        self.filter = np.array([[0, 0, -1, 0, 1]], dtype=float)

    def extract_features(self, x_gray):
        """Computes the ratio between the intensity level of every pixel
        and the mean gray level of its neighbourhood"""
        diff = cv2.filter2D(x_gray, -1, self.filter) ** 2

        # Dividing the image in tiles and computing the median value in each tile
        # then aggregate all values
        t = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0).float()
        t = self.unfold(t).permute(0, 2, 1).squeeze()
        ratio = np.mean(t.numpy(), -1)
        features = [ratio.sum()]
        return features


class NandaCutlerContrast(BaseModel):
    """
    From "Practical calibrations for a real-time digital omnidirectional camera" (Nanda and Cutler, 2001)
    https://www.researchgate.net/profile/Ross-Cutler/publication/228952354_Practical_calibrations_for_a_real-time_digital_omnidirectional_camera/links/09e4150bc3a55d3861000000/Practical-calibrations-for-a-real-time-digital-omnidirectional-camera.pdf
    """

    def __init__(self, img_size=512):
        super().__init__(img_size, 1)

        self.filters = []
        for i in range(9):
            f = np.zeros(9, dtype=float)
            if i != 4:
                f[i] = -1
                f[4] = 1
                f = f.reshape((3, 3))
                self.filters.append(f)

    def extract_features(self, x_gray):
        """Computes the contrast measure"""
        diffs = np.stack([cv2.filter2D(x_gray, -1, f) for f in self.filters])
        diffs = np.abs(diffs).sum(axis=0)
        features = [diffs.sum()]
        return features


class MeanMethodFocus(PatchModel):
    """
    From "Adaptive shape from focus with an error estimation in light microscopy" (Helmli and Scherer, 2001)
    https://ieeexplore.ieee.org/document/938626

    This measure was meant to be computed in a specific region of the image. If we compute it
    for the whole image, we would get huge values. That's why I'm dividing the image into tiles,
    compute the mean of each tile and then aggregate the results.
    """

    def __init__(self, img_size=512, eps=1e-4, patch_size=11):
        super().__init__(img_size, 1, patch_size)
        self.filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float) / 8
        self.eps = eps
        self.unfold = torch.nn.Unfold(patch_size, stride=patch_size)

    def extract_features(self, x_gray):
        """Computes the ratio between the intensity level of every pixel
        and the mean gray level of its neighbourhood"""
        x = x_gray / 255
        means = cv2.filter2D(x, -1, self.filter)
        ratio_1 = means / (x + self.eps)
        ratio_2 = 1 / ratio_1
        mask_1 = ratio_1 >= 1
        mask_2 = np.bitwise_not(mask_1)
        ratio = (ratio_1 * mask_1) + (ratio_2 * mask_2)

        # Dividing the image in tiles and computing the median value in each tile
        # then aggregate all values
        t = torch.from_numpy(ratio).unsqueeze(0).unsqueeze(0).float()
        t = self.unfold(t).permute(0, 2, 1).squeeze()
        ratio = np.mean(t.numpy(), -1)
        features = [ratio.sum()]
        return features


# TODO: Implement the S3 measure:
# Use this repo: https://github.com/Xiaoming-Zhao/s3_sharpness_measure


focus_models_dict = {
    "blur_effect": BlurEffect,
    "var_lap": VarianceOfLaplacian,
    "brenner": BrennerFocus,
    "nanda_cutler": NandaCutlerContrast,
    "mean_method": MeanMethodFocus,
}

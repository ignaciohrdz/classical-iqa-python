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


class DeltaDifferences(BaseModel):
    """From 'Sharpness Estimation for Document and Scene Images' (Kumar et al., 2012)
    See:
        https://ieeexplore.ieee.org/document/6460868
        https://github.com/umang-singhal/pydom
    """

    def __init__(
        self,
        img_size,
        median_size=3,
        window_size=5,
        edge_thresh=1e-4,
        sharp_thresh=2,
        eps=1e-4,
        maxdom=False,
    ):
        super().__init__(img_size, 1)
        self.median_size = median_size
        self.window_size = window_size
        self.edge_thresh = edge_thresh
        self.sharp_thresh = sharp_thresh
        self.eps = eps
        self.maxdom = maxdom

        # DoM = [Im(i+2,j) - Im(i,j)] - [Im(i,j) - Im(i-2,j)]
        #     = Im(i+2,j) - 2xIm(i,j) + Im(i-2, j)
        self.dom_filter = np.array([[1, 0, -2, 0, 1]], dtype=float)
        self.contrast_filter = np.array([[-1, 1, 0]])
        self.window_sum_filter = np.ones((window_size, window_size), dtype=float)
        self.smoothing_filter = np.array([[0.5, 0, -0.5]], dtype=float)

    def extract_features(self, x_gray):

        # Median filtering is used to reduce noise
        x_median = cv2.medianBlur(x_gray, self.median_size) / 255.0

        # Difference of differences
        dom_x = np.abs(cv2.filter2D(x_median, -1, self.dom_filter))
        dom_y = np.abs(cv2.filter2D(x_median, -1, self.dom_filter.T))

        # Contrast (used for normalisation)
        contrast = np.abs(cv2.filter2D(x_gray, -1, self.contrast_filter)) / 255.0
        contrast_sum = cv2.filter2D(contrast, -1, self.window_sum_filter)
        contrast_sum += 1 / 255  # to prevent NaNs

        # Finding the edges
        edges_x = cv2.filter2D(x_gray, -1, self.smoothing_filter)
        edges_y = cv2.filter2D(x_gray, -1, self.smoothing_filter.T)
        edges_x = edges_x / (edges_x.max() + self.eps)
        edges_y = edges_y / (edges_y.max() + self.eps)

        s = 0
        for dom_i, edge_i in zip([dom_x, dom_y], [edges_x, edges_y]):
            # Aggregate the DOM values
            if self.maxdom:
                dom_i_agg = cv2.dilate(dom_i, self.window_sum_filter)
            else:
                dom_i_agg = cv2.filter2D(dom_i, -1, self.window_sum_filter)

            # Normalisation
            sharpness_i = dom_i_agg / contrast_sum

            # cv2.imshow("Sharpness", sharpness_i)
            # cv2.imshow("Edge", edge_i)
            # cv2.waitKey()

            is_edge = (edge_i >= self.edge_thresh).astype(float)
            is_sharp = (sharpness_i >= self.sharp_thresh).astype(float)
            r_i = (is_sharp * is_edge).sum() / is_edge.sum()
            s += r_i**2
        s = np.sqrt(s)
        features = [s / np.sqrt(2)]

        return features


# TODO: Implement the S3 measure:
# Use this repo: https://github.com/Xiaoming-Zhao/s3_sharpness_measure


focus_models_dict = {
    "blur_effect": BlurEffect,
    "var_lap": VarianceOfLaplacian,
    "brenner": BrennerFocus,
    "nanda_cutler": NandaCutlerContrast,
    "mean_method": MeanMethodFocus,
    "dom": DeltaDifferences,
}

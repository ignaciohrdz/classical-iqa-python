"""IQA measures for blur analysis"""

from skimage.measure import blur_effect
from classiqa.base import BaseModel


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

"""IQA models based on Natural Scene Statistics (NSS)"""

from classiqa.model import BaseModel
import cv2
import numpy as np
import brisque
import skimage


class GMLOG(BaseModel):
    """Our implementation of 'Blind Image Quality Assessment Using Joint
    Statistics of Gradient Magnitude and Laplacian Features'. This is the M3 model,
    which combines marginal distributions and dependency measures

    See: https://ieeexplore.ieee.org/document/6894197
    """

    def __init__(
        self,
        img_size=512,
        sigma=0.5,
        bins_gm=10,
        bins_log=10,
        epsilon=0.25,
        alpha=0.0001,
    ):
        n_features = 2 * (bins_gm + bins_log)
        super().__init__(img_size, n_features)

        self.sigma = sigma
        self.bins_gm = bins_gm
        self.bins_log = bins_log
        self.epsilon = epsilon
        self.alpha = alpha

        # Using the author's settings
        self.kernel_size = int(2 * np.ceil(3 * self.sigma) + 1 + 2)

        # Kernel for Joint Adaptive Normalization (JAN)
        self.sigma_jan = 2 * self.sigma
        self.kernel_size_jan = int(2 * np.ceil(3 * self.sigma_jan) + 1)

    def extract_features(self, x_gray):
        """Computes the GM LOG features"""

        # Computation of Gradient Magnitude and Laplacian of Gaussian channels
        # (the Sobel Operator is an approximation of Gaussian smoothing + differentiation)
        grad_x = cv2.Sobel(x_gray, cv2.CV_64F, 1, 0, self.kernel_size)
        grad_y = cv2.Sobel(x_gray, cv2.CV_64F, 0, 1, self.kernel_size)
        gm_i = cv2.magnitude(grad_x, grad_y)

        l_i = cv2.GaussianBlur(
            x_gray,
            ksize=(self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )
        l_i = cv2.Laplacian(l_i, cv2.CV_64F, ksize=self.kernel_size)
        l_i = np.absolute(l_i)

        # Joint Adaptive Normalisation (JAN)
        f_i = np.sqrt(gm_i**2 + l_i**2)
        n_i = np.sqrt(
            cv2.GaussianBlur(
                f_i**2,
                ksize=(self.kernel_size_jan, self.kernel_size_jan),
                sigmaX=self.sigma_jan,
                sigmaY=self.sigma_jan,
            )
        )
        gm_i = gm_i / (n_i + self.epsilon)
        l_i = l_i / (n_i + self.epsilon)

        # cv2.imshow("Image", x_gray)
        # cv2.imshow("Gradient magnitude", gm_i)
        # cv2.imshow("Laplacian of Gaussian", l_i)
        # cv2.waitKey()

        # Statitical feature description
        # Normalised histogram
        k, _, _ = np.histogram2d(
            gm_i.flatten(),
            l_i.flatten(),
            bins=[self.bins_gm, self.bins_log],
        )
        k /= k.sum()

        # Marginal distributions
        n_g = np.sum(k, axis=1)
        n_l = np.sum(k, axis=0)

        # Independency distributions (eq. 14 and 15 in the paper)
        q_g = (k / (n_l + self.alpha)).mean(axis=1)
        q_l = (k / (n_g + self.alpha)).mean(axis=0)

        features = n_g.tolist() + n_l.tolist() + q_g.tolist() + q_l.tolist()
        return features


class BRISQUE(BaseModel):
    """A wrapper for the BRISQUE model from:
    https://github.com/rehanguha/brisque
    """

    def __init__(self, img_size):
        super().__init__(img_size, 36)  # BRISQUE has 36 features
        self.model = brisque.BRISQUE(url=False)

    def extract_features(self, x_gray):
        x_downscaled = cv2.resize(
            x_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
        )
        features = []
        for x in [x_gray, x_downscaled]:
            scale_ftrs = self.model.calculate_brisque_features(
                x, kernel_size=7, sigma=7 / 6
            )
            features.append(scale_ftrs)
        features = np.concatenate(features)

        return features


nss_model_dict = {"gmlog": GMLOG, "brisque": BRISQUE}

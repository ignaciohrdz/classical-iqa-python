import cv2
import numpy as np
import pandas as pd
import random
from .data import split_dataset
import pickle

random.seed(420)


class GMLOG:
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

        self.img_size = img_size
        self.sigma = sigma
        self.bins_gm = bins_gm
        self.bins_log = bins_log
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_features = 2 * (bins_gm + bins_log)

        # Using the author's settings
        self.kernel_size = int(2 * np.ceil(3 * self.sigma) + 1 + 2)

        # Kernel for Joint Adaptive Normalization (JAN)
        self.sigma_jan = 2 * self.sigma
        self.kernel_size_jan = int(2 * np.ceil(3 * self.sigma_jan) + 1)

    def prepare_input(self, x):
        """Initial conversion to grayscale and resizing"""

        x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        if self.img_size > 0:
            ratio = self.img_size / max(x_gray.shape)
            x_gray = cv2.resize(
                x_gray,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_CUBIC,
            )
        return x_gray

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

    def generate_feature_db(self, dset, test_size=0.3):
        """Creates the feature database that will be used to fit the SVR
        :param dset: a DataFrame with columns [image_name, image_path, score, [img_set]]
                    (not all datasets have the img_set columns, only those that contain
                      groups of distorted images created from the same pristine source)
        """

        # Creating the train/test splits
        if "is_test" not in dset.columns:
            dset = split_dataset(dset, test_size)

        feature_db = []
        for i, row in enumerate(dset.to_dict("records")):
            im_name = row["image_name"]
            im_path = row["image_path"]
            im_set = row["image_set"]
            im_score = row["score"]
            im_split = row["is_test"]
            print(f"[{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)
            img_gray = self.prepare_input(img)
            ftrs = list(self.extract_features(img_gray))
            feature_db.append([im_name] + ftrs + [im_score, im_split, im_set])

        feature_cols = list(range(1, self.n_features + 1))
        db_cols = ["image_name"] + feature_cols + ["MOS", "is_test", "image_set"]
        feature_db = pd.DataFrame(feature_db, columns=db_cols)

        return feature_db

    def __call__(self, x):
        x_gray = self.prepare_input(x)
        fts = self.extract_features(x_gray)
        features = np.array(fts)

        return features

    def export(self, path_save):
        path_pkl = path_save / "feature_extractor.pkl"
        print("Saving feature extractor to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

import cv2
import numpy as np
import pandas as pd
import random
from .data import split_dataset
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from .metrics import lcc, srocc
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
        test_size=0.3,
        svr_regressor=None,
    ):

        self.img_size = img_size
        self.sigma = sigma
        self.bins_gm = bins_gm
        self.bins_log = bins_log
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_features = 2 * (bins_gm + bins_log)
        self.test_size = test_size
        self.svr_regressor = svr_regressor

        # Using the author's settings
        self.kernel_size = int(2 * np.ceil(3 * self.sigma) + 1)
        self.gaussian_kernel = self.make_gaussian_kernel(
            self.kernel_size, self.sigma * 2
        )
        self.gaussian_deriv_x, self.gaussian_deriv_y = (
            self.make_gaussian_first_derivative(self.kernel_size, self.sigma)
        )
        self.log_kernel = self.make_laplacian_of_gaussian(self.kernel_size, self.sigma)

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

    def make_gaussian_kernel(self, n, sigma):
        """Creates a normalised Gaussian kernel (as in the paper)"""
        Y, X = np.indices((n, n)) - int(n / 2)
        gaussian_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
            -(X**2 + Y**2) / (2 * sigma**2)
        )
        gaussian_kernel = gaussian_kernel / np.sum(np.abs(gaussian_kernel))
        return gaussian_kernel

    def make_gaussian_first_derivative(self, n, sigma):
        """Creates the Gaussian partial derivative kernels (as in the paper)"""
        Y, X = np.indices((n, n)) - int(n / 2)
        gaussian_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
            -(X**2 + Y**2) / (2 * sigma**2)
        )

        # In the paper the kernels are not normalised)
        vertical_kernel = gaussian_kernel * (-X / sigma**2)
        horizontal_kernel = gaussian_kernel * (-Y / sigma**2)
        # vertical_kernel = vertical_kernel / np.sum(np.abs(vertical_kernel))
        # horizontal_kernel = horizontal_kernel / np.sum(np.abs(horizontal_kernel))

        return vertical_kernel, horizontal_kernel

    def make_laplacian_of_gaussian(self, n, sigma):
        """Creates the Laplacian of Gaussian kernel (as in the paper)"""
        Y, X = np.indices((n, n)) - int(n / 2)
        log_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
            -(X**2 + Y**2) / (2 * sigma**2)
        )

        # In the paper the kernel is not normalised
        log_kernel *= ((X**2) + (Y**2) - (2 * sigma**2)) / sigma**4
        # log_kernel /= np.sum(np.abs(log_kernel))

        return log_kernel

    def extract_features(self, x_gray):
        """Computes the GM LOG features"""

        # Computation of Gradient Magnitude and Laplacian of Gaussian channels
        grad_x = cv2.filter2D(src=x_gray, ddepth=-1, kernel=self.gaussian_deriv_x)
        grad_y = cv2.filter2D(src=x_gray, ddepth=-1, kernel=self.gaussian_deriv_y)
        g_i = np.sqrt(grad_x**2 + grad_y**2)
        l_i = np.abs(cv2.filter2D(src=x_gray, ddepth=-1, kernel=self.log_kernel))

        # Joint Adaptive Normalisation (JAN)
        f_i = np.sqrt(g_i**2 + l_i**2)
        n_i = np.sqrt(
            cv2.filter2D(
                src=(f_i**2).astype(np.uint8),
                ddepth=-1,
                kernel=self.gaussian_kernel,
            )
        )
        g_i = g_i / (n_i + self.epsilon)
        l_i = l_i / (n_i + self.epsilon)

        # Statitical feature description
        # Normalised histogram
        k, _, _ = np.histogram2d(
            g_i.flatten(),
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

    def generate_feature_db(self, dset):
        """Creates the feature database that will be used to fit the SVR
        :param dset: a DataFrame with columns [image_name, image_path, score, [img_set]]
                    (not all datasets have the img_set columns, only those that contain
                      groups of distorted images created from the same pristine source)
        """

        # Creating the train/test splits
        if "is_test" not in dset.columns:
            dset = split_dataset(dset, self.test_size)

        feature_db = []
        for i, row in enumerate(dset.to_dict("records")):
            im_name = row["image_name"]
            im_path = row["image_path"]
            im_score = row["score"]
            im_split = row["is_test"]
            print(f"[{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)
            img_gray = self.prepare_input(img)
            ftrs = list(self.extract_features(img_gray))
            feature_db.append([im_name] + ftrs + [im_score, im_split])

        feature_cols = list(range(1, self.n_features + 1))
        db_cols = ["image_name"] + feature_cols + ["MOS", "is_test"]
        feature_db = pd.DataFrame(feature_db, columns=db_cols)

        return feature_db

    def fit(self, feature_db, n_jobs=4):
        """
        Fit an SVR model to a given dset of ftrs
        :param feature_db: dataframe with 43 columns:
                        - image name
                        - 40 ftrs
                        - MOS
                        - split
        :param n_jobs: number of threads for GridSearchCV
        """

        train_mask = feature_db["is_test"] == 0
        test_mask = feature_db["is_test"] == 1
        feature_cols = feature_db.columns[1:-2]

        X_train = feature_db.loc[train_mask, feature_cols].values
        y_train = feature_db.loc[train_mask, "MOS"].values

        X_test = feature_db.loc[test_mask, feature_cols].values
        y_test = feature_db.loc[test_mask, "MOS"].values

        params = {
            "svr__C": np.arange(1.0, 10, 0.5),
            "svr__epsilon": np.arange(0.1, 2.0, 0.1),
        }

        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), SVR()),
            param_grid=params,
            cv=5,
            n_jobs=n_jobs,
            verbose=1,
            scoring={"LCC": make_scorer(lcc), "SROCC": make_scorer(srocc)},
            error_score=0,
            refit="SROCC",
        )

        print("Fitting an SVR for GM-LOG features")
        search.fit(X_train, y_train)
        self.svr_regressor = search.best_estimator_
        print(self.svr_regressor[1].C, self.svr_regressor[1].epsilon)

        # Test metrics
        y_pred = self.svr_regressor.predict(X_test)
        self.test_results = {
            "LCC": lcc(y_test, y_pred),
            "SROCC": srocc(y_test, y_pred),
        }

        return search.cv_results_

    def __call__(self, x):
        x_gray = self.prepare_input(x)
        fts = self.extract_features(x_gray)
        features = np.array(fts)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(features.reshape(1, -1))
        else:
            return features

    def export(self, path_save):
        path_pkl = path_save / "estimator.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_score(self, f):
        """Predicts the score from a set of features (f)"""
        score = self.svr_regressor.predict(f)
        return score

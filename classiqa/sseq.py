import cv2
import torch
from torch import nn
import numpy as np
import pandas as pd
from scipy.stats import skew
from .metrics import lcc, srocc
from .data import split_dataset

from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


class SSEQ:
    """Spatial-Spectral Entropy-based Quality (SSEQ) index (Liu et al.)"""

    def __init__(
        self,
        block_size=8,
        img_size=-1,
        percentile=0.6,
        scales=3,
        eps=1e-5,
        feature_db=None,
        svr_regressor=None,
        test_size=0.3,
    ):
        self.block_size = block_size
        self.img_size = img_size
        self.percentile = percentile
        self.scales = scales
        self.eps = eps
        self.unfold = nn.Unfold(kernel_size=self.block_size, stride=self.block_size)
        self.svr_regressor = svr_regressor
        self.test_size = test_size
        self.test_results = {"LCC": 0.0, "SROCC": 0.0}
        self.n_features = self.scales * 4

        self.m = self.make_dct_matrix()
        self.m_t = self.m.T

    def prepare_input(self, x):
        """Initial conversion to grayscale and resizing"""

        x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x_gray = self.crop_input(x_gray)
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

    def __call__(self, x):
        # Initial resizing
        x_gray = self.prepare_input(x)

        # Extracting the ftrs at different scales
        ftrs = self.extract_features(x_gray)

        # If we have loaded a SVR model, we predict the IQA score
        # The ftrs are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(ftrs.reshape(1, -1))
        else:
            return ftrs

    def extract_features(self, x_gray):

        x_all_scales = [x_gray]
        for s in range(1, self.scales):
            ratio = 0.5**s
            x_scale = cv2.resize(
                x_gray,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_CUBIC,
            )
            x_all_scales.append(x_scale)

        spatial_features = []
        spectral_features = []
        for x in x_all_scales:
            # Using Pytorch for extracting local image patches
            t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
            t = self.unfold(t).permute(0, 2, 1).squeeze()
            t = t.view(t.shape[0], self.block_size, self.block_size)

            # Spatial entropy
            # In order to compute it faster, I will use offsetting
            #  instead of computing row-wise entropy
            # https://discuss.pytorch.org/t/count-number-occurrence-of-value-per-row/137061/5
            t_flat = t.reshape(t.shape[0], -1).int()
            min_length = 256 * t_flat.shape[0]
            t_flat_offset = t_flat + 256 * torch.arange(t_flat.shape[0]).unsqueeze(1)
            counts = torch.bincount(
                t_flat_offset.flatten(), minlength=min_length
            ).reshape(t_flat.shape[0], 256)
            mask = (counts > 0).float()
            p = counts / counts.sum(dim=1).unsqueeze(1)
            log_p = torch.log2(p).nan_to_num(posinf=0.0, neginf=0.0)
            se = np.sort(-1 * ((p * log_p * mask).sum(dim=1)).numpy())
            se_pooled = self.percentile_pooling(se)
            spatial_features.extend([se_pooled.mean(), skew(se)])

            # Spectral entropy
            m = torch.unsqueeze(torch.tensor(self.m), 0).repeat(t.shape[0], 1, 1)
            m_t = torch.unsqueeze(torch.tensor(self.m_t), 0).repeat(t.shape[0], 1, 1)
            t_dct = torch.bmm(torch.bmm(m, t), m_t)
            t_dct[:, 0, 0] = self.eps  # discarding the DC component
            p_sum = (t_dct**2).sum(axis=(1, 2)).unsqueeze(1).unsqueeze(1)
            p_i = (t_dct**2) / p_sum  # normalized spectral probability maps
            p_i[p_i == 0] = self.eps  # prevent NaNs
            fe = np.sort((p_i * torch.log2(p_i)).sum(axis=(1, 2)).numpy())  # entropy
            fe_pooled = self.percentile_pooling(fe)
            spectral_features.extend([fe_pooled.mean(), skew(fe)])

        # Float32 is more memory-efficient
        features = spatial_features + spectral_features
        features = np.array(features, dtype=np.float32)

        return features

    def crop_input(self, x):
        """We make sure the image is divisible into NxN tiles (N = block_size)
        If the image is not divisible, we crop it start from the top-left corner"""
        h, w = x.shape
        h_cropped = h - (h % self.block_size)
        w_cropped = w - (w % self.block_size)
        return x[:h_cropped, :w_cropped]

    def make_dct_matrix(self):
        """DCT can be computed as a matrix multiplication"""
        m = np.zeros((self.block_size, self.block_size), dtype=np.float32)

        m[0, :] = np.sqrt(1 / self.block_size)
        for row in range(1, self.block_size):
            for col in range(self.block_size):
                k = np.sqrt(2 / self.block_size)
                m[row, col] = k * (
                    np.cos((np.pi * (2 * col + 1) * row) / (2 * self.block_size))
                )

        return m

    def percentile_pooling(self, x):
        """Percentile pooling, as explained in the paper"""
        x_size = len(x)
        start = int(x_size * 0.5 * (1 - self.percentile))
        end = int(x_size - x_size * 0.5 * (1 - self.percentile))
        return x[start:end]

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
        :param feature_db: dataframe with 14 columns:
                        - image name
                        - 12 ftrs
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

        # X_train = np.float16(X_train)
        # X_test = np.float16(X_test)

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

        print("Fitting an SVR for SSEQ features")
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

    def predict_score(self, f):
        """Predicts the score from a set of features (f)"""
        score = self.svr_regressor.predict(f)
        return score

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
import random
from .data import split_dataset
from pathlib import Path
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from .metrics import lcc, srocc

random.seed(420)


class HOSA:
    """Blind Image Quality Assessment Based on High Order Statistics Aggregation
    by Xu et al. (https://ieeexplore.ieee.org/document/7501619)"""

    def __init__(
        self,
        patch_size=7,
        codebook_size=100,
        reduce_local=-1,
        r=5,
        alpha=0.2,
        beta=0.05,
        img_size=512,
        eps=10,
        svr_regressor=None,
        test_size=0.3,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        # Parameters related to the K nearest codewords
        self.codebook = None
        self.codebook_stats = None
        self.codebook_size = codebook_size
        self.reduce_local = reduce_local
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.n_dims = self.patch_size**2

        # For normalization
        self.eps = eps

        self.n_features = 3 * self.n_dims * self.codebook_size
        self.svr_regressor = svr_regressor
        self.test_size = test_size

    def crop_input(self, x):
        """We make sure the image is divisible into NxN tiles (N = patch_size)
        If the image is not divisible, we crop it
        starting from the top-left corner"""
        h, w = x.shape
        h_cropped = h - (h % self.patch_size)
        w_cropped = w - (w % self.patch_size)
        return x[:h_cropped, :w_cropped]

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
        x_gray = self.prepare_input(x)
        fts = self.extract_features(x_gray)

        features = np.array(fts)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(features.reshape(1, -1))
        else:
            return features

    def zca_whitening(self, x):
        """
        Computes ZCA whitening matrix (aka Mahalanobis whitening).
        Source: https://stackoverflow.com/a/38590790
        :param x: [m x n] matrix (feature_length x n_samples)
        :returns zca_mat: [m x m] matrix
        """
        # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
        sigma = np.cov(x.T, rowvar=True)  # [M x M]

        # Singular Value Decomposition. X = U * np.diag(S) * V
        # U: eigenvectors
        # S: eigenvalues
        U, S, _ = np.linalg.svd(sigma)

        epsilon = 1e-5  # prevents division by zero

        zca_mat = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
        x_white = np.dot(zca_mat, x.T)

        # Float16 is more memory-efficient
        x_white = np.float16(x_white)

        return x_white.T

    def extract_local_features(self, x_gray):
        """Extracts the local features of an image
        :param x_gray: grayscale image
        :returns local_ftrs: the local features
        """
        # Using Pytorch for extracting local image patches
        t = torch.from_numpy(x_gray).unsqueeze(0).unsqueeze(0).float()
        t = self.unfold(t).permute(0, 2, 1).squeeze()
        t = t.view(t.shape[0], self.patch_size, self.patch_size)

        # Now that the image has been divided into tiles, we flatten them
        local_ftrs = t.view(t.shape[0], -1)

        # Divisive normalization transform (DNT)
        mu = local_ftrs.mean(dim=1).unsqueeze(1)
        sigma = local_ftrs.std(dim=1).unsqueeze(1)
        local_ftrs = (local_ftrs - mu) / (sigma + self.eps)

        # Back to numpy
        local_ftrs = local_ftrs.cpu().numpy()

        # ZCA whitening
        local_ftrs = self.zca_whitening(local_ftrs)

        # Using float16 more memory-efficient
        local_ftrs = np.float16(local_ftrs)

        # I we're running HOSA on limited hardware,
        # we can't have all features because it takes too much RAM
        n_ftrs = len(local_ftrs)
        if (self.reduce_local > 0) and (self.reduce_local < n_ftrs):
            idxs = list(range(n_ftrs))
            random.shuffle(idxs)
            idxs = idxs[: self.reduce_local]
            local_ftrs = local_ftrs[idxs]

        return local_ftrs

    def generate_codebook(self, dset):
        """In HOSA, we need to generate the K-word codebook (K clusters)
        and the corresponding features of each cluster.

        The HOSA paper did not provide a definition of coskewness,
        so I had to look it up everywhere else. I found one here:
        'Coskewness and the short-term predictability for Bitcoin return'
        (Chen et al.)"""

        all_ftrs = []
        paths = dset["image_path"].values
        for i, im_path in enumerate(paths):
            im_name = Path(im_path).name
            print(f"[Codebook][{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)
            img_gray = self.prepare_input(img)
            local_ftrs = self.extract_local_features(img_gray)
            all_ftrs.append(local_ftrs)
        all_ftrs = np.concatenate(all_ftrs, axis=0)

        # Clustering
        print(
            f"Generating {self.codebook_size}-word codebook"
            f"(from {len(all_ftrs)} samples)"
        )
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=420)
        kmeans.fit(all_ftrs)
        self.codebook = np.float16(kmeans.cluster_centers_)  # [100 x D]

        # Computing the high order statistics of the clusters
        assigments = kmeans.labels_
        self.codebook_stats = []
        for i in range(self.codebook_size):
            cls_i_ftrs = all_ftrs[assigments == i]
            cls_mean = self.codebook[i, :]

            # The authors assume diagnoal covariance
            cls_cov = np.diag(np.cov(cls_i_ftrs, rowvar=False))

            # Coskewness (from another paper)
            diffs = cls_i_ftrs - cls_mean
            cls_std = np.sqrt(cls_cov)
            cls_coskw = np.mean(diffs * diffs**2) / (cls_std * cls_std**2)
            self.codebook_stats.append(list(cls_mean) + list(cls_cov) + list(cls_coskw))

        # Using float16 is more memory-efficient
        self.codebook_stats = np.float16(np.array(self.codebook_stats))

    def extract_features(self, x_gray):
        """Extracts the local features of an image, for which
        the r-nearest words from the main visual codebook are found.
        Once the nearest clusters are found, we compute the features"""

        local_ftrs = self.extract_local_features(x_gray)  # [M x D]

        # Computing the distance to the 100-word codebook
        # (codebook is [100 x D])
        norms = cdist(local_ftrs, self.codebook, "euclidean")  # [M x 100]
        rnn_idx = np.argsort(norms, axis=-1)[:, : self.r]  # [M x r]

        cluster_mask = np.zeros_like(norms)  # [M x 100]
        cluster_mask[rnn_idx] = 1  # [M x 100]

        # An efficient way of computing the soft weights?
        norms_sq = norms**2
        exp_norms = np.exp(-self.beta * norms_sq)  # [M x 100]
        weights = exp_norms / np.sum(exp_norms * cluster_mask, axis=0)  # [M x 100]
        weights = weights * cluster_mask  # [M x 100]

        # The statistics of all cluster assigments
        mu_hat = np.dot(weights.T, local_ftrs)  # [100 x D]
        var_hat = np.zeros_like(self.codebook)  # [100 x D]
        skw_hat = np.zeros_like(self.codebook)  # [100 x D]

        for i in range(self.codebook_size):
            # Say we have N features assigned to cluster i
            i_mask = (rnn_idx == i).sum(axis=1) > 0  # [M x 1]
            if sum(i_mask) > 0:
                cls_i_ftrs = local_ftrs[i_mask, :]  # [N x D]
                cls_i_mean = mu_hat[i, :]  # [1 x D]
                cls_i_weights = weights[i_mask, i]  # [N x 1]
                cls_i_diff = cls_i_ftrs - cls_i_mean  # [N x D]

                # Variance
                cls_i_var = np.dot(cls_i_weights.T, cls_i_diff**2)  # [1 x D]
                var_hat[i] = cls_i_var

                # Skewness is also [1 x D]
                div = cls_i_diff**3 / (cls_i_var**1.5 + self.eps)  # [N x D]
                cls_i_skw = np.dot(cls_i_weights.T, div)  # [1 x D]
                skw_hat[i] = cls_i_skw

        # Residuals
        cdbk_mean, cdbk_var, cdbk_skw = np.split(self.codebook_stats, 3, axis=1)
        mu_res = mu_hat - cdbk_mean
        var_res = var_hat - cdbk_var
        skw_res = skw_hat - cdbk_skw

        # We join everything back to create a single descriptor
        out = np.concatenate((mu_res, var_res, skw_res), axis=1)
        out = np.reshape(out, (1, -1))

        # Element-wise signed power normalization
        out = np.sign(out) * np.abs(out) ** self.alpha

        return np.float16(np.squeeze(out))

    def generate_feature_db(self, dset):
        """Creates the feature database that will be used to fit the SVR
        :param dset: a DataFrame with columns [image_name, image_path, score, [img_set]]
                    (not all datasets have the img_set columns, only those that contain
                      groups of distorted images created from the same pristine source)
        """

        # Creating the train/test splits
        if "is_test" not in dset.columns:
            dset = split_dataset(dset, self.test_size)

        # We first create the codebook
        if not self.codebook or not self.codebook_stats:
            print("Generating the visual codebook with the training set")
            train_dset = dset.loc[dset["is_test"] == 0, :].copy()
            self.generate_codebook(train_dset)

        # Then, we compute the main features (m, v, s) for every image
        feature_db = []
        for i, row in enumerate(dset.to_dict("records")):
            im_name = row["image_name"]
            im_path = row["image_path"]
            im_score = row["score"]
            im_split = row["is_test"]
            print(f"[Features][{i+1}/{len(dset)}]: Processing {im_name}")
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
        :param feature_db: dataframe with 147003 columns:
                    - image name
                    - 147K ftrs
                    - MOS
                    - test split indicator
        :param n_jobs: number of threads for GridSearchCV
        :param test_size: test set size
        """

        # Making the splits
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
            "svr__C": np.arange(5, 10, 0.5),
            "svr__epsilon": np.arange(0.25, 2.0, 0.25),
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

        print("Fitting an SVR for HOSA features")
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
        """Predicts the score from a set of ftrs"""
        score = self.svr_regressor.predict(f)
        return score


class LFA(HOSA):
    """Before HOSA, the authors developed LFA
    "Local feature aggregation for blind image quality assessment"
    https://ieeexplore.ieee.org/abstract/document/7457832"""

    def __init__(
        self,
        patch_size=7,
        codebook_size=100,
        reduce_local=-1,
        r=5,
        alpha=0.2,
        beta=0.05,
        img_size=512,
        eps=10,
        svr_regressor=None,
        test_size=0.3,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        # Parameters related to the K nearest codewords
        self.codebook = None
        self.codebook_size = codebook_size
        self.reduce_local = reduce_local
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.n_dims = self.patch_size**2

        # For normalization
        self.eps = eps

        self.n_features = 3 * self.n_dims * self.codebook_size
        self.svr_regressor = svr_regressor
        self.test_size = test_size

    def generate_codebook(self, dset):
        """In HOSA, we need to generate the K-word codebook (K clusters)
        and the corresponding features of each cluster.

        The HOSA paper did not provide a definition of coskewness,
        so I had to look it up everywhere else. I found one here:
        'Coskewness and the short-term predictability for Bitcoin return'
        (Chen et al.)"""

        all_ftrs = []
        paths = dset["image_path"].values
        for i, im_path in enumerate(paths):
            im_name = Path(im_path).name
            print(f"[Codebook][{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)
            img_gray = self.prepare_input(img)
            local_ftrs = self.extract_local_features(img_gray)
            all_ftrs.append(local_ftrs)
        all_ftrs = np.concatenate(all_ftrs, axis=0)

        # Clustering
        print(
            f"Generating {self.codebook_size}-word codebook"
            f"(from {len(all_ftrs)} samples)"
        )
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=420)
        kmeans.fit(all_ftrs)
        self.codebook = np.float16(kmeans.cluster_centers_)  # [100 x D]

    def extract_features(self, x_gray):
        """Extracts the local features of an image, for which
        the r-nearest words from the main visual codebook are found.
        Once the nearest clusters are found, we compute the features"""

        local_ftrs = self.extract_local_features(x_gray)  # [M x D]

        # Computing the distance to the 100-word codebook
        # (codebook is [100 x D])
        norms = cdist(local_ftrs, self.codebook, "euclidean")  # [M x 100]
        rnn_idx = np.argsort(norms, axis=-1)[:, : self.r]  # [M x r]

        # Computing the weights
        norms_sq = norms**2
        weights = np.exp(-self.beta * norms_sq)  # [M x 100]

        # The statistics of all cluster assigments
        v = np.zeros_like(self.codebook)  # [100 x D]

        for i in range(self.codebook_size):
            # Say we have N features assigned to cluster i
            i_mask = (rnn_idx == i).sum(axis=1) > 0  # [M x 1]
            if sum(i_mask) > 0:
                cls_i_ftrs = local_ftrs[i_mask, :]  # [N x D]
                cls_i_mean = self.codebook[i, :]  # [1 x D]
                cls_i_weights = weights[i_mask, i]  # [N x 1]
                cls_i_diff = cls_i_ftrs - cls_i_mean  # [N x D]
                cls_i_v = np.dot(cls_i_weights.T, cls_i_diff)  # [1 x D]
                v[i] = cls_i_v

        # Convert to a single vector
        out = np.reshape(v, (1, -1))

        # Element-wise signed power normalization + L2 normalization
        out = np.sign(out) * np.abs(out) ** self.alpha
        out = out / np.linalg.norm(out)

        return np.float16(np.squeeze(out))

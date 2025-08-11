import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
import pandas as pd
import random
from .data import split_dataset
from .processing import zca_whitening
from pathlib import Path
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from .metrics import lcc, srocc
import os
import pickle

random.seed(420)

# TODO: Use h5py to deal with large datasets during k-means


class CORNIA:
    """Unsupervised Feature Learning Framework for No-reference
    Image Quality Assessment by Ye et al. (https://ieeexplore.ieee.org/document/6247789)
    """

    def __init__(
        self,
        img_size=512,
        patch_size=7,
        num_patches=10000,
        codebook_size=2500,
        use_minibatch=True,
        eps=1e-6,
        codebook=None,
        svr_regressor=None,
        test_size=0.2,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.codebook = codebook
        self.codebook_size = codebook_size
        self.use_minibatch = use_minibatch
        self.eps = eps

        self.n_features = self.num_patches
        self.svr_regressor = svr_regressor
        self.test_size = test_size

        # We set stride to half the patch size to allow some overlapping
        self.patch_stride = self.patch_size // 2
        self.unfold = nn.Unfold(
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )

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

    def extract_local_features(self, x_gray):
        """Extracts the local features of an image. Each feature comes
         from a BxB patch of the , which is then normalised
        :param x_gray: grayscale image
        :returns local_ftrs: the local features
        """
        # Using Pytorch for extracting local image patches
        t = torch.from_numpy(x_gray).unsqueeze(0).unsqueeze(0).float()
        t = self.unfold(t).permute(0, 2, 1).squeeze()
        t = t.view(t.shape[0], self.patch_size, self.patch_size)

        # Now that the image has been divided into tiles, we flatten them
        local_ftrs = t.view(t.shape[0], -1)

        # We pick N different patches from the image
        pick_idxs = np.arange(t.shape[0])
        random.shuffle(pick_idxs)
        pick_idxs = pick_idxs[: self.num_patches]
        local_ftrs = local_ftrs[pick_idxs, :]

        # Divisive normalization transform (DNT)
        mu = local_ftrs.mean(dim=1).unsqueeze(1)
        sigma = local_ftrs.std(dim=1).unsqueeze(1)
        local_ftrs = (local_ftrs - mu) / (sigma + self.eps)

        # Back to numpy
        local_ftrs = local_ftrs.cpu().numpy()

        # ZCA whitening
        local_ftrs = zca_whitening(local_ftrs)

        # Using float16 is more memory-efficient
        local_ftrs = np.float16(local_ftrs)

        return local_ftrs

    def generate_codebook(self, dset):
        """Generate the K-word codebook (K clusters) from the training set."""

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
        if self.use_minibatch:
            # For faster computations, scikit-learn suggests a batch_size
            # greater than 256 * number of cores to enable parallelism on all cores.
            # So I decided to use 1024 * num_cores, why not??
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                random_state=420,
                batch_size=1024 * os.cpu_count(),
            )
        else:
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=420)
        kmeans.fit(all_ftrs)
        self.codebook = np.float16(kmeans.cluster_centers_)  # [100 x D]

    def extract_features(self, x_gray):
        """Extracts the local features of an image, and compares them to each
        cluster from the codebook"""

        local_ftrs = self.extract_local_features(x_gray)

        # Soft-assignment coding
        # Using Pytorch makes this slightly faster than numpy
        dot_similarity = torch.matmul(
            torch.from_numpy(local_ftrs), torch.from_numpy(self.codebook.T)
        ).numpy()
        # dot_similarity = local_ftrs.dot(self.codebook.T)
        encoding_pos = np.maximum(dot_similarity, 0)
        encoding_neg = np.maximum(-dot_similarity, 0)
        coefs = np.hstack((encoding_pos, encoding_neg))

        # Max pooling on the columns
        # This is different from what the authors say (max of each column
        # instead of each row), but it wouldn't make any sense otherwise
        # Because the lenght of the feature vector may be different for each image
        # if there weren't enough patches to sample
        coefs = np.max(coefs, axis=1)

        return np.float16(coefs)

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
        if not self.codebook:
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
        :param feature_db: dataframe with the columns:
                    - image name
                    - feature i ... feature N
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

        # The authors used a linear kernel
        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), SVR(kernel="linear")),
            param_grid=params,
            cv=5,
            n_jobs=n_jobs,
            verbose=1,
            scoring={"LCC": make_scorer(lcc), "SROCC": make_scorer(srocc)},
            error_score=0,
            refit="SROCC",
        )

        print("Fitting an SVR for CORNIA features")
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

    def export(self, path_save):
        # Export the codebook and the codebook stats
        print("Exporting the CORNIA data to: ", path_save)
        np.savetxt(path_save / "cornia_codebook.csv", self.codebook, delimiter=",")

        # Exporting the model
        path_pkl = path_save / "estimator.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, x):
        x_gray = self.prepare_input(x)
        ftrs = self.extract_features(x_gray)
        ftrs = np.array(ftrs)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(ftrs.reshape(1, -1))
        else:
            return ftrs


class HOSA:
    """Blind Image Quality Assessment Based on High Order Statistics Aggregation
    by Xu et al. (https://ieeexplore.ieee.org/document/7501619)"""

    def __init__(
        self,
        patch_size=7,
        codebook_size=100,
        use_minibatch=True,
        local_ftrs_frac=0.75,
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
        self.use_minibatch = use_minibatch
        self.local_ftrs_frac = local_ftrs_frac
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
        ftrs = self.extract_features(x_gray)
        ftrs = np.array(ftrs)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(ftrs.reshape(1, -1))
        else:
            return ftrs

    def extract_local_features(self, x_gray):
        """Extracts the local features of an image. Each feature comes
         from a BxB patch of the , which is then normalised
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
        local_ftrs = zca_whitening(local_ftrs)

        # Using float16 is more memory-efficient
        local_ftrs = np.float16(local_ftrs)

        # I we're running HOSA/LFA on limited hardware,
        # we can't have all features because it takes too much RAM
        n_sample = int(self.local_ftrs_frac * local_ftrs.shape[0])
        idxs = list(range(local_ftrs.shape[0]))
        random.shuffle(idxs)
        idxs = idxs[:n_sample]
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
        if self.use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                random_state=420,
                batch_size=1024 * os.cpu_count(),
            )
        else:
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
        :param feature_db: dataframe with the columns:
                    - image name
                    - feature i ... feature N
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

    def export(self, path_save):
        # Export the codebook and the codebook stats
        print("Exporting the HOSA data to: ", path_save)
        np.savetxt(path_save / "hosa_codebook.csv", self.codebook, delimiter=",")
        np.savetxt(path_save / "codebook_stats.csv", self.codebook_stats, delimiter=",")

        # Exporting the model
        path_pkl = path_save / "estimator.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


class LFA(HOSA):
    """Before HOSA, the authors developed LFA
    "Local feature aggregation for blind image quality assessment"
    https://ieeexplore.ieee.org/abstract/document/7457832"""

    def __init__(
        self,
        patch_size=7,
        codebook_size=100,
        use_minibatch=True,
        local_ftrs_frac=0.50,
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
        self.use_minibatch = use_minibatch
        self.local_ftrs_frac = local_ftrs_frac
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.n_dims = self.patch_size**2

        # For normalization
        self.eps = eps

        self.n_features = self.n_dims * self.codebook_size
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
        if self.use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                random_state=420,
                batch_size=1024 * os.cpu_count(),
            )
        else:
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

    def export(self, path_save):
        # Export the codebook
        print("Exporting the LFA data to: ", path_save)
        np.savetxt(path_save / "lfa_codebook.csv", self.codebook, delimiter=",")

        # Exporting the model
        path_pkl = path_save / "estimator.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


class SOM:
    """SOM: Semantic Obviousness Metric for Image Quality Assessment
    by Zhang et al. (https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_SOM_Semantic_Obviousness_2015_CVPR_paper.pdf)
    """

    def __init__(
        self,
        bing_path="classiqa/models/bing",
        max_regions=1000,
        num_samples=50,
        num_patches=5000,
        regions_per_patch=50,
        patch_size=7,
        num_training_reps=3,
        codebook_size=10000,
        use_minibatch=True,
        test_size=0.2,
        codebook=None,
        svr_regressor=None,
        eps=1e-6,  # to prevent NaNs
    ):
        # SOM hyperparameters (suggested by the authors)
        self.max_regions = max_regions  # K in the paper
        self.num_samples = num_samples  # N in the paper
        self.num_patches = num_patches  # M in the paper
        self.patch_size = patch_size  # B in the paper
        self.num_training_reps = num_training_reps  # E in the paper
        self.codebook_size = codebook_size  # W in the paper
        self.n_features = (2 * codebook_size) + max_regions

        # Some new parameters for my implementation
        self.regions_per_patch = regions_per_patch
        self.svr_regressor = svr_regressor
        self.test_size = test_size
        self.codebook = codebook
        self.use_minibatch = use_minibatch
        self.eps = eps

        # Loading the saliency model
        self.saliency_model = cv2.saliency.ObjectnessBING.create()
        self.saliency_model.setTrainingPath(bing_path)

    def extract_objectness_regions(self, x):
        """Object-like region detection with BING
        Based on: https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/"""
        _, boxes = self.saliency_model.computeSaliency(x)
        boxes = np.squeeze(boxes)  # shape is D x 1 x 4 (D = num detected regions)
        if boxes.dtype != "O":
            # The implementation of BING in opencv-contrib should be returning
            # the scores in descending order, but the Python code seems to be doing it
            # in ascending order. See: https://github.com/opencv/opencv_contrib/issues/404
            objectness = self.saliency_model.getobjectnessValues()
            objectness = objectness[: self.max_regions]
        else:
            # No detections for this image, so we'll sample patches
            # from the whole image
            print("No object-like regions found")
            x_end = x.shape[1] - 1
            y_end = x.shape[0] - 1
            boxes = [[0, 0, x_end, y_end]]
            objectness = np.empty(0)

        # The objectness vector must be K
        if len(objectness) < self.max_regions:
            n_missing = self.max_regions - len(objectness)
            objectness = np.concatenate((objectness, np.zeros(n_missing)))

        objectness = objectness.astype(np.float16)

        return boxes, objectness

    def extract_patch_descriptor(self, x, boxes):
        """Once the objectness regions are extracted, we sample
        M different BxB patches from those regions (see the SOM paper)"""
        patches = []

        # Pick one region randomly, and then some random coordinates
        # inside the regions to extract a BxB patch
        while len(patches) < self.num_patches:
            b = random.choice(boxes)
            max_b_x = b[2] - self.patch_size
            max_b_y = b[3] - self.patch_size
            x_start = np.random.uniform(
                low=b[0], high=max_b_x, size=self.regions_per_patch
            ).astype(int)
            y_start = np.random.uniform(
                low=b[1], high=max_b_y, size=self.regions_per_patch
            ).astype(int)
            x_end = x_start + self.patch_size
            y_end = y_start + self.patch_size
            for p in zip(x_start, x_end, y_start, y_end):
                patch_b = x[p[2] : p[3], p[0] : p[1]]
                patches.append(patch_b)

        # Flattened patches
        patches = np.stack(patches)
        ftrs = patches.reshape(patches.shape[0], -1)

        # Normalised patches
        mu = ftrs.mean(dim=1).unsqueeze(1)
        sigma = ftrs.std(dim=1).unsqueeze(1)
        ftrs = (ftrs - mu) / (sigma + self.eps)

        # ZCA whitening and converting to float16 for more efficiency
        zca_whitening(ftrs)
        ftrs = np.float16(ftrs)

        return ftrs

    def generate_codebook(self, dset):
        """In SOM, we oversample the training set to generate
        the K-word codebook (K clusters). This codebook only uses the
        local features, not the semantic objectness scores (X in the paper)"""

        local_ftrs = []
        paths = dset["image_path"].values
        for i, im_path in enumerate(paths):
            im_name = Path(im_path).name
            print(f"[Codebook][{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)

            # At this stage we don't need the objectness scores,
            # just the detections (the bounding boxes)
            objs, _ = self.extract_objectness_regions(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # We repeat the sampling and local feature extraction step
            # to increase the training set size
            for _ in range(self.num_training_reps):
                img_ftrs = self.extract_patch_descriptor(img_gray, objs)
                local_ftrs.append(img_ftrs)

        local_ftrs = np.concatenate(local_ftrs, axis=0)

        # Clustering
        print(
            f"Generating {self.codebook_size}-word codebook"
            f"(from {len(local_ftrs)} samples)"
        )
        if self.use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=self.codebook_size, random_state=420)
        else:
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=420)
        kmeans.fit(local_ftrs)
        self.codebook = np.float16(kmeans.cluster_centers_)  # [codebook_size x (B x B)]

    def extract_features(self, x):
        """Once we have the codebook, we can compute the final descriptor,
        which is a combination of the semantic obviousness (X) and the
        local characteristics (Z)
        F = [X, Z]
        """

        objs, semantic_ftrs = self.extract_objectness_regions(x)
        img_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        local_ftrs = self.extract_patch_descriptor(img_gray, objs)

        # Soft-assignment coding
        dot_similarity = np.dot(local_ftrs, self.codebook.T)
        encoding_pos = np.maximum(dot_similarity, 0)
        encoding_neg = np.maximum(-dot_similarity, 0)
        coefs = np.hstack((encoding_pos, encoding_neg))

        # Max pooling on the rows
        coefs = np.max(coefs, axis=0)

        # Feature fusion
        final_ftrs = np.concatenate((semantic_ftrs, coefs))

        return final_ftrs

    def generate_feature_db(self, dset):
        """Creates the feature database that will be used to fit the regressor
        :param dset: a DataFrame with columns [image_name, image_path, score, [img_set]]
                    (not all datasets have the img_set columns, only those that contain
                      groups of distorted images created from the same pristine source)
        """

        # Creating the train/test splits
        if "is_test" not in dset.columns:
            dset = split_dataset(dset, self.test_size)

        # We first create the codebook
        if not self.codebook:
            print("Generating the visual codebook with the training set")
            train_dset = dset.loc[dset["is_test"] == 0, :].copy()
            self.generate_codebook(train_dset)

        # Then, we compute the main features for every image
        feature_db = []
        for i, row in enumerate(dset.to_dict("records")):
            im_name = row["image_name"]
            im_path = row["image_path"]
            im_score = row["score"]
            im_split = row["is_test"]
            print(f"[Features][{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)
            ftrs = list(self.extract_features(img))
            feature_db.append([im_name] + ftrs + [im_score, im_split])

        feature_cols = list(range(1, self.n_features + 1))
        db_cols = ["image_name"] + feature_cols + ["MOS", "is_test"]
        feature_db = pd.DataFrame(feature_db, columns=db_cols)

        return feature_db

    def fit(self, feature_db, n_jobs=4):
        """
        Fit an SVR model to a given dset of ftrs
        :param feature_db: dataframe with the columns:
                    - image name
                    - feature i ... feature N
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

        # The authors of SOM used a linear kernel
        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), SVR(kernel="linear")),
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

    def export(self, path_save):
        # Exporting the model
        path_pkl = path_save / "estimator.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, x):
        ftrs = self.extract_features(x)
        ftrs = np.array(ftrs)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(ftrs.reshape(1, -1))
        else:
            return ftrs

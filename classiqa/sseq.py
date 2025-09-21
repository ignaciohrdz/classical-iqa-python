import cv2
import torch
from torch import nn
import numpy as np
import pandas as pd
from scipy.stats import skew
from .data import split_dataset
import pickle


class SSEQ:
    """Spatial-Spectral Entropy-based Quality (SSEQ) index (Liu et al.)"""

    def __init__(
        self,
        block_size=8,
        img_size=-1,
        percentile=0.6,
        scales=3,
        eps=1e-5,
        svr_regressor=None,
    ):
        self.block_size = block_size
        self.img_size = img_size
        self.percentile = percentile
        self.scales = scales
        self.eps = eps
        self.unfold = nn.Unfold(kernel_size=self.block_size, stride=self.block_size)
        self.svr_regressor = svr_regressor
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

    def generate_feature_db(self, dset, test_size=0.3):
        """Creates the feature database that will be used to fit the SVR
        :param dset: a DataFrame with columns [image_name, image_path, score, [img_set]]
                    (not all datasets have the img_set columns, only those that contain
                      groups of distorted images created from the same pristine source)
        :param test_size: percentage of images for the test set
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

    def export(self, path_save):
        path_pkl = path_save / "feature_extractor.pkl"
        print("Saving feature extractor to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

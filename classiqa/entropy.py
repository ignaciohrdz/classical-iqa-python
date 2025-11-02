import cv2
import torch
from torch import nn
import numpy as np
import pandas as pd
from scipy.stats import skew
from .data import split_dataset
import pickle
from itertools import combinations


class SSEQ:
    """Spatial-Spectral Entropy-based Quality (SSEQ) index (Liu et al.)"""

    def __init__(
        self,
        img_size=512,
        block_size=8,
        percentile=0.6,
        scales=3,
        eps=1e-5,
    ):
        self.block_size = block_size
        self.img_size = img_size
        self.percentile = percentile
        self.scales = scales
        self.eps = eps
        self.unfold = nn.Unfold(kernel_size=self.block_size, stride=self.block_size)
        self.n_features = self.scales * 4

        self.m = self.make_dct_matrix()
        self.m_t = self.m.T

    def crop_input(self, x):
        """We make sure the image is divisible into NxN tiles (N = block_size)
        If the image is not divisible, we crop it start from the top-left corner"""
        h, w = x.shape
        h_cropped = h - (h % self.block_size)
        w_cropped = w - (w % self.block_size)
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
        """Creates the feature database that will be used to fit the regressor
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


class ENIQA:
    """Entropy-based No-reference IQA (ENIQA) by Chen et al."""

    def __init__(
        self,
        img_size=512,
        scales=[1, 0.5],
        block_size=8,
        epsilon=1e-6,
        saliency_sigma=2.5,
        saliency_gauss_ksize=11,
        avg_ksize=3,
        lgf_sigmaonf=0.55,
        lgf_mult=1.31,
        lgf_dthetaonsigma=1.10,
        lgf_min_wavelenght=2.4,
        lgf_scale_factor=0.87,
        lgf_n_scale=2,
        lgf_n_orient=4,
    ):
        self.img_size = img_size
        self.scales = scales
        self.block_size = block_size
        self.unfold = nn.Unfold(kernel_size=self.block_size, stride=self.block_size)

        self.epsilon = epsilon

        self.avg_ksize = avg_ksize
        self.avg_kernel = np.ones((self.avg_ksize, self.avg_ksize))
        self.avg_kernel[self.avg_ksize // 2, self.avg_ksize // 2] = 0
        self.avg_kernel /= self.avg_ksize**2 - 1

        self.saliency_sigma = saliency_sigma
        self.saliency_gauss_ksize = saliency_gauss_ksize

        # lgf stands for Log-Gabor filter
        self.lgf_sigmaonf = lgf_sigmaonf
        self.lgf_mult = lgf_mult
        self.lgf_dthetaonsigma = lgf_dthetaonsigma
        self.lgf_min_wavelenght = lgf_min_wavelenght
        self.lgf_scale_factor = lgf_scale_factor
        self.lgf_n_scale = lgf_n_scale
        self.lgf_n_orient = lgf_n_orient

        # This will have to be calculated once we have everything ready
        self.n_features = 28 * len(self.scales)

    def crop_input(self, x_color):
        """We make sure the image is divisible into NxN tiles (N = block_size)
        If the image is not divisible, we crop it start from the top-left corner"""
        h, w = x_color.shape[:2]
        h_cropped = h - (h % self.block_size)
        w_cropped = w - (w % self.block_size)
        return x_color[:h_cropped, :w_cropped]

    def prepare_input(self, x, img_size=None):
        """Initial conversion to grayscale and resizing"""

        x_color = self.crop_input(x)

        if not img_size:
            img_size = self.img_size
        if img_size > 0:
            ratio = self.img_size / max(x_color.shape[:2])
            x_color = cv2.resize(
                x_color,
                None,
                fx=ratio,
                fy=ratio,
                # They used nearest-neighbour in the paper
                interpolation=cv2.INTER_NEAREST,
            )
        x_gray = cv2.cvtColor(x_color, cv2.COLOR_BGR2GRAY)
        return x_color, x_gray

    def compute_saliency(self, x_gray):
        # Salciency Detection: A Spectral Residual Approach
        # http://www.houxiaodi.com/assets/papers/cvpr07.pdf
        # (This is the method used in ENIQA)
        spec = np.fft.fft2(x_gray)
        spec[spec == 0] = self.epsilon
        mag_spec = np.log(np.abs(spec))
        phase = np.angle(spec)

        res_spec = mag_spec - cv2.blur(mag_spec, (3, 3))
        saliency_fft = np.fft.ifft2(np.exp(res_spec + 1j * phase))
        saliency = np.abs(saliency_fft) ** 2

        # Post-processing (as in their MATLAB code)
        saliency = cv2.GaussianBlur(
            saliency,
            ksize=(self.saliency_gauss_ksize, self.saliency_gauss_ksize),
            sigmaX=self.saliency_sigma,
            sigmaY=self.saliency_sigma,
        )

        # For visualization
        # saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        # cv2.imshow("Image", x_gray)
        # cv2.imshow("Saliency", saliency_norm)
        # cv2.waitKey()

        return saliency

    def make_log_gabor_filters(self, img):
        """Create the log-Gabor filters (almost identical to the original repo)"""

        # Standard deviation of the angular component of the filter.
        theta_sigma = np.pi / self.lgf_n_orient / self.lgf_dthetaonsigma

        # Pre-compute X and Y matrices with ranges normalized to +/- 0.5
        # (ensure the range is symmetrical)
        rows, cols = img.shape
        if cols % 2:
            x_range = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
        else:
            x_range = np.arange(-cols / 2, cols / 2) / cols

        if rows % 2:
            y_range = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
        else:
            y_range = np.arange(-rows / 2, rows / 2) / rows

        # Normalized radius from the center and polar angle
        x, y = np.meshgrid(x_range, y_range)
        radius = np.sqrt(x**2 + y**2)
        theta = np.arctan2(-y, x)  # -y for positive anti-clockwise angles

        # cv2.imshow("Radius", radius)
        # cv2.imshow("Angle", np.abs(theta) / theta.max())
        # cv2.waitKey()

        # Quadrant shift radius and theta so that filters are constructed
        # with 0 frequency at the corners.
        radius = np.fft.ifftshift(radius)
        radius[0, 0] = 1  # to avoid log(0) at the zero-frequency point.
        theta = np.fft.ifftshift(theta)

        # cv2.imshow("Radius (shift)", radius)
        # cv2.imshow("Angle (shift)", np.abs(theta) / theta.max())
        # cv2.waitKey()

        # Initialize the filter bank
        rad_comps = []
        ang_comps = []

        # Construct the radial filter components
        for s in range(self.lgf_n_scale):
            wavelength = self.lgf_min_wavelenght * (self.lgf_scale_factor**s)
            fo = 1.0 / wavelength
            comp = np.exp(
                -(np.log(radius / fo) ** 2) / (2 * np.log(self.lgf_sigmaonf) ** 2)
            )
            comp[0, 0] = 0  # set the 0 frequency point back to zero.
            rad_comps.append(comp)

            # For visualization
            # cv2.imshow("Radial component", comp)
            # cv2.waitKey()

        # Construct the angular filter components
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        for o in range(self.lgf_n_orient):
            angl = o * np.pi / self.lgf_n_orient  # Filter angle.
            # Calculate the angular distance from the specified filter orientation
            # We must use sine and cosine to avoid the angular wrap-around problem
            diff_sin = sintheta * np.cos(angl) - costheta * np.sin(angl)
            diff_cos = costheta * np.cos(angl) + sintheta * np.sin(angl)
            diff_theta = np.abs(np.arctan2(diff_sin, diff_cos))
            comp = np.exp(-(diff_theta**2) / (2 * theta_sigma**2))
            ang_comps.append(comp)

            # For visualization
            # cv2.imshow("Angular component", comp)
            # cv2.waitKey()

        # Combine radial and angular components to get the filters
        banks = []
        for o in range(self.lgf_n_orient):
            for s in range(self.lgf_n_scale):
                banks.append(rad_comps[s] * ang_comps[o])

        # For visualization
        # for b in banks:
        #     cv2.imshow("log-Gabor filter", b)
        #     cv2.waitKey()

        return banks

    def block_entropy(self, t_flat, lim=256):
        """After converting an image into N tiles of shape BxB and flattening each tile,
        we compute their entropy"""
        # In order to compute it faster, I will use offsetting
        #  instead of computing row-wise entropy
        # https://discuss.pytorch.org/t/count-number-occurrence-of-value-per-row/137061/5
        min_length = lim * t_flat.shape[0]
        t_flat_offset = t_flat + lim * torch.arange(t_flat.shape[0]).unsqueeze(1)
        counts = torch.bincount(t_flat_offset.flatten(), minlength=min_length)
        counts = counts.reshape(t_flat.shape[0], lim)
        p = counts / (self.block_size**2)
        p[counts == 0] = 1  # to prevent NaNs
        entropy = -(torch.log2(p) * p).sum(dim=1)
        return entropy

    def convert_to_blocks(self, arr):
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        t = self.unfold(t).permute(0, 2, 1).squeeze()
        t = t.view(t.shape[0], self.block_size, self.block_size)
        return t

    def mutual_information(self, x_1, x_2):
        # One-dimensional entropy
        h_1 = []
        for arr in [x_1, x_2]:
            _, counts = np.unique(arr, return_counts=True)
            p = counts / counts.sum()
            h_ch = -np.sum(p * np.log2(p))
            h_1.append(h_ch)

        # Two-dimensional entropy
        x_1 /= 255.0
        x_2 /= 255.0
        x_encoded = ((x_1 * 255) + x_2).astype(int)
        _, counts = np.unique(x_encoded, return_counts=True)
        p = counts / counts.sum()
        h_2 = -np.sum(p * np.log2(p))

        return sum(h_1) - h_2

    def compute_scale_features(self, x, img_size):
        scale_ftrs = {}

        # Initial resizing
        x_color, x_gray = self.prepare_input(x, img_size=img_size)

        # Feature group 1: MI between RGB channels
        x_b, x_g, x_r = cv2.split(x_color.astype(float))
        mi_bg = self.mutual_information(x_b, x_g)
        mi_br = self.mutual_information(x_b, x_r)
        mi_gr = self.mutual_information(x_g, x_r)
        scale_ftrs["group_1"] = [mi_bg, mi_br, mi_gr]

        # Before moving on, we compute saliency from the grayscale image
        saliency = self.compute_saliency(x_gray)
        t_saliency = torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0).float()
        t_saliency = self.unfold(t_saliency).permute(0, 2, 1).squeeze()
        t_saliency = t_saliency.sum(axis=1).numpy()
        sorted_idxs = np.argsort(t_saliency)[::-1]
        n_important = int(0.80 * t_saliency.shape[0])
        important_idxs = sorted_idxs[:n_important]

        # Feature group 2: Mean and skewness of TE for the grayscale image
        # We create the local patches of the grayscale image and keep
        # only the important ones, according to their saliency values
        # Note: I use .copy() because I was getting this error:
        # https://discuss.pytorch.org/t/negative-strides-in-tensor-error/134287
        t_patches = self.convert_to_blocks(x_gray)
        t_patches = t_patches[important_idxs.copy(), ...]
        t_patches /= 255.0

        # Compute the neighbourhood mean and convert it to blocks
        x_avg = cv2.filter2D(x_gray, ddepth=-1, kernel=self.avg_kernel)
        t_avg = self.convert_to_blocks(x_avg)
        t_avg = t_avg[important_idxs.copy(), ...] / 255.0

        # Compute TE of the encoded pairs of pixels (center pixel + one neighbour)
        t_encoded = ((t_patches * 255) + t_avg).int()
        t_encoded = t_encoded.view(t_encoded.shape[0], -1)
        te = self.block_entropy(t_encoded, lim=256 + 1).numpy()
        te_mean = te.mean()
        te_skew = skew(te)
        scale_ftrs["group_2"] = [te_mean, te_skew]

        # Feature group 3: mean and skewness of TE for eight sub-band images
        scale_ftrs["group_3"] = []
        lgf_bank = self.make_log_gabor_filters(x_gray)
        x_lgf = []
        for subband in lgf_bank:
            x_band = np.abs(np.fft.ifft2(np.fft.fft2(x_gray) * subband))
            x_lgf.append(x_band)
            t_band = self.convert_to_blocks(x_band).int()
            t_band = t_band.view(t_band.shape[0], -1)
            te_band = self.block_entropy(t_band).numpy()
            band_ftrs = [te_band.mean(), skew(te_band)]
            scale_ftrs["group_3"].extend(band_ftrs)

        # Feature group 4: MI of sub-band images in different orientations
        orients_idxs = [[0, 4], [1, 5], [2, 6], [3, 7]]
        x_lgf_orients = []
        for idxs in orients_idxs:
            orient_sum = np.stack([x_lgf[idxs[0]], x_lgf[idxs[1]]]).sum(0)
            x_lgf_orients.append(orient_sum)

        scale_ftrs["group_4"] = []
        for orient_pair in combinations(x_lgf_orients, 2):
            scale_ftrs["group_4"].append(self.mutual_information(*orient_pair))

        # Feature group 5: MI of sub-band images at different center frequencies
        x_lgf_scale1 = np.stack(x_lgf[:4]).sum(0)
        x_lgf_scale2 = np.stack(x_lgf[4:]).sum(0)
        scale_ftrs["group_5"] = [self.mutual_information(x_lgf_scale1, x_lgf_scale2)]

        return scale_ftrs

    def extract_features(self, x):
        # The features must be extracted at several scales
        scale_dicts = []
        for s in self.scales:
            img_size = int(self.img_size * s)
            s_ftrs = self.compute_scale_features(x, img_size)
            scale_dicts.append(s_ftrs)

        # Concatenate the dicts to make a single list that follows
        # the exact order described in the ENIQA paper
        features = []
        for grp in range(5):  # there are 5 feature groups
            group_name = f"group_{grp+1}"
            grp_ftrs = []
            for s_dict in scale_dicts:
                grp_ftrs.extend(s_dict[group_name])
            features.extend(grp_ftrs)

        # Float32 is more memory-efficient
        features = np.array(features, dtype=np.float32)

        return features

    def generate_feature_db(self, dset, test_size=0.3):
        """Creates the feature database that will be used to fit the regressor
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
            ftrs = list(self.extract_features(img))
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

    def __call__(self, x):

        # Initial resizing (just to ensure we always operate at the same size)
        x_color, _ = self.prepare_input(x)

        # Extracting the ftrs at different scales
        ftrs = self.extract_features(x_color)

        return ftrs

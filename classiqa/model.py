"""This is the base model"""

import cv2
import numpy as np
import pandas as pd
import random
from .data import split_dataset
import pickle

random.seed(420)


class BaseModel:
    """A basic template for all IQA models. It just does basic image preprocessing
    and defines the basic workflow for feature database generation, which is needed
    to train the score regressor later"""

    def __init__(self, img_size, n_features, bgr_input=False):

        self.img_size = img_size
        self.n_features = n_features
        self.bgr_input = bgr_input

    def prepare_input(self, x):
        """Initial conversion to grayscale and resizing"""

        if not self.bgr_input:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        if self.img_size > 0:
            ratio = self.img_size / max(x.shape[:2])
            x = cv2.resize(
                x,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_CUBIC,
            )
        return x

    def extract_features(self, x):
        """This function should return the features, and the first step
        should always be prepare_input"""

        x_gray = self.prepare_input(x)
        features = [np.zeros(self.n_features)]
        return features

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
            ftrs = list(self.extract_features(img))
            feature_db.append([im_name] + ftrs + [im_score, im_split, im_set])

        feature_cols = list(range(1, self.n_features + 1))
        db_cols = ["image_name"] + feature_cols + ["MOS", "is_test", "image_set"]
        feature_db = pd.DataFrame(feature_db, columns=db_cols)

        return feature_db

    def __call__(self, x):
        fts = self.extract_features(x)
        features = np.array(fts)

        return features

    def export(self, path_save):
        path_pkl = path_save / "feature_extractor.pkl"
        print("Saving feature extractor to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


class PatchModel(BaseModel):
    """This model assumes the metric uses patches of patch_size x patch_size,
    so it ensures that all inputs can be safely divided into patches of such size"""

    def __init__(self, img_size, n_features, patch_size):
        super().__init__(img_size, n_features)
        self.patch_size = patch_size

    def crop_input(self, x):
        """We make sure the image is divisible into NxN tiles (N = patch_size)
        If the image is not divisible, we crop it
        starting from the top-left corner"""
        h, w = x.shape[:2]
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

"""This is the base model"""

import cv2
import numpy as np
import pandas as pd
import random
from .data import split_dataset
import pickle

random.seed(420)


class BaseModel:
    """A wrapper for Scikit-image's blur_effect

    See: https://scikit-image.org/docs/stable/auto_examples/filters/plot_blur_effect.html
    """

    def __init__(self, img_size, n_features):

        self.img_size = img_size
        self.n_features = n_features

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
        """This function should return the features"""
        features = [np.zeros(self.n_features)]
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

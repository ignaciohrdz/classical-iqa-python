"""This is an example of how to use the models"""

import cv2
import pickle
from classiqa.utils import extract_test_args
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    args = extract_test_args()

    # Debugging
    args.model = "gmlog"
    args.dataset = "tid2013"

    path_image_original = Path("images/test_image_orig.jpg")
    path_image_distorted = Path("images/test_image_dist.jpg")

    # Loading the feature extractor and the regressor
    dset_name = args.dataset
    path_model = Path(args.path_models) / args.model / args.regressor / args.dataset
    path_extractor_file = path_model / "feature_extractor.pkl"
    path_regressor_file = path_model / "score_regressor.pkl"

    with open(path_extractor_file, "rb") as f:
        feature_extractor = pickle.load(f)
    with open(path_regressor_file, "rb") as f:
        score_regressor = pickle.load(f)

    # Feature extraction and score prediction (original vs distorted)
    img_original = cv2.imread(str(path_image_original))
    img_distorted = cv2.imread(str(path_image_distorted))

    ftrs_orig = feature_extractor(img_original)
    score_orig = score_regressor(np.expand_dims(ftrs_orig, 0))[0]

    ftrs_dist = feature_extractor(img_distorted)
    score_dist = score_regressor(np.expand_dims(ftrs_dist, 0))[0]
    print(f"Original: {score_orig:.5f}, Distorted: {score_dist:.5f}")

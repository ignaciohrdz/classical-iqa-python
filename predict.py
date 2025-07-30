"""This is an example of how to use the models"""

import cv2
import pickle
from classiqa.utils import extract_test_args
from pathlib import Path


if __name__ == "__main__":
    args = extract_test_args()

    # Debugging
    args.model = "sseq"
    args.use_dataset = "csiq"

    path_image_original = Path("images/test_image_orig.jpg")
    path_image_distorted = Path("images/test_image_dist.jpg")

    dset_name = args.use_dataset
    path_model = Path(args.path_models) / args.model / args.use_dataset
    path_model_file = path_model / "estimator.pkl"
    with open(path_model_file, "rb") as f:
        estimator = pickle.load(f)

    img_original = cv2.imread(str(path_image_original))
    img_distorted = cv2.imread(str(path_image_distorted))
    score_orig = estimator(img_original)[0]
    score_dist = estimator(img_distorted)[0]
    print(f"Original: {score_orig:.5f}, Distorted: {score_dist:.5f}")

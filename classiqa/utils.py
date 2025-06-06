import argparse
from classiqa.data import dataset_names
from classiqa.sseq import SSEQ
from classiqa.gmlog import GMLOG
from classiqa.hosa import HOSA, LFA

MODEL_DICT = {"sseq": SSEQ, "lfa": LFA, "hosa": HOSA, "gmlog": GMLOG}


def extract_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_datasets",
        type=str,
        help="Path to the directory with all datasets",
    )
    parser.add_argument(
        "--use_dataset",
        type=str,
        help=f"Dataset name: {dataset_names}",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="IQA model name",
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default="models",
        help="Location of the trained SVR models and measures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing feature database",
    )
    args = parser.parse_args()

    return args


def extract_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="IQA model name",
    )
    parser.add_argument(
        "--use_dataset",
        type=str,
        help=f"Dataset name: {dataset_names}",
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default="models",
        help="Location of the trained SVR models and measures",
    )
    args = parser.parse_args()
    return args


def resolve_model(model_name):
    valid_keys = list(MODEL_DICT.keys())
    if model_name not in valid_keys:
        print("ERROR: You must use one of these:", valid_keys)
        estimator = None
    else:
        estimator = MODEL_DICT[model_name]()
    return estimator

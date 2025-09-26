import argparse
from classiqa.data import dataset_names
from classiqa.sseq import SSEQ
from classiqa.gmlog import GMLOG
from classiqa.codebook import CORNIA, HOSA, LFA, SOM

import json

MODEL_DICT = {
    "sseq": SSEQ,
    "lfa": LFA,
    "hosa": HOSA,
    "gmlog": GMLOG,
    "cornia": CORNIA,
    "som": SOM,
}


def extract_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_datasets",
        type=str,
        help="Path to the directory with all datasets",
    )
    parser.add_argument(
        "--dataset",
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
        default="classiqa/models",
        help="Location of the trained SVR models and measures",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Image size",
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
        "--dataset",
        type=str,
        help=f"Dataset name: {dataset_names}",
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default="classiqa/models",
        help="Location of the trained SVR models and measures",
    )
    args = parser.parse_args()
    return args


def resolve_model(model_name, img_size):
    valid_keys = list(MODEL_DICT.keys())
    if model_name not in valid_keys:
        print("ERROR: You must use one of these:", valid_keys)
        estimator = None
    else:
        estimator = MODEL_DICT[model_name](img_size=img_size)
    return estimator


def export_results(path_json, dataset, model, metrics):
    if path_json.exists():
        with open(path_json, "r") as f:
            all_results = json.load(f)
        if dataset not in all_results.keys():
            all_results[dataset] = {model: metrics}
        elif model not in all_results[dataset].keys():
            all_results[dataset][model] = metrics
    else:
        all_results = {dataset: {model: metrics}}

    with open(path_json, "w") as f:
        json.dump(all_results, f)

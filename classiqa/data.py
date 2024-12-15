import pandas as pd
from pathlib import Path
import scipy
import random

random.seed(420)


def split_dataset(dset, test_size=0.2):
    """Splits the dataset into a training and a test set

    :param dset: a DataFrame with columns [image_name, image_path, score, [image_set]]
                    (not all datasets have the image_set column, only those that contain
                      groups of distorted images created from the same pristine source)
    """
    has_img_set = "image_set" in dset.columns
    dset.loc[:, "is_test"] = 0
    if has_img_set:
        # Some images are similar and come from the same pristine source
        img_sets = sorted(dset["image_set"].unique().tolist())
        random.shuffle(img_sets)
        test_sets = img_sets[: int(len(img_sets) * test_size)]
        dset.loc[dset["image_set"].isin(test_sets), "is_test"] = 1
    else:
        # Every image is a unique sample
        idxs = list(range(len(dset)))
        random.shuffle(idxs)
        test_idx = idxs[: int(len(idxs) * test_size)]
        dset.loc[test_idx, "is_test"] = 1
    return dset


def prepare_koniq(path_koniq: Path):
    """Prepares the KonIQ-10k dataset for training"""
    path_images = path_koniq / "1024x768"
    path_scores = path_koniq / "koniq10k_scores_and_distributions.csv"
    dataset = pd.read_csv(path_scores)
    dataset.rename(columns={"MOS": "score"}, inplace=True)
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_images / x))
    return dataset.loc[:, ["image_path", "image_name", "score"]]


def prepare_kadid(path_kadid: Path):
    """Prepares the KADID-10k dataset for training"""
    path_images = path_kadid / "images"
    path_scores = path_kadid / "dmos.csv"
    dataset = pd.read_csv(path_scores)
    dataset.rename(columns={"dist_img": "image_name", "dmos": "score"}, inplace=True)
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_images / x))
    return dataset.loc[:, ["image_path", "image_name", "score"]]


def prepare_csiq(path_csiq: Path):
    """Prepares the CSIQ dataset for training"""
    path_distorted_images = path_csiq / "dst_imgs"
    path_scores = path_csiq / "csiq.DMOS.xlsx"
    dataset = pd.read_excel(path_scores, sheet_name="all_by_image")
    dataset["image"] = dataset["image"].astype(str)
    dataset.set_index(["image", "dst_type", "dst_lev"], inplace=True)

    csiq_data = pd.DataFrame(columns=["image_path", "image_name", "score", "image_set"])
    for f in path_distorted_images.glob("**/*.png"):
        src_img_name, dst_type, dst_level = str(f.stem).lower().split(".")

        # Distortion name changes
        dst_type = dst_type.replace("awgn", "noise")
        dst_type = dst_type.replace("jpeg2000", "jpeg 2000")

        # For some reason, not all the distorted images were rated,
        # and they don"t appear in the Excel file
        try:
            score = dataset.loc[(src_img_name, dst_type, int(dst_level)), "dmos"]
            csiq_data.loc[len(csiq_data), :] = [str(f), f.name, score, src_img_name]
        except KeyError:
            print("Score not found for this CSIQ image: ", f.name)
    return csiq_data


def prepare_tid(path_tid: Path):
    """Prepares the TID2013 dataset for training"""
    path_images = path_tid / "distorted_images"
    path_scores = path_tid / "mos_with_names.txt"
    dataset = pd.read_csv(path_scores, names=["score", "image_name"], sep=" ")
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_images / x))
    dataset["image_set"] = dataset["image_name"].apply(
        lambda x: x.split("_")[0].lower()
    )
    return dataset.loc[:, ["image_path", "image_name", "score", "image_set"]]


def prepare_liveiqa(path_liveiqa: Path):
    """Prepares the LIVE-IQA dataset (database release 2) for training"""
    # Loading the MATLAB file
    path_dataset = path_liveiqa / "databaserelease2"
    path_scores = path_dataset / "dmos.mat"
    mat_file = scipy.io.loadmat(str(path_scores))
    dmos = mat_file["dmos"]
    is_original = mat_file["orgs"]

    # The Readme.txt file from the database tells us
    # the number of images per distortion
    distortion_sizes = {
        "jp2k": 227,
        "jpeg": 233,
        "wn": 174,
        "gblur": 174,
        "fastfading": 174,
    }
    dataset = pd.DataFrame(columns=["image_path", "image_name", "score"])
    offset = 0
    for dist_name, n_imgs in distortion_sizes.items():
        for i in range(n_imgs):
            if is_original[0, offset + i] == 0:
                image_name = "img{}.bmp".format(i + 1)
                image_path = path_dataset / dist_name / image_name
                score = dmos[0, offset + i]
                dataset.loc[len(dataset), :] = [image_path, image_name, score]
        offset += n_imgs
    return dataset


def prepare_nitsiqa(path_nitsiqa):
    """Prepares the NITSIQA dataset for training"""
    col_changes = {
        "Distorted Image Name": "image_name",
        "Score": "score",
        "Original Image Name": "image_set",
    }
    path_dataset = path_nitsiqa / "Database"
    path_scores = path_dataset / "Score.xlsx"
    dataset = pd.read_excel(path_scores, sheet_name="Sheet1")
    dataset.rename(columns=col_changes, inplace=True)
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_dataset / x))
    nitsiqa_data = dataset.loc[:, ["image_path", "image_name", "score", "image_set"]]
    return nitsiqa_data


dataset_fn_dict = {
    "koniq10k": prepare_koniq,
    "kadid10k": prepare_kadid,
    "csiq": prepare_csiq,
    "tid2013": prepare_tid,
    "liveiqa": prepare_liveiqa,
    "nitsiqa": prepare_nitsiqa,
}

dataset_names = list(dataset_fn_dict.keys())

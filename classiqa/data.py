import pandas as pd
from pathlib import Path
import scipy
import random
import cv2
import numpy as np

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
        # Case A: Some images are similar and come from the same pristine source
        img_sets = sorted(dset["image_set"].unique().tolist())
        random.shuffle(img_sets)
        test_sets = img_sets[: int(len(img_sets) * test_size)]
        dset.loc[dset["image_set"].isin(test_sets), "is_test"] = 1
    else:
        # Case B: Every image is a unique sample
        idxs = list(range(len(dset)))
        random.shuffle(idxs)
        test_idx = idxs[: int(len(idxs) * test_size)]
        dset.loc[test_idx, "is_test"] = 1
        dset["image_set"] = "image_" + dset.index.astype(str)
    return dset


# TODO: Add URLs and explain the user how to download the datasets


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


def prepare_csiq_cornia(path_csiq: Path):
    """Prepares the CSIQ dataset with some additional distortions
    as in the CORNIA paper: https://ieeexplore.ieee.org/document/6247789"""

    # The base dataset
    csiq_data = prepare_csiq(path_csiq)

    # Expanding the dataset with more distortions
    extra_dataset = []
    path_src = path_csiq / "src_imgs"
    path_extension = path_csiq / "cornia_extension"

    # Salt & pepper noise
    # Based on: https://stackoverflow.com/a/27342545
    path_sp = path_extension / "salt_pepper"
    path_sp.mkdir(exist_ok=True, parents=True)
    for i, sp_thresh in enumerate([5, 15, 20]):
        for f in path_src.glob("*"):
            # Loading and distorting the source image
            img = cv2.imread(str(f))
            noise = np.random.randint(0, 255, size=img.shape[:2])
            img[noise < sp_thresh, :] = 0
            img[noise > 255 - sp_thresh, :] = 255

            # Saving the distorted image
            dst_name = f"{f.stem}.SP.{i+1}.png"
            path_dst = path_sp / dst_name
            cv2.imwrite(str(path_dst), img)
            extra_dataset.append(
                {
                    "image_path": str(path_dst),
                    "image_name": dst_name,
                    "score": -1,
                    "image_set": f.stem,
                }
            )

    # Poisson noise
    # https://stackoverflow.com/a/36331042
    path_poisson = path_extension / "poisson"
    path_poisson.mkdir(exist_ok=True, parents=True)
    for i, scale in enumerate([20, 40, 60]):
        for f in path_src.glob("*"):
            # Loading and distorting the source image
            img = cv2.imread(str(f)) / 255.0
            noise = (np.random.poisson(img * scale) / scale) * 255
            noise = noise.astype(np.uint8)

            # Saving the distorted image
            dst_name = f"{f.stem}.POISSON.{i+1}.png"
            path_dst = path_poisson / dst_name
            cv2.imwrite(str(path_dst), noise)
            extra_dataset.append(
                {
                    "image_path": str(path_dst),
                    "image_name": dst_name,
                    "score": -1,
                    "image_set": f.stem,
                }
            )

    # Speckle noise
    # https://www.geeksforgeeks.org/computer-vision/noise-tolerance-in-opencv/
    path_speckle = path_extension / "speckle"
    path_speckle.mkdir(exist_ok=True, parents=True)
    for f in path_src.glob("*"):
        # Loading and distorting the source image
        img = cv2.imread(str(f))
        noise = np.random.normal(0, 1, img.shape).astype(np.float32)
        img = img + img * noise
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Saving the distorted image
        dst_name = f"{f.stem}.SPECKLE.1.png"
        path_dst = path_speckle / dst_name
        cv2.imwrite(str(path_dst), img)
        extra_dataset.append(
            {
                "image_path": str(path_dst),
                "image_name": dst_name,
                "score": -1,
                "image_set": f.stem,
            }
        )

    extra_dataset = pd.DataFrame(extra_dataset)
    dataset = pd.concat([csiq_data, extra_dataset], ignore_index=True)

    return dataset


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
    dmos = mat_file["dmos"][0]
    is_original = mat_file["orgs"][0]

    # The Readme.txt file from the database tells us
    # the number of images per distortion
    distortion_sizes = {
        "jp2k": 227,
        "jpeg": 233,
        "wn": 174,
        "gblur": 174,
        "fastfading": 174,
    }

    # We will use the information files from each subset to group the images
    img_sets = {}
    for dist_name in distortion_sizes.keys():
        path_info = path_dataset / dist_name / "info.txt"
        dist_info = pd.read_csv(path_info, sep=" ", header=None)
        dist_info = dist_info.iloc[:, :3]
        dist_info.columns = ["source", "dest", "param"]
        img_sets[dist_name] = dist_info

    dataset = pd.DataFrame(columns=["image_path", "image_name", "score", "image_set"])
    offset = 0
    for dist_name, n_imgs in distortion_sizes.items():
        for i in range(n_imgs):
            if is_original[offset + i] == 0:
                image_name = "img{}.bmp".format(i + 1)
                image_path = path_dataset / dist_name / image_name
                dist_info = img_sets[dist_name]
                image_set = dist_info.loc[dist_info["dest"] == image_name, "source"]
                image_set = image_set.values[0]
                score = dmos[offset + i]
                dataset.loc[len(dataset), :] = [
                    image_path,
                    image_name,
                    score,
                    image_set,
                ]
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


def prepare_cidiq(path_cidiq):
    """Prepares the CID:IQ dataset for training"""
    path_scores = path_cidiq / "MOS50.mat"
    path_dataset = path_cidiq / "Images" / "Reproduction"

    # Loading the MOS
    scores = scipy.io.loadmat(str(path_scores))["MOS50"]
    scores = scores.squeeze().tolist()

    # There are 690 scores and 690 images. The folders are in numerica order,
    # so I assume the MOS values follow the same order
    files = list(path_dataset.glob("**/*.bmp"))
    img_paths = [str(f) for f in files]
    img_names = [f.name for f in files]
    img_sets = [n.split("_")[0] for n in img_names]

    cidiq_data = pd.DataFrame(
        {
            "image_path": img_paths,
            "image_name": img_names,
            "score": scores,
            "image_set": img_sets,
        }
    )

    return cidiq_data


# TODO: Get the CID2013 dataset


dataset_fn_dict = {
    "koniq10k": prepare_koniq,
    "kadid10k": prepare_kadid,
    "cidiq": prepare_cidiq,
    "csiq": prepare_csiq,
    "csiq+": prepare_csiq_cornia,
    "tid2013": prepare_tid,
    "liveiqa": prepare_liveiqa,
    "nitsiqa": prepare_nitsiqa,
}

dataset_names = list(dataset_fn_dict.keys())

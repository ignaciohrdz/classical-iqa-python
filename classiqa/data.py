import pandas as pd
from pathlib import Path
import scipy
import random
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A

random.seed(420)


# TODO: keep working on this
def IqaDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        image_paths=None,
        image_scores=None,
        augment=True,
        crop_pct=0.90,
        flip_prob=0.5,
    ):
        """
        Arguments:
            data: DataFrame of dataset X (generated with the 'prepare_X' functions).
            image_paths: list of image paths
            image_scores: list of image scores (labels)
            augment: Apply simple augmentations (random crop and/or horizontal flip).
            crop_pct: Percentage of the image to crop. The crop should not be too agressive to ensure
                        most of the image's content is still present
            flip_prob: random horizontal flip probability
        """
        super(Dataset, self).__init__()
        self.data = data

        # Normal usage: pass a DataFrame with paths and labels (scores)
        # Special usage (cross validation): pass the list of paths and labels and no DataFrame
        if self.data:
            image_paths = self.data["image_path"].tolist()
            image_scores = self.data["image_scores"].tolist()
        self.image_paths = image_paths
        self.image_scores = image_scores

        self.augment = augment
        self.crop_pct = crop_pct
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        score = self.image_scores[idx].astype(np.float16)

        # Optional augmentation
        if self.augment:
            # Random crop
            img_h, img_w = image.shape[:2]
            crop_h, crop_w = [int(d * self.crop_pct) for d in image.shape[:2]]

            w_start = random.randint(0, img_w - crop_w)
            h_start = random.randint(0, img_h - crop_h)

            w_end = w_start + crop_w
            h_end = h_start + crop_h
            image = image[h_start:h_end, w_start:w_end, :]

            # Random flip
            if random.random() >= self.flip_prob:
                image = cv2.flip(image, 1)

        sample = {"image": image, "score": score}

        if self.transform:
            sample = self.transform(sample)

        return sample


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


dataset_fn_dict = {
    "koniq10k": prepare_koniq,
    "kadid10k": prepare_kadid,
    "csiq": prepare_csiq,
    "tid2013": prepare_tid,
    "liveiqa": prepare_liveiqa,
    "nitsiqa": prepare_nitsiqa,
    "cidiq": prepare_cidiq,
}

dataset_names = list(dataset_fn_dict.keys())

import cv2
import pickle
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import random
from .data import split_dataset
from pathlib import Path
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from .metrics import lcc, srocc
from .processing import zca_whitening

random.seed(420)


class SOM:
    """SOM: Semantic Obviousness Metric for Image Quality Assessment
    by Zhang et al. (https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_SOM_Semantic_Obviousness_2015_CVPR_paper.pdf)
    """

    def __init__(
        self,
        bing_path="classiqa/models/bing",
        max_regions=1000,
        num_samples=50,
        num_patches=5000,
        patch_sampling_step=50,
        patch_size=7,
        num_training_reps=3,
        codebook_size=10000,
        use_minibatch=True,
        nms_iou_thresh=0.8,
        test_size=0.2,
        codebook=None,
        svr_regressor=None,
    ):
        # SOM hyperparameters (suggested by the authors)
        self.max_regions = max_regions  # K in the paper
        self.num_samples = num_samples  # N in the paper
        self.num_patches = num_patches  # M in the paper
        self.patch_size = patch_size  # B in the paper
        self.num_training_reps = num_training_reps  # E in the paper
        self.codebook_size = codebook_size  # W in the paper
        self.nms_iou_thresh = nms_iou_thresh
        self.n_features = (2 * codebook_size) + max_regions

        # Some new parameters for my implementation
        self.patch_sampling_step = patch_sampling_step
        self.svr_regressor = svr_regressor
        self.test_size = test_size
        self.codebook = codebook
        self.use_minibatch = use_minibatch

        # Loading the saliency model
        self.saliency_model = cv2.saliency.ObjectnessBING.create()
        self.saliency_model.setTrainingPath(bing_path)

    def non_maximum_suppression(self, boxes):
        """From: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/"""

        if len(boxes) == 0:
            return []

        boxes = boxes.astype("float")

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # This is my small contribution: as we will later pick M patches of size BxB
        # I must make sure that the regions' short side is eaqual or greater than B
        short_side = np.minimum(x2 - x1 + 1, y2 - y1 + 1)
        idxs = np.delete(idxs, np.where(short_side < self.patch_size)[0])

        picked = []
        print("Number of detections before NMS: ", len(boxes))
        while (len(picked) < self.max_regions) and (len(idxs) > 0):
            last = len(idxs) - 1
            i = idxs[last]
            picked.append(i)

            # Compare box i to all the others
            other_idxs = idxs[:last]

            xx1 = np.maximum(x1[i], x1[other_idxs])
            yy1 = np.maximum(y1[i], y1[other_idxs])
            xx2 = np.minimum(x2[i], x2[other_idxs])
            yy2 = np.minimum(y2[i], y2[other_idxs])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[other_idxs]
            overlap_idxs = np.where(overlap > self.nms_iou_thresh)[0]

            # Delete the smaller boxes that overlap with box i
            idxs = np.delete(idxs, np.concatenate(([last], overlap_idxs)))

        boxes = boxes[picked].astype(int)
        print("Number of detections after NMS: ", len(boxes))

        return boxes

    def extract_objectness_regions(self, x):
        """Object-like region detection with BING
        Based on: https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/"""
        _, boxes = self.saliency_model.computeSaliency(x)
        boxes = np.squeeze(boxes)
        if boxes.dtype != "O":  # shape is D x 1 x 4 (D = num detected regions)
            # Not sure if I really need to do this. Is this already done by BING?
            # TODO: Leave it as something optional and explain in README
            # boxes = self.non_maximum_suppression(boxes)

            # We will also need the objectness scores
            # The implementation of BING in opencv-contrib should be returning
            # the scores in descending order, but the Python code seems to be doing it
            # in ascending order. See: https://github.com/opencv/opencv_contrib/issues/404
            objectness = self.saliency_model.getobjectnessValues()
            objectness = objectness[: self.max_regions]

            # For debugging
            # for b in boxes[: self.num_samples]:
            #     (startX, startY, endX, endY) = b.flatten()
            #     output = x.copy()
            #     color = np.random.randint(0, 255, size=(3,))
            #     color = [int(c) for c in color]
            #     cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
            #     cv2.imshow("Image", output)
            #     cv2.waitKey(0)
        else:
            # No detections for this image, so we'll sample patches
            #  from the whole image
            print("No object-like regions found")
            x_end = x.shape[1] - 1
            y_end = x.shape[0] - 1
            boxes = [[0, 0, x_end, y_end]]
            objectness = np.empty(0)

        # The objectness vector must be K
        if len(objectness) < self.max_regions:
            n_missing = self.max_regions - len(objectness)
            objectness = np.concatenate((objectness, np.zeros(n_missing)))

        objectness = objectness.astype(np.float16)

        return boxes, objectness

    def extract_patch_descriptor(self, x, boxes):
        """Once the objectness regions are cleaned with NMS, we sample
        M different BxB patches from those regions (see the SOM paper)"""
        patches = []

        # Pick one region randomly, and then some random coordinates
        # inside the regions to extract a BxB patch
        while len(patches) < self.num_patches:
            b = random.choice(boxes)
            max_b_x = b[2] - self.patch_size
            max_b_y = b[3] - self.patch_size
            x_start = np.random.uniform(
                low=b[0], high=max_b_x, size=self.patch_sampling_step
            ).astype(int)
            y_start = np.random.uniform(
                low=b[1], high=max_b_y, size=self.patch_sampling_step
            ).astype(int)
            x_end = x_start + self.patch_size
            y_end = y_start + self.patch_size
            for p in zip(x_start, x_end, y_start, y_end):
                patch_b = x[p[2] : p[3], p[0] : p[1]]
                patches.append(patch_b)

        # Flattened patches
        patches = np.stack(patches)
        ftrs = patches.reshape(patches.shape[0], -1)

        # Normalised patches
        eps = 1e-6  # to prevent NaNs
        mu, sigma = ftrs.mean(1), ftrs.std(1)
        ftrs = (ftrs - np.expand_dims(mu, -1)) / (np.expand_dims(sigma, -1) + eps)

        # ZCA whitening and converting to float16 for more efficiency
        zca_whitening(ftrs)
        ftrs = np.float16(ftrs)

        return ftrs

    def generate_codebook(self, dset):
        """In SOM, we oversample the training set to generate
        the K-word codebook (K clusters). This codebook only uses the
        local features, not the semantic objectness scores (X in the paper)"""

        local_ftrs = []
        paths = dset["image_path"].values
        for i, im_path in enumerate(paths):
            im_name = Path(im_path).name
            print(f"[Codebook][{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)

            # At this stage we don't need the objectness scores,
            # just the detections (the bounding boxes)
            objs, _ = self.extract_objectness_regions(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # We repeat the sampling and local feature extraction step
            # to increase the training set size
            for _ in range(self.num_training_reps):
                img_ftrs = self.extract_patch_descriptor(img_gray, objs)
                local_ftrs.append(img_ftrs)

        local_ftrs = np.concatenate(local_ftrs, axis=0)

        # Clustering
        print(
            f"Generating {self.codebook_size}-word codebook"
            f"(from {len(local_ftrs)} samples)"
        )
        if self.use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=self.codebook_size, random_state=420)
        else:
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=420)
        kmeans.fit(local_ftrs)
        self.codebook = np.float16(kmeans.cluster_centers_)  # [codebook_size x (B x B)]

    def extract_features(self, x):
        """Once we have the codebook, we can compute the final descriptor,
        which is a combination of the semantic obviousness (X) and the
        local characteristics (Z)
        F = [X, Z]
        """

        objs, semantic_ftrs = self.extract_objectness_regions(x)
        img_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        local_ftrs = self.extract_patch_descriptor(img_gray, objs)

        # Soft-assignment coding
        dot_similarity = np.dot(local_ftrs, self.codebook.T)
        encoding_pos = np.maximum(dot_similarity, 0)
        encoding_neg = np.maximum(-dot_similarity, 0)
        coefs = np.hstack((encoding_pos, encoding_neg))

        # Max pooling on the rows
        coefs = np.max(coefs, axis=0)

        # Feature fusion
        final_ftrs = np.concatenate((semantic_ftrs, coefs))

        return final_ftrs

    def generate_feature_db(self, dset):
        """Creates the feature database that will be used to fit the regressor
        :param dset: a DataFrame with columns [image_name, image_path, score, [img_set]]
                    (not all datasets have the img_set columns, only those that contain
                      groups of distorted images created from the same pristine source)
        """

        # Creating the train/test splits
        if "is_test" not in dset.columns:
            dset = split_dataset(dset, self.test_size)

        # We first create the codebook
        if not self.codebook:
            print("Generating the visual codebook with the training set")
            train_dset = dset.loc[dset["is_test"] == 0, :].copy()
            self.generate_codebook(train_dset)

        # Then, we compute the main features for every image
        feature_db = []
        for i, row in enumerate(dset.to_dict("records")):
            im_name = row["image_name"]
            im_path = row["image_path"]
            im_score = row["score"]
            im_split = row["is_test"]
            print(f"[Features][{i+1}/{len(dset)}]: Processing {im_name}")
            img = cv2.imread(im_path)
            ftrs = list(self.extract_features(img))
            feature_db.append([im_name] + ftrs + [im_score, im_split])

        feature_cols = list(range(1, self.n_features + 1))
        db_cols = ["image_name"] + feature_cols + ["MOS", "is_test"]
        feature_db = pd.DataFrame(feature_db, columns=db_cols)

        return feature_db

    def fit(self, feature_db, n_jobs=4):
        """
        Fit an SVR model to a given dset of ftrs
        :param feature_db: dataframe with the columns:
                    - image name
                    - feature i ... feature N
                    - MOS
                    - test split indicator
        :param n_jobs: number of threads for GridSearchCV
        :param test_size: test set size
        """

        # Making the splits
        train_mask = feature_db["is_test"] == 0
        test_mask = feature_db["is_test"] == 1
        feature_cols = feature_db.columns[1:-2]

        X_train = feature_db.loc[train_mask, feature_cols].values
        y_train = feature_db.loc[train_mask, "MOS"].values

        X_test = feature_db.loc[test_mask, feature_cols].values
        y_test = feature_db.loc[test_mask, "MOS"].values

        # X_train = np.float16(X_train)
        # X_test = np.float16(X_test)

        params = {
            "svr__C": np.arange(5, 10, 0.5),
            "svr__epsilon": np.arange(0.25, 2.0, 0.25),
        }

        # The authors of SOM used a linear kernel
        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), SVR(kernel="linear")),
            param_grid=params,
            cv=5,
            n_jobs=n_jobs,
            verbose=1,
            scoring={"LCC": make_scorer(lcc), "SROCC": make_scorer(srocc)},
            error_score=0,
            refit="SROCC",
        )

        print("Fitting an SVR for HOSA features")
        search.fit(X_train, y_train)
        self.svr_regressor = search.best_estimator_
        print(self.svr_regressor[1].C, self.svr_regressor[1].epsilon)

        # Test metrics
        y_pred = self.svr_regressor.predict(X_test)
        self.test_results = {
            "LCC": lcc(y_test, y_pred),
            "SROCC": srocc(y_test, y_pred),
        }

        return search.cv_results_

    def predict_score(self, f):
        """Predicts the score from a set of ftrs"""
        score = self.svr_regressor.predict(f)
        return score

    def export(self, path_save):
        # Exporting the model
        path_pkl = path_save / "estimator.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, x):
        ftrs = self.extract_features(x)
        ftrs = np.array(ftrs)

        # If we have loaded a SVR model, we predict the IQA score
        # The features are returned otherwise
        if self.svr_regressor is not None:
            return self.predict_score(ftrs.reshape(1, -1))
        else:
            return ftrs


if __name__ == "__main__":

    x = cv2.imread("../images/test_image_orig.jpg")
    som = SOM()
    som(x)

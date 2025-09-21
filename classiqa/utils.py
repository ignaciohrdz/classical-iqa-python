import argparse
from classiqa.data import dataset_names
from classiqa.sseq import SSEQ
from classiqa.gmlog import GMLOG
from classiqa.codebook import CORNIA, HOSA, LFA, SOM

import numpy as np
from .metrics import lcc, srocc
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
import pickle

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
        default="classiqa/models",
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
        default="classiqa/models",
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


class ScoreRegressor:
    """This is the simplest regressor. It fits a SVR given some precomputed features"""

    def __init__(self):
        self.svr_regressor = None

    def fit(self, feature_db, n_jobs=4):
        """
        Fit an SVR model to a given dset of features
        :param feature_db: dataframe with 15 columns:
                        - image name
                        - N features
                        - MOS
                        - split
                        - image set
        :param n_jobs: number of threads for GridSearchCV
        """

        train_mask = feature_db["is_test"] == 0
        test_mask = feature_db["is_test"] == 1
        feature_cols = feature_db.columns[1:-3]

        X_train = feature_db.loc[train_mask, feature_cols].values
        y_train = feature_db.loc[train_mask, "MOS"].values

        X_test = feature_db.loc[test_mask, feature_cols].values
        y_test = feature_db.loc[test_mask, "MOS"].values

        params = {
            "svr__C": np.arange(1.0, 10, 0.5),
            "svr__epsilon": np.arange(0.1, 2.0, 0.1),
        }

        # In datasets with artificial distortions, we must make sure that all the images
        # in a set are put in the same split. We do this with GroupKFold
        cv = GroupKFold(n_splits=5)
        image_sets = feature_db.loc[train_mask, "image_set"].tolist()

        search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(), SVR()),
            param_grid=params,
            cv=cv.split(X_train, y_train, groups=image_sets),
            n_jobs=n_jobs,
            verbose=1,
            scoring={"LCC": make_scorer(lcc), "SROCC": make_scorer(srocc)},
            error_score=0,
            refit="SROCC",
        )

        search.fit(X_train, y_train)
        best_svr = search.best_estimator_[1]
        best_c = best_svr.C
        best_eps = best_svr.epsilon

        # Fittting the SVR on the entire training set
        print(f"Training the SVR with the best params (C={best_c}, eps={best_eps:.2f})")
        self.svr_regressor = make_pipeline(
            StandardScaler(), SVR(C=best_c, epsilon=best_eps)
        )
        self.svr_regressor.fit(X_train, y_train)

        # Test metrics
        y_pred = self.svr_regressor.predict(X_test)
        test_results = {
            "LCC": lcc(y_test, y_pred),
            "SROCC": srocc(y_test, y_pred),
        }

        return search.cv_results_, test_results

    def export(self, path_save):
        path_pkl = path_save / "score_regressor.pkl"
        print("Saving best SVR model to ", str(path_pkl))
        with open(path_pkl, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_score(self, f):
        """Predicts the score from a set of features (f)"""
        score = self.svr_regressor.predict(f)
        return score

    def __call__(self, f):
        score = None
        if self.svr_regressor:
            score = self.predict_score(f)
        return score

import numpy as np
from .metrics import lcc, srocc
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
import pickle


# TODO: Implement a NN regressor with sklearn + our IqaDataset class (to enable data augmentation)


class ScoreRegressor:
    """This is the simplest regressor. It fits a SVR given some precomputed features"""

    def __init__(self, epsilon=0.0):
        self.svr_regressor = None
        self.epsilon = epsilon

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

        # Using similar ranges to those used by the GM-LOG authors
        params = {
            "model__C": [pow(10.0, i) for i in np.arange(-4, 5)],
            "model__gamma": [pow(10.0, i) for i in np.arange(-4, 1)],
        }

        if not self.epsilon:
            params["model__epsilon"] = np.arange(0.1, 1.2, 0.2)

        # In datasets with artificial distortions, we must make sure that all the images
        # in a set are put in the same split. We do this with GroupKFold
        cv = GroupKFold(n_splits=5)
        image_sets = feature_db.loc[train_mask, "image_set"].tolist()
        folds_crossval = cv.split(X_train, y_train, groups=image_sets)

        is_linear = False
        if len(X_train) > 10000:
            svr = LinearSVR(max_iter=5000, epsilon=self.epsilon)
            del params["model__gamma"]
            is_linear = True
        else:
            svr = SVR(kernel="rbf", epsilon=self.epsilon, max_iter=5000)

        # Scale the output for higher stability
        self.output_scaler = StandardScaler()
        y_train = self.output_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = self.output_scaler.transform(y_test.reshape(-1, 1)).ravel()

        estimator = Pipeline(steps=[("scaler", StandardScaler()), ("model", svr)])
        search = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            cv=folds_crossval,
            n_jobs=n_jobs,
            verbose=1,
            scoring={"LCC": make_scorer(lcc), "SROCC": make_scorer(srocc)},
            error_score=0,
            refit="SROCC",
        )
        search.fit(X_train, y_train)

        # After running search.fit, we will already have the best model trained
        # on the entire training set (because we used the 'refit' argument)
        self.svr_regressor = search.best_estimator_
        best_params = {
            "C": self.svr_regressor[1].C,
            "epsilon": self.svr_regressor[1].epsilon,
        }
        if not is_linear:
            best_params["gamma"] = self.svr_regressor[1].gamma
        print("Best params: ", best_params)

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
        score = self.output_scaler.inverse_transform(score.reshape(-1, 1)).ravel()
        return score

    def __call__(self, f):
        score = None
        if self.svr_regressor:
            score = self.predict_score(f)
        return score

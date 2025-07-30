"""Train a SVR on IQA datasets"""

from classiqa.utils import extract_train_args, resolve_model
from classiqa.data import dataset_fn_dict
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    args = extract_train_args()

    args.path_datasets = "/home/ignaciohmon/Datasets/iqa_datasets"
    args.use_dataset = "csiq"
    args.model = "lfa"
    args.overwrite = True

    # Define the score without a regressor to obtain the features
    estimator = resolve_model(args.model)
    n_features = estimator.n_features
    path_model = Path(args.path_models) / args.model / args.use_dataset
    path_model.mkdir(exist_ok=True, parents=True)

    path_dataset = Path(args.path_datasets) / args.use_dataset
    path_feature_db = path_dataset / f"feature_db_{args.model}.csv"

    if not path_feature_db.exists() or args.overwrite:
        # We generate the feature database
        # that will be used to fit an SVR later
        dataset = dataset_fn_dict[args.use_dataset](path_dataset)
        feature_db = estimator.generate_feature_db(dataset)
        feature_db.to_csv(path_feature_db, index=False)
    else:
        print("Found feature database. Loading...")
        feature_db = pd.read_csv(path_feature_db).fillna(0)

    print("Fitting an SVR model...")
    results = estimator.fit(feature_db)
    results = pd.DataFrame(results)
    results.to_csv(path_model / f"{args.use_dataset}_results.csv", index=False)
    print("Test results: ", estimator.test_results)

    # Exporting the IQA model
    estimator.export(path_model)

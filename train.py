"""Train a SVR on IQA datasets"""

from classiqa.utils import (
    extract_train_args,
    resolve_model,
    export_results,
)
from classiqa.regression import regressor_dict
from classiqa.data import dataset_fn_dict
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    args = extract_train_args()

    args.path_datasets = "/home/ignaciohmon/projects/datasets/iqa_datasets"
    args.use_dataset = "nitsiqa"
    args.model = "lfa"
    args.overwrite = True

    # Define the score without a regressor to obtain the features
    feature_extractor = resolve_model(args.model, args.img_size)
    n_dims = 200 if args.model == "lfa" else 0
    regressor = regressor_dict[args.regressor](pca_components=n_dims)
    n_features = feature_extractor.n_features

    if args.use_dataset == "csiq" and args.model in ["cornia", "lfa", "hosa", "som"]:
        print(
            "[ERROR]: You can't use this dataset to train this measure"
            " because it's used for codebook construction"
        )
        exit()

    path_model = Path(args.path_models) / args.model / args.regressor / args.use_dataset
    path_model.mkdir(exist_ok=True, parents=True)

    path_dataset = Path(args.path_datasets) / args.use_dataset
    path_feature_db = path_dataset / f"feature_db_{args.model}.csv"
    print(f"Training a {args.model} (+ {args.regressor}) model on {args.use_dataset}")

    if not path_feature_db.exists() or args.overwrite:
        # We generate the feature database
        # that will be used to fit an SVR later
        dataset = dataset_fn_dict[args.use_dataset](path_dataset)

        if args.model in ["cornia", "lfa", "hosa"]:
            # We will use the extended CSIQ dataset to generate the codebook
            print(f"Generating the {args.model.upper()} with CSIQ+")
            path_csiq = Path(args.path_datasets) / "csiq"
            codebook_dset = dataset_fn_dict["csiq+"](path_csiq)
            feature_extractor.generate_codebook(codebook_dset)

        feature_db = feature_extractor.generate_feature_db(dataset)
        feature_db.to_csv(path_feature_db, index=False)
    else:
        print("Found feature database. Loading...")
        feature_db = pd.read_csv(path_feature_db).fillna(0)

    print(f"Fitting an {args.regressor} model...")
    results, corr_metrics = regressor.fit(feature_db)
    results = pd.DataFrame(results)
    results.to_csv(path_model / f"{args.use_dataset}_results.csv", index=False)
    print("Test results: ", results)
    print("Test metrics: ", corr_metrics)

    # Exporting the model
    feature_extractor.export(path_model)
    regressor.export(path_model)

    # Saving the results
    path_json = Path("results.json")
    export_results(
        path_json, args.use_dataset, args.model, args.regressor, corr_metrics
    )

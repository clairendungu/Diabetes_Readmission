"""Train the final 30-day readmission model and save it to a file.

Usage:
    python train.py                              # uses data/diabetic_data.csv and best_params.json
    python train.py --model "random forest"      # override the model chosen in the notebook
    python train.py --data path/to/diabetic_data.csv --output model.bin

The notebook writes best_params.json (chosen model and tuned parameters). If that
file is missing, sensible default parameters for XGBoost are used instead.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import features

RANDOM_STATE = 42
TEST_SIZE = 0.2
FLAG_FRACTION = 0.20  # flag the top 20% highest-risk patients as "high risk"


def make_preprocessor(kind, cat_features, num_features):
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30)
    if kind == "linear":
        skewed = [c for c in features.SKEWED if c in num_features]
        other_num = [c for c in num_features if c not in skewed]
        return ColumnTransformer([
            ("skewed", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), skewed),
            ("num", StandardScaler(), other_num),
            ("cat", cat, cat_features),
        ])
    return ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", cat, cat_features),
    ])


def build_pipeline(model_name, params, cat_features, num_features, scale_pos_weight):
    if model_name == "logistic regression":
        pipe = Pipeline([
            ("prep", make_preprocessor("linear", cat_features, num_features)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    elif model_name == "random forest":
        pipe = Pipeline([
            ("prep", make_preprocessor("tree", cat_features, num_features)),
            ("model", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=20, class_weight="balanced_subsample",
                n_jobs=-1, random_state=RANDOM_STATE)),
        ])
    elif model_name == "xgboost":
        pipe = Pipeline([
            ("prep", make_preprocessor("tree", cat_features, num_features)),
            ("model", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)),
        ])
    else:
        raise ValueError(f"Unknown model '{model_name}'. "
                         "Choose: logistic regression, random forest, xgboost.")
    return pipe.set_params(**params)


def load_config(params_path):
    path = Path(params_path)
    if path.exists():
        return json.loads(path.read_text())
    print(f"Warning: {path} not found, using default XGBoost parameters.")
    return {"model": "xgboost", "best_params": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/diabetic_data.csv")
    parser.add_argument("--params", default="best_params.json")
    parser.add_argument("--output", default="model.bin")
    parser.add_argument("--model", default=None, help="override the model name from best_params.json")
    args = parser.parse_args()

    config = load_config(args.params)
    model_name = args.model or config["model"]
    # Tuned parameters only apply to the model they were tuned for
    params = config["best_params"] if model_name == config["model"] else {}

    print("Loading and cleaning data...")
    df = features.clean(features.load_raw(args.data))
    rare_drugs = features.find_rare_drugs(df)
    X = features.build_features(df, rare_drugs)
    y = df["readmit_30"]
    cat_features, num_features = features.split_feature_types(X.columns)
    print(f"{len(df)} patients, {len(X.columns)} features, positive rate {y.mean():.3f}")

    # Same split as the notebook (stratified 80/20, random_state=42)
    X_full, X_test, y_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scale_pos_weight = (y_full == 0).sum() / (y_full == 1).sum()
    pipeline = build_pipeline(model_name, params, cat_features, num_features, scale_pos_weight)

    print(f"Training {model_name}...")
    pipeline.fit(X_full, y_full)

    p_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, p_test)),
        "test_pr_auc": float(average_precision_score(y_test, p_test)),
        "test_positive_rate": float(y_test.mean()),
    }
    # Patients scoring at or above this value are flagged as high risk (top 20% on the test set)
    threshold = float(np.quantile(p_test, 1 - FLAG_FRACTION))
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.3f}   Test PR-AUC: {metrics['test_pr_auc']:.3f} "
          f"(random baseline {metrics['test_positive_rate']:.3f})")
    print(f"High-risk threshold (top {int(FLAG_FRACTION * 100)}%): {threshold:.4f}")

    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "features": list(X.columns),
        "rare_drugs": rare_drugs,
        "threshold": threshold,
        "flag_fraction": FLAG_FRACTION,
        "metrics": metrics,
    }
    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

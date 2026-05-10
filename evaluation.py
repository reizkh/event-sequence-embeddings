import torch
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import mlflow
import lightgbm as lgb


def run_classifier_cv(
    cv_vector_dataset: np.ndarray,
    cv_labels: np.ndarray,
) -> None:
    model = CatBoostClassifier(
        verbose=0,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    cv_results = cross_validate(
        model, # type: ignore
        cv_vector_dataset,
        cv_labels,
        scoring="roc_auc"
    )

    mlflow.log_metric("global_CV_rocauc", cv_results["test_score"].mean())

def run_local_cv(
    cv_vector_dataset: np.ndarray,
    cv_labels: np.ndarray
) -> None:
    amount_model = CatBoostRegressor(
        verbose=0,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    amount_cv_results = cross_validate(
        amount_model, # type: ignore
        cv_vector_dataset,
        cv_labels[:, 0],
        scoring="neg_root_mean_squared_error"
    )
    mlflow.log_metric("local_CV_logamount_rmse", -amount_cv_results["test_score"].mean())
    var = cv_labels[:, 0].var()
    mlflow.log_metric("local_CV_logamount_r2", (1 + amount_cv_results["test_score"]/var).mean())

    mcc_model = LogisticRegression()
    cv_labels[:, 1:2] = OrdinalEncoder(max_categories=10).fit_transform(cv_labels[:, 1:2]) # type: ignore
    mcc_cv_results = cross_validate(
        mcc_model, # type: ignore
        cv_vector_dataset,
        cv_labels[:, 1],
        scoring="roc_auc_ovr_weighted",
        cv=StratifiedKFold()
    )
    mlflow.log_metric("local_CV_mcc_rocauc", mcc_cv_results["test_score"].mean())


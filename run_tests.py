from training import train_encoder, run_classifier_cv
from dataset import load_and_split_data, create_vector_dataset

import dagshub
import mlflow
import torch
from scipy.stats import uniform, randint
from numpy import logspace
from sklearn.metrics import roc_auc_score


hyperparams = {
    "embedding_size": 128,
    "category_embedding_size": 128,
    "num_epochs": 80,
    "margin": 0.5,
    "learning_rate": 6e-4,
    "weight_decay": 5e-6,
    "n_samples_in_batch": 64,
    "subseq_min": 15,
    "subseq_max": 150,
    "k": 5,
    "cat_features": ["MCC", "trx_category"],
    "cat_coverage": 0.99,
    "classifier": "logistic_regression",
    "optimizer": "Adam"
}
clf_hyperparams = {
    "logistic_regression": {
        "C": logspace(-3, 2, 10)
    },
    "lightgbm": {
        "n_estimators": randint(10, 1000),
        "max_depth": randint(3, 15),
        "learning_rate": logspace(-3, -1),
        "num_leaves": randint(10, 100),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(15, 100),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1)
    },
    "catboost": {
        "depth": randint(3, 10),
        "learning_rate": logspace(-3, -1),
        "l2_leaf_reg": randint(1, 100),
        "bagging_temperature": uniform(0, 10),
        "random_strength": uniform(0, 10),
        "border_count": randint(32, 255)
    }
}

rounds = 5
dagshub.init("event-sequence-embeddings", "reizkh")
device = "cuda" if torch.cuda.is_available() else "cpu"
mlflow.set_experiment("Final baseline testing")
for _ in range(rounds):
    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)
        enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
            "pytorch-lifestream/rosbank-churn", 
            "pytorch-lifestream/rosbank-churn",
            cat_features=hyperparams["cat_features"],
            cat_coverage=hyperparams["cat_coverage"],
            random_state=None
        )
        encoder = train_encoder(
            enc_train_dataset,
            enc_val_dataset,
            vocab_sizes,
            hyperparams,
            mlflow_run=run
        )
        cv_vector_dataset, cv_labels = create_vector_dataset(
            encoder,
            classifier_cv_dataset,
            hyperparams["embedding_size"],
            device
        )
        clf = run_classifier_cv(
            cv_vector_dataset,
            cv_labels,
            hyperparams,
            clf_hyperparams=clf_hyperparams
        )

        test_vector_dataset, test_labels = create_vector_dataset(
            encoder,
            test_dataset,
            hyperparams["embedding_size"],
            device
        )

        y_pred = clf.predict_proba(test_vector_dataset)[:,1] # type: ignore
        roc_auc = roc_auc_score(test_labels, y_pred)
        mlflow.log_metric("test_roc_auc", float(roc_auc))


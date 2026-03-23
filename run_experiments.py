from dataset import load_and_split_data, create_vector_dataset
from training import train_encoder, run_classifier_cv

import mlflow
import dagshub
import torch
from sklearn.model_selection import ParameterSampler
from numpy import logspace, linspace
from tqdm.auto import tqdm
from scipy.stats import randint, uniform


cat_coverage = 0.95
cat_features = ["MCC", "trx_category"]
enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
    "pytorch-lifestream/rosbank-churn", 
    "pytorch-lifestream/rosbank-churn",
    cat_features=cat_features,
    cat_coverage=cat_coverage
)

hyperparams_distributions = {
    "embedding_size": [128],
    "category_embedding_size": [128],
    "num_epochs": [15, 30, 60],
    "margin": linspace(0.1, 2.0),
    "learning_rate": linspace(1e-3, 1e-2),
    "weight_decay": logspace(-6, -4),
    "n_samples_in_batch": [64, 128, 256],
    "subseq_min": [5, 10, 15, 20, 25, 30],
    "subseq_max": [100, 125, 150, 175, 200, 250],
    "k": [3, 4, 5, 6],
    "cat_features": [cat_features],
    "cat_coverage": [cat_coverage],
    "classifier": ["logistic_regression"],
    "optimizer": ["Adam"]
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

sampler = ParameterSampler(
    param_distributions=hyperparams_distributions,
    n_iter=15
)

device = "cuda" if torch.cuda.is_available() else "cpu"

dagshub.init("event-sequence-embeddings", "reizkh")
# mlflow.config.enable_async_logging()
# mlflow.set_experiment("Using multiple features")
for hyperparams in tqdm(sampler, desc="Random search of hyperparameters", leave=False):
    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)

        best_encoder = train_encoder(
            enc_train_dataset,
            enc_val_dataset,
            vocab_sizes,
            hyperparams,
            mlflow_run=run
        )

        cv_vector_dataset, cv_labels = create_vector_dataset(
            best_encoder,
            classifier_cv_dataset,
            hyperparams["embedding_size"],
            device
        )

        clf = run_classifier_cv(
            cv_vector_dataset,
            cv_labels,
            hyperparams,
            clf_hyperparams
        )
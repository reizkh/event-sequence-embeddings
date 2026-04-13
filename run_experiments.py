from dataset import load_and_split_data, create_vector_dataset
from training import train_encoder, run_classifier_cv
from encoder import LSTMEncoder

import mlflow
import dagshub
import torch
from sklearn.model_selection import ParameterSampler
from numpy import logspace, linspace
from tqdm.auto import tqdm
from scipy.stats import randint, uniform


cat_coverage = 0.99
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
    "num_epochs": [80],
    "margin": [0.5],
    "alpha": logspace(-2, 0, 6),
    "threshold": linspace(0.5, 1.0, 5),
    "learning_rate": [9e-4],
    "weight_decay": [2e-5],
    "n_samples_in_batch": [64],
    "subseq_min": [15],
    "subseq_max": [150],
    "k": [5],
    "cat_features": [cat_features],
    "cat_coverage": [cat_coverage],
    "classifier": ["catboost"],
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
    "catboost": {}
}

sampler = ParameterSampler(
    param_distributions=hyperparams_distributions,
    n_iter=30
)

device = "cuda" if torch.cuda.is_available() else "cpu"

dagshub.init("event-sequence-embeddings", "reizkh")

run_id = "afe3cbce965b4decb7db2e76546401ed"
checkpoint_path = "model_checkpoint.pth"

with mlflow.start_run(run_id=run_id) as mlflow_run:
    path = mlflow.artifacts.download_artifacts(
        run_id=mlflow_run.info.run_id,
        artifact_path="models/best_model/" + checkpoint_path
    )
    encoder = LSTMEncoder(
        cat_vocab_sizes=vocab_sizes,
        hidden_size=int(mlflow_run.data.params["embedding_size"]),
        cat_embedding_dims=[128, 128]
    ).to(device)
    encoder.load_state_dict(torch.load(path, map_location=device))

    vector_dataset, labels = create_vector_dataset(
        encoder,
        enc_train_dataset,
        int(mlflow_run.data.params["embedding_size"]),
        device
    )

# mlflow.config.enable_async_logging()
mlflow.set_experiment("CoLES with soft labels")
for hyperparams in tqdm(sampler, desc="Random search of hyperparameters", leave=False):
    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)

        best_encoder = train_encoder(
            enc_train_dataset,
            enc_val_dataset,
            vocab_sizes,
            hyperparams,
            run,
            torch.tensor(vector_dataset).to(device)
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
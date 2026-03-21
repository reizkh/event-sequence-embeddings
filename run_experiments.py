from dataset import load_and_split_data, create_vector_dataset
from training import train_encoder, run_classifier_cv

import mlflow
import dagshub
import torch
from sklearn.model_selection import ParameterSampler
from numpy import logspace, linspace
from tqdm.auto import tqdm

enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_size = load_and_split_data(
    "pytorch-lifestream/rosbank-churn", 
    "pytorch-lifestream/rosbank-churn"
)

hyperparams_distributions = {
    "embedding_size": [128],
    "category_embedding_size": [128],
    "num_epochs": [15, 30, 60],
    "margin": linspace(0.1, 0.6),
    "learning_rate": logspace(1e-4, 1e-1),
    "weight_decay": logspace(1e-6, 1e-2),
    "n_samples_in_batch": [64],
    "subseq_min": [5, 10, 15, 20, 25, 30],
    "subseq_max": [100, 125, 150, 175, 200, 250],
    "k": [3, 4, 5, 6],
    "vocab_size": [vocab_size],
    "classifier": ["logistic_regression"],
    "optimizer": ["SGD", "Adam"]
}

sampler = ParameterSampler(
    param_distributions=hyperparams_distributions,
    n_iter=10
)

device = "cuda" if torch.cuda.is_available() else "cpu"

for hyperparams in tqdm(sampler, desc="Random search of hyperparameters"):
    dagshub.init("event-sequence-embeddings", "reizkh")
    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)

        best_encoder = train_encoder(
            enc_train_dataset,
            enc_val_dataset,
            vocab_size,
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
            hyperparams
        )
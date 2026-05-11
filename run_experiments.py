from dataset import load_and_split_data, create_global_dataset, create_local_dataset
from training import train_encoder
from evaluation import run_classifier_cv, run_local_cv

import mlflow
import dagshub
import torch
from sklearn.model_selection import ParameterSampler
from numpy import logspace, linspace
from tqdm.auto import tqdm
from scipy.stats import randint, uniform


import dotenv

dotenv.load_dotenv()


hyperparams_distributions = {
    "embedding_size": [128],
    "category_embedding_size": [128],
    "num_epochs": [2],
    "margin": [0.5],
    "learning_rate": [1e-3],
    "weight_decay": [2e-5],
    "n_samples_in_batch": [64],
    "subseq_min": [15],
    "subseq_max": [150],
    "k": [5],
    "cat_features": [["MCC", "trx_category"]],
    "cat_coverage": [0.9],
    "classifier": ["logistic_regression"],
    "optimizer": ["Adam"],
    "add_sep": [False],
    "cmlm_lambda": [0.0, 0.001, 0.1],
    "mi_bound_lambda": [0.1, 0.5],
    "club_lr_ratio": [1, 3],
    "mask_pr": [0.01],
    "club_pr": [0.01]
}

enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
    "pytorch-lifestream/rosbank-churn", 
    "pytorch-lifestream/rosbank-churn",
    cat_features=hyperparams_distributions["cat_features"][0],
    cat_coverage=hyperparams_distributions["cat_coverage"][0],
    add_sep=hyperparams_distributions["add_sep"][0]
)

sampler = ParameterSampler(
    param_distributions=hyperparams_distributions,
    n_iter=1
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# dagshub.init("event-sequence-embeddings", "reizkh")
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
            run
        )

        global_eval_ds, global_eval_labels = create_global_dataset(
            best_encoder,
            classifier_cv_dataset,
            device
        )
        run_classifier_cv(
            global_eval_ds,
            global_eval_labels
        )

        local_eval_ds, local_eval_labels = create_local_dataset(
            best_encoder,
            classifier_cv_dataset,
            device,
            window_stride=8,
            sep_events=hyperparams["add_sep"]
        )
        run_local_cv(
            local_eval_ds,
            local_eval_labels
        )
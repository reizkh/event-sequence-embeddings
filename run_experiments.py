from dataset import load_and_split_data, create_global_dataset, create_local_dataset
from training import train_encoder
from evaluation import run_classifier_cv, run_local_cv

import mlflow
import dagshub
import torch
import dotenv
import yaml
from sklearn.model_selection import ParameterGrid

dotenv.load_dotenv()

with open("cv_param_grid.yaml") as f:
    param_grid = list(ParameterGrid(yaml.safe_load(f)))


enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
    "pytorch-lifestream/rosbank-churn", 
    "pytorch-lifestream/rosbank-churn",
    cat_features=param_grid[0]["cat_features"],
    cat_coverage=param_grid[0]["cat_coverage"],
    add_sep=param_grid[0]["add_sep"]
)

dagshub.init("event-sequence-embeddings", "reizkh")
device = "cuda" if torch.cuda.is_available() else "cpu"
rounds = 1
for hyperparams in param_grid:
    for _ in range(rounds):
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
            )
            run_local_cv(
                local_eval_ds,
                local_eval_labels
            )
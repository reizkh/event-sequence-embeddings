from training import train_encoder, train_downstream_models
from dataset import load_and_split_data, create_global_dataset, create_local_dataset

import dagshub
import mlflow
import torch
import dotenv
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import yaml
from sklearn.model_selection import ParameterGrid

dotenv.load_dotenv()

with open("eval_param_grid.yaml") as f:
    param_grid = list(ParameterGrid(yaml.safe_load(f)))

enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
    "pytorch-lifestream/rosbank-churn", 
    "pytorch-lifestream/rosbank-churn",
    cat_features=param_grid[0]["cat_features"],
    cat_coverage=param_grid[0]["cat_coverage"],
    add_sep=param_grid[0]["add_sep"]
)

dagshub.init("event-sequence-embeddings", "reizkh")
mlflow.config.enable_async_logging()
mlflow.set_experiment("CoLES+CMLM / LSTM+1-MLP eval")
device = "cuda" if torch.cuda.is_available() else "cpu"
rounds = 5
for hyperparams in param_grid:
    for _ in range(rounds):
        with mlflow.start_run() as run:
            mlflow.log_params(hyperparams)
            encoder = train_encoder(
                enc_train_dataset,
                enc_val_dataset,
                vocab_sizes,
                hyperparams,
                mlflow_run=run
            )
            global_vector_dataset, global_labels = create_global_dataset(
                encoder,
                classifier_cv_dataset,
                device
            )
            local_vector_dataset, local_labels = create_local_dataset(
                encoder,
                classifier_cv_dataset,
                device
            )
            test_global_dataset, test_global_labels = create_global_dataset(
                encoder,
                test_dataset,
                device
            )
            test_local_dataset, test_local_labels = create_local_dataset(
                encoder,
                test_dataset,
                device
            )

            label_transform = OrdinalEncoder(max_categories=50) # type: ignore
            local_labels[:, 1:] = label_transform.fit_transform(local_labels[:, 1:])
            test_local_labels[:, 1:] = label_transform.transform(test_local_labels[:, 1:])

            models = train_downstream_models(
                local_vector_dataset,
                global_vector_dataset,
                local_labels,
                global_labels,
            )

            y_pred = models[0].predict_proba(test_global_dataset)[:,1]
            roc_auc = roc_auc_score(test_global_labels, y_pred)
            mlflow.log_metric("global_test_rocauc", float(roc_auc))

            y_pred = models[1].predict(test_local_dataset)
            rmse = root_mean_squared_error(test_local_labels[:, 0], y_pred)
            mlflow.log_metric("local_test_logamount_rmse", rmse)

            y_pred = models[2].predict_proba(test_local_dataset)
            roc_auc = roc_auc_score(test_local_labels[:, 1], y_pred, average="weighted", multi_class="ovr")
            mlflow.log_metric("local_test_mcc_rocauc", float(roc_auc))
from training import train_encoder, train_downstream_models
from dataset import load_and_split_data, create_global_dataset, create_local_dataset

import dagshub
import mlflow
import torch
from scipy.stats import uniform, randint
import numpy as np
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder


hyperparams = {
    "embedding_size": 128,
    "category_embedding_size": 128,
    "num_epochs": 1,
    "margin": 0.5,
    "learning_rate": 1e-3,
    "weight_decay": 2e-5,
    "n_samples_in_batch": 64,
    "subseq_min": 15,
    "subseq_max": 150,
    "k": 5,
    "cat_features": ["MCC", "trx_category"],
    "cat_coverage": 0.99,
    "optimizer": "Adam",
    "add_sep": False,
    "club_lr_ratio": 1,
    "cmlm_lambda": 0.1,
    "mi_bound_lambda": 0.1
}
enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
    "pytorch-lifestream/rosbank-churn", 
    "pytorch-lifestream/rosbank-churn",
    cat_features=hyperparams["cat_features"],
    cat_coverage=hyperparams["cat_coverage"]
)

# dagshub.init("event-sequence-embeddings", "reizkh")
device = "cuda" if torch.cuda.is_available() else "cpu"

rounds = 1
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

        label_transform = OrdinalEncoder(max_categories=10) # type: ignore
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
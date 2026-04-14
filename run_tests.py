from dataset import load_and_split_data, create_vector_dataset
from training import train_encoder, run_classifier_cv
from encoder import LSTMEncoder

import mlflow
import mlflow.artifacts
import dagshub
import torch
from sklearn.metrics import roc_auc_score
from numpy import logspace
from scipy.stats import randint, uniform
import dotenv


dotenv.load_dotenv(".env")


cat_coverage = 0.99
cat_features = ["MCC", "trx_category"]
enc_train_dataset, enc_val_dataset, classifier_cv_dataset, test_dataset, vocab_sizes = load_and_split_data(
    "pytorch-lifestream/rosbank-churn",
    "pytorch-lifestream/rosbank-churn",
    cat_features=cat_features,
    cat_coverage=cat_coverage
)

hyperparams = {
    "embedding_size": 128,
    "category_embedding_size": 128,
    "num_epochs": 80,
    "margin": 0.5,
    "learning_rate": 9e-4,
    "weight_decay": 2e-5,
    "n_samples_in_batch": 64,
    "subseq_min": 15,
    "subseq_max": 150,
    "k": 5,
    "cat_features": cat_features,
    "cat_coverage": cat_coverage,
    "classifier": "catboost",
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
    "catboost": {}
}


device = "cuda" if torch.cuda.is_available() else "cpu"
rounds = 5
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

mlflow.set_experiment("Baseline test (CoLES) catboost")
for _ in range(rounds):
    with mlflow.start_run() as run:
        mlflow.log_params(hyperparams)

        encoder = train_encoder(
            enc_train_dataset,
            enc_val_dataset,
            vocab_sizes,
            hyperparams,
            run,
            torch.tensor(vector_dataset).to(device)
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
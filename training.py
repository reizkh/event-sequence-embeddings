from dataset import ClientTransactionsDataset, random_slices_collate_fn, create_vector_dataset
from encoder import LSTMEncoder
from loss import soft_contrastive_loss_euclidean

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np
import mlflow
import mlflow.artifacts
from tqdm.auto import trange, tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint


class LGBMWithEarlyStopping(lgb.LGBMClassifier):
    def __init__(self, val_split=0.2, early_stopping_rounds=50, **kwargs):
        super().__init__(**kwargs)
        self.val_split = val_split
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, **fit_params):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=42, stratify=y if hasattr(y, 'dtype') else None
        )
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["eval_metric"] = "auc"

        return super().fit(X_train, y_train, **fit_params)


def train_encoder(
    train_dataset: ClientTransactionsDataset,
    val_dataset: ClientTransactionsDataset,
    vocab_sizes: List[int],
    hyperparams: Dict[str, Any],
    mlflow_run: Any,
    dataset_embeddings: torch.Tensor,
    checkpoint_path: str = "model_checkpoint.pth"
) -> LSTMEncoder:
    dataset_embeddings = F.normalize(dataset_embeddings)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = LSTMEncoder(
        cat_vocab_sizes=vocab_sizes,
        cat_embedding_dims=[hyperparams["category_embedding_size"]] * len(vocab_sizes), 
        hidden_size=hyperparams["embedding_size"]
    ).to(device)

    if hyperparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            encoder.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )
    elif hyperparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer type: {hyperparams["optimizer"]}")

    collate_fn = lambda batch: random_slices_collate_fn(
        batch, 
        hyperparams["subseq_min"], 
        hyperparams["subseq_max"], 
        hyperparams["k"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparams["n_samples_in_batch"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hyperparams["n_samples_in_batch"],
        collate_fn=collate_fn,
        drop_last=True
    )

    best_loss = float('inf')

    for epoch in trange(hyperparams["num_epochs"], desc="Epoch"):
        # --- Training Phase ---
        encoder.train()
        total_train_loss = 0.0
        
        for ids, transactions, lengths in tqdm(train_loader, leave=False, desc="Training"):
            optimizer.zero_grad()

            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                transactions, lengths=lengths, batch_first=True, enforce_sorted=False
            ).to(device)
            
            embeddings = encoder(packed_inputs)
            loss = soft_contrastive_loss_euclidean(
                ids,
                embeddings,
                dataset_embeddings,
                hyperparams["margin"],
                hyperparams["alpha"],
                hyperparams["threshold"]
            )
            loss.backward()
            
            loss_value = loss.item()
            total_train_loss += loss_value

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        mlflow.log_metric("avg_train_epoch_loss", avg_train_loss, step=epoch)

        # --- Validation Phase ---
        encoder.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for ids, transactions, lengths in tqdm(val_loader, leave=False, desc="Validation"):
                packed_inputs = nn.utils.rnn.pack_padded_sequence(
                    transactions, lengths=lengths, batch_first=True, enforce_sorted=False
                ).to(device)
                embeddings = encoder(packed_inputs)
                loss = soft_contrastive_loss_euclidean(
                    ids,
                    embeddings,
                    dataset_embeddings,
                    hyperparams["margin"],
                    hyperparams["alpha"],
                    hyperparams["threshold"]
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        mlflow.log_metric("avg_val_epoch_loss", avg_val_loss, step=epoch)

        # --- Checkpointing ---
        torch.save(encoder.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path=f"models/epoch_{epoch}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            mlflow.log_metric("best_loss", best_loss, step=epoch)
            mlflow.log_artifact(checkpoint_path, artifact_path="models/best_model")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # --- Loading best checkpoint ---
    path = mlflow.artifacts.download_artifacts(
        run_id=mlflow_run.info.run_id,
        artifact_path="models/best_model/" + checkpoint_path
    )
    best_encoder = LSTMEncoder(
        cat_vocab_sizes=vocab_sizes,
        cat_embedding_dims=[hyperparams["category_embedding_size"]] * len(vocab_sizes), 
        hidden_size=hyperparams["embedding_size"]
    ).to(device)
    best_encoder.load_state_dict(torch.load(path, map_location=device))
    os.remove(path)
        
    return best_encoder

def run_classifier_cv(
    cv_vector_dataset: np.ndarray,
    cv_labels: List,
    hyperparams: Dict[str, Any],
    clf_hyperparams: Dict[str, Any],
    n_iter: int = 15
) -> BaseEstimator:
    param_distributions = clf_hyperparams[hyperparams["classifier"]]

    if hyperparams["classifier"] == "logistic_regression":
        model = LogisticRegression()
    elif hyperparams["classifier"] == "lightgbm":
        model = LGBMWithEarlyStopping(
            objective="binary",
            n_jobs=-1,
            verbose=-1
        )
    elif hyperparams["classifier"] == "catboost":
        model = CatBoostClassifier(
            verbose=0,
            task_type="GPU" if torch.cuda.is_available() else "CPU"
        )

    rs = RandomizedSearchCV(
        estimator=model, # type: ignore
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=5,
        verbose=3,
        n_jobs=-1 if hyperparams["classifier"] != "catboost" else 1,
    )
    rs.fit(cv_vector_dataset, cv_labels)

    mlflow.log_metric("CV_roc_auc", rs.best_score_)
    mlflow.log_params({f"{hyperparams["classifier"]}_" + k: v for k, v in rs.best_params_.items()})

    return rs.best_estimator_
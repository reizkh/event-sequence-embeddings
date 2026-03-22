from dataset import ClientTransactionsDataset, random_slices_collate_fn, create_vector_dataset
from encoder import LSTMEncoder
from loss import contrastive_loss_euclidean

import torch
from torch import nn
from typing import Dict, Any, List
import numpy as np
import mlflow
import mlflow.artifacts
from tqdm.auto import trange, tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import lightgbm as lgb
from scipy.stats import uniform, randint


def train_encoder(
    train_dataset: ClientTransactionsDataset,
    val_dataset: ClientTransactionsDataset,
    vocab_size: int,
    hyperparams: Dict[str, Any],
    mlflow_run: Any,
    checkpoint_path: str = "model_checkpoint.pth"
) -> LSTMEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = LSTMEncoder(
        vocab_size=vocab_size,
        embedding_size=hyperparams["category_embedding_size"], 
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
            loss = contrastive_loss_euclidean(ids, embeddings)
            loss.backward()
            
            loss_value = loss.item()
            total_train_loss += loss_value
            
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
                loss = contrastive_loss_euclidean(ids, embeddings)
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
        vocab_size=hyperparams["vocab_size"], 
        embedding_size=hyperparams["category_embedding_size"], 
        hidden_size=hyperparams["embedding_size"]
    ).to(device)
    best_encoder.load_state_dict(torch.load(path, map_location=device))
    os.remove(path)
        
    return best_encoder

def run_classifier_cv(
    cv_vector_dataset: np.ndarray,
    cv_labels: List,
    hyperparams: Dict[str, Any],

) -> BaseEstimator:
    if hyperparams["classifier"] == "logistic_regression":
        model = LogisticRegression()
        param_distributions = { "C": np.logspace(-3, 2, 10) }
    elif hyperparams["classifier"] == "lightgbm":
        model = lgb.LGBMClassifier(
            objective='binary',
            n_jobs=-1,
            verbose=-1
        )

        param_distributions = {
            'n_estimators': randint(10, 200),
            'max_depth': randint(3, 15),
            'learning_rate': np.logspace(-3, 0),
            'num_leaves': randint(10, 100),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_samples': randint(1, 100),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
    
    rs = RandomizedSearchCV(
        estimator=model, # type: ignore
        param_distributions=param_distributions,
        n_iter=30,
        scoring='roc_auc',
        cv=5,
        verbose=3,
        n_jobs=-1,
    )
    rs.fit(cv_vector_dataset, cv_labels)

    mlflow.log_metric("CV_roc_auc", rs.best_score_)
    mlflow.log_params({f"{hyperparams["classifier"]}_" + k: v for k, v in rs.best_params_.items()})

    return rs.best_estimator_
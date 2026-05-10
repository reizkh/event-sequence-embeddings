from dataset import ClientTransactionsDataset, random_slices_collate_fn
from encoder import LSTMEncoder
from loss import contrastive_loss_euclidean, softmax_loss
from club import CLUB

import torch
from torch import nn
from typing import Dict, Any, List
import numpy as np
import mlflow
import mlflow.artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
from tqdm.auto import trange, tqdm
import os
from catboost import CatBoostClassifier, CatBoostRegressor


# class LGBMWithEarlyStopping(lgb.LGBMClassifier):
#     def __init__(self, val_split=0.2, early_stopping_rounds=50, **kwargs):
#         super().__init__(**kwargs)
#         self.val_split = val_split
#         self.early_stopping_rounds = early_stopping_rounds

#     def fit(self, X, y, **fit_params):
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=self.val_split, random_state=42, stratify=y if hasattr(y, 'dtype') else None
#         )
#         fit_params["eval_set"] = [(X_val, y_val)]
#         fit_params["eval_metric"] = "auc"

#         return super().fit(X_train, y_train, **fit_params)


def calculate_snr(log_q: torch.Tensor) -> float:
    off_diag = log_q[~torch.eye(log_q.size(0), dtype=torch.bool, device=log_q.device)]
    snr = (torch.diag(log_q) - off_diag.mean()).mean().item() / (off_diag.std().item() + 1e-8)
    return snr


def train_encoder(
    train_dataset: ClientTransactionsDataset,
    val_dataset: ClientTransactionsDataset,
    vocab_sizes: List[int],
    hyperparams: Dict[str, Any],
    mlflow_run: Any,
    checkpoint_path: str = "model_checkpoint.pth"
) -> LSTMEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = LSTMEncoder(
        cat_vocab_sizes=vocab_sizes,
        hidden_size=hyperparams["embedding_size"],
        sep_tokens=hyperparams["add_sep"],
        mask_pr=hyperparams["mask_pr"],
        club_pr=hyperparams["club_pr"]
    ).to(device)

    club = CLUB(
        emb_dim=hyperparams["embedding_size"]
    ).to(device)

    if hyperparams["optimizer"] == "SGD":
        opt_enc = torch.optim.SGD(
            encoder.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )
        opt_club = torch.optim.SGD(
            club.parameters(),
            lr=hyperparams["learning_rate"] * hyperparams["club_lr_ratio"],
        )
    elif hyperparams["optimizer"] == "Adam":
        opt_enc = torch.optim.Adam(
            encoder.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )
        opt_club = torch.optim.Adam(
            club.parameters(),
            lr=hyperparams["learning_rate"] * hyperparams["club_lr_ratio"],
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
        train_metrics = {
            "epoch_loss": 0.0,
            "club_pos": 0.0,
            "club_neg": 0.0,
            "mi_bound": 0.0,
            "signal_to_noise_ratio": 0.0,
            "club_grad_norm": 0.0
        }
        
        for ids, transactions, lengths in tqdm(train_loader, leave=False, desc="Training"):
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                transactions, lengths=lengths, batch_first=True, enforce_sorted=False
            ).to(device)
            embeddings = encoder.forward(packed_inputs)

            log_likelihood = club(embeddings["club_z1"].detach(), embeddings["club_z2"].detach())
            opt_club.zero_grad()
            mle_loss = -log_likelihood.diag().mean()
            mle_loss.backward()
            opt_club.step()

            log_likelihood = club(embeddings["club_z1"], embeddings["club_z2"])
            pos_term = log_likelihood.diag().mean()
            neg_term = log_likelihood.mean()
            mi_bound = pos_term - neg_term

            opt_enc.zero_grad()
            loss = (
                contrastive_loss_euclidean(ids, embeddings["coles_vectors"], margin=hyperparams["margin"]) +
                hyperparams["cmlm_lambda"] * softmax_loss(embeddings["cmlm_queries"], embeddings["cmlm_targets"]) + 
                hyperparams["mi_bound_lambda"] * mi_bound
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt_enc.step()
            
            train_metrics["epoch_loss"] += loss.item()
            train_metrics["club_pos"] += pos_term.item()
            train_metrics["club_neg"] += neg_term.item()
            train_metrics["mi_bound"] += mi_bound.item()
            train_metrics["club_grad_norm"] += sum(p.grad.norm() for p in club.parameters() if p is not None) # type: ignore
            train_metrics["signal_to_noise_ratio"] += calculate_snr(log_likelihood)
        mlflow.log_metrics({"avg_train_"+key: val / len(train_loader) for key, val in train_metrics.items()}, step=epoch)

        # --- Validation Phase ---
        encoder.eval()
        val_metrics = {
            "epoch_loss": 0.0,
            "club_pos": 0.0,
            "club_neg": 0.0,
            "mi_bound": 0.0,
            "signal_to_noise_ratio": 0.0,
        }        
        with torch.no_grad():
            for ids, transactions, lengths in tqdm(val_loader, leave=False, desc="Validation"):
                packed_inputs = nn.utils.rnn.pack_padded_sequence(
                    transactions, lengths=lengths, batch_first=True, enforce_sorted=False
                ).to(device)
                embeddings = encoder(packed_inputs)

                log_likelihood = club(embeddings["club_z1"], embeddings["club_z2"])
                pos_term = log_likelihood.diag().mean()
                neg_term = log_likelihood.mean()
                mi_bound = pos_term - neg_term

                loss = (
                    contrastive_loss_euclidean(ids, embeddings["coles_vectors"], margin=hyperparams["margin"]) +
                    hyperparams["cmlm_lambda"] * softmax_loss(embeddings["cmlm_queries"], embeddings["cmlm_targets"]) + 
                    hyperparams["mi_bound_lambda"] * mi_bound
                )
                val_metrics["epoch_loss"] += loss.item()
                val_metrics["club_pos"] += pos_term.item()
                val_metrics["club_neg"] += neg_term.item()
                val_metrics["mi_bound"] += mi_bound.item()
                val_metrics["signal_to_noise_ratio"] += calculate_snr(log_likelihood)

        mlflow.log_metrics({"avg_val_"+key: val / len(val_loader) for key, val in val_metrics.items()}, step=epoch)

        # --- Checkpointing ---
        torch.save(encoder.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path=f"models/epoch_{epoch}")
        
        if val_metrics["epoch_loss"]/len(val_loader) < best_loss:
            best_loss = val_metrics["epoch_loss"]/len(val_loader)
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
        hidden_size=hyperparams["embedding_size"]
    ).to(device)
    best_encoder.load_state_dict(torch.load(path, map_location=device))
    os.remove(path)
        
    return best_encoder

def train_downstream_models(
    local_vector_dataset: np.ndarray,
    global_vector_dataset: np.ndarray,
    local_labels: np.ndarray,
    global_labels: np.ndarray,
):
    churn_model = CatBoostClassifier(
        verbose=0,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    churn_model.fit(global_vector_dataset, global_labels)

    amount_model = CatBoostRegressor(
        verbose=0,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    amount_model.fit(local_vector_dataset, local_labels[:, 0])

    mcc_model = LogisticRegression()
    mcc_model.fit(local_vector_dataset, local_labels[:, 1])

    return churn_model, amount_model, mcc_model
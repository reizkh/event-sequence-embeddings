from encoder import LSTMEncoder
from dataset import ClientTransactionsDataset, random_slices_collate_fn, create_vector_dataset
from loss import contrastive_loss_euclidean

import mlflow
import mlflow.artifacts
from datasets import load_dataset
from torch import nn
import torch
from tqdm import trange, tqdm
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform


# 1. Load and split dataset
ds1 = load_dataset("pytorch-lifestream/rosbank-churn", "train") # labeled part -> enc_train_B/val/test = CV/test
labeled_df: pd.DataFrame = ds1["train"].to_pandas().drop(["target_sum"], axis=1) # type: ignore

ds2 = load_dataset("pytorch-lifestream/rosbank-churn", "test") # unlabeled part -> enc_train_A
unlabeled_df: pd.DataFrame = ds2["train"].to_pandas() # type: ignore
unlabeled_df["target_flag"] = None

full_df = pd.concat([labeled_df, unlabeled_df])

full_df["cl_id"] = pd.factorize(full_df.cl_id)[0]

labeled_clients = full_df.loc[~full_df["target_flag"].isna(), "cl_id"].unique()
enc_train_clients_A = full_df.loc[full_df["target_flag"].isna(), "cl_id"].unique()
crossval_clients, test_clients = train_test_split(
    labeled_clients,
    test_size=0.15,
    random_state=0
)
enc_train_clients_B, val_clients = train_test_split(
    crossval_clients,
    test_size=0.7*0.15,
    random_state=0
)
enc_train_clients = np.concat([enc_train_clients_A, enc_train_clients_B])

most_frequent_mcc = (full_df.loc[full_df["cl_id"].isin(enc_train_clients), "MCC"].value_counts() /
                     full_df["cl_id"].isin(enc_train_clients).sum())
most_frequent_mcc = most_frequent_mcc.index[most_frequent_mcc.cumsum() < 0.9]
vocab_size = len(most_frequent_mcc) + 1

full_df["MCC"] = full_df["MCC"].where(full_df["MCC"].isin(most_frequent_mcc), -1)

enc_train_df = full_df.loc[full_df["cl_id"].isin(enc_train_clients)]
enc_val_df = full_df.loc[full_df["cl_id"].isin(val_clients)]
crossval_df = full_df.loc[full_df["cl_id"].isin(crossval_clients)]
test_df = full_df.loc[full_df["cl_id"].isin(test_clients)]

enc_train_dataset = ClientTransactionsDataset(enc_train_df)
enc_val_dataset = ClientTransactionsDataset(enc_val_df)
boosting_cv_dataset = ClientTransactionsDataset(crossval_df)
test_dataset = ClientTransactionsDataset(test_df)


# 2. Set hyperparameters
embedding_size = 128
category_embedding_size = 128
num_epochs = 30
margin = 0.5
learning_rate = 0.001
n_samples_in_batch = 64
subseq_min = 15
subseq_max = 150
k = 5

hyperparams = {
    "embedding_size": embedding_size,
    "category_embedding_size": category_embedding_size,
    "num_epochs": num_epochs,
    "margin": margin,
    "learning_rate": learning_rate,
    "n_samples_in_batch": n_samples_in_batch,
    "subseq_min": subseq_min,
    "subseq_max": subseq_max,
    "k": k,
    "vocab_size": vocab_size
}


# 3. Initialize model
encoder = LSTMEncoder(
    vocab_size=vocab_size,
    embedding_size=category_embedding_size, 
    hidden_size=embedding_size
)


# 4. Train encoder
checkpoint_path = "model_checkpoint.pth"
best_loss = float('inf')

optimizer = torch.optim.SGD(
    encoder.parameters(), 
    lr=learning_rate
)

enc_train_dataloader = torch.utils.data.DataLoader(
    enc_train_dataset,
    batch_size=n_samples_in_batch,
    shuffle=True,
    collate_fn=lambda batch: random_slices_collate_fn(batch, subseq_min, subseq_max, k),
    drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    enc_val_dataset,
    batch_size=n_samples_in_batch,
    collate_fn=lambda batch: random_slices_collate_fn(batch, subseq_min, subseq_max, k),
    drop_last=True
)

with mlflow.start_run() as run:
    mlflow.log_params(hyperparams)
    
    global_step = 0
    
    for epoch in trange(num_epochs):
        # Train epoch
        total_loss = 0.0
        encoder.train()
        pbar = tqdm(enc_train_dataloader, leave=False, desc="Training")
        for ids, transactions, lengths in pbar:
            encoder.zero_grad()

            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                transactions,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False
            )
            embeddings = encoder.forward(packed_inputs)

            loss = contrastive_loss_euclidean(ids, embeddings)
            loss.backward()
            
            loss_value = loss.item()
            total_loss += loss_value
            
            mlflow.log_metric("batch_loss", loss_value, step=global_step)
            global_step += 1
            
            optimizer.step()
        
        avg_train_epoch_loss = total_loss / len(enc_train_dataloader)
        mlflow.log_metric("avg_train_epoch_loss", avg_train_epoch_loss, step=epoch)
       
        # Validate epoch
        total_loss = 0.0
        encoder.eval()
        pbar = tqdm(val_dataloader, leave=False, desc="Validating")
        for ids, transactions, lengths in pbar:
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                transactions,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False
            )
            embeddings = encoder.forward(packed_inputs)

            loss = contrastive_loss_euclidean(ids, embeddings)
            loss_value = loss.item()
            total_loss += loss_value

        avg_val_epoch_loss = total_loss / len(val_dataloader)
        mlflow.log_metric("avg_val_epoch_loss", avg_val_epoch_loss, step=epoch)

        # Save model
        torch.save(encoder.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path=f"models/epoch_{epoch}")
        
        if avg_val_epoch_loss < best_loss:
            best_loss = avg_val_epoch_loss
            mlflow.log_metric("best_loss", best_loss, step=epoch)
            mlflow.log_artifact(checkpoint_path, artifact_path="models/best_model")

# 5. Train gradient boosting on embeddings

    path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="models/best_model/model_checkpoint.pth")
    best_encoder = LSTMEncoder(
        vocab_size=hyperparams["vocab_size"], 
        embedding_size=category_embedding_size, 
        hidden_size=embedding_size
    )
    best_encoder.load_state_dict(torch.load(path))

    cv_vector_dataset, cv_labels = create_vector_dataset(best_encoder, boosting_cv_dataset, embedding_size)
    test_vector_dataset, test_labels = create_vector_dataset(best_encoder, test_dataset, embedding_size)

    scaler = StandardScaler()
    cv_vector_dataset = scaler.fit_transform(cv_vector_dataset)
    test_vector_dataset = scaler.transform(test_vector_dataset)
    
    # model = lgb.LGBMClassifier(
    #     objective='binary',
    #     n_jobs=-1,
    #     verbose=-1
    # )

    # param_distributions = {
    #     'n_estimators': randint(10, 200),
    #     'max_depth': randint(3, 15),
    #     'learning_rate': uniform(0.01, 0.3),
    #     'num_leaves': randint(10, 100),
    #     'subsample': uniform(0.6, 0.4),
    #     'colsample_bytree': uniform(0.6, 0.4),
    #     'min_child_samples': randint(1, 100),
    #     'reg_alpha': uniform(0, 1),
    #     'reg_lambda': uniform(0, 1)
    # }

    model = LogisticRegression()
    param_distributions = {
        "C": np.logspace(-3, 2, 15)
    }

    rs = GridSearchCV(
        estimator=model,
        param_grid=param_distributions,
        scoring='roc_auc',
        cv=5,
        verbose=3,
        n_jobs=-1,
    )

    rs.fit(cv_vector_dataset, cv_labels)

    mlflow.log_params({"logreg_" + k: v for k, v in rs.best_params_.items()})

# 6. Evaluate model on test set

    final_clf = LogisticRegression(**rs.best_params_)
    y_pred = final_clf.fit(cv_vector_dataset, cv_labels).predict_proba(test_vector_dataset)

    mlflow.log_metrics({
        "auc": float(roc_auc_score(test_labels, y_pred[:,1]))
    })

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
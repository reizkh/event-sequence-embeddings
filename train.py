from encoder import LSTMEncoder
from dataset import ClientTransactionsDataset, random_slices_collate_fn
from loss import contrastive_loss_euclidean

import mlflow
from datasets import load_dataset
from torch import nn
import torch
from tqdm import trange, tqdm
import os

# 1. Load dataset
ds = load_dataset("pytorch-lifestream/rosbank-churn", "train")
df = ds['train'].to_pandas()
dataset = ClientTransactionsDataset(df) # type: ignore


# 2. Set hyperparameters
embedding_size = 512
category_embedding_size = 64
num_epochs = 25
margin = 0.5
learning_rate = 0.004
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
    "k": k
}

# 3. Initialize model
encoder = LSTMEncoder(
    dataset.MCC_vocab_size, 
    embedding_size=category_embedding_size, 
    hidden_size=embedding_size
)


# 4. Training
checkpoint_path = "model_checkpoint.pth"
best_loss = float('inf')

optimizer = torch.optim.SGD(
    encoder.parameters(), 
    lr=learning_rate
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=n_samples_in_batch,
    shuffle=True,
    collate_fn=lambda batch: random_slices_collate_fn(batch, subseq_min, subseq_max, k),
    drop_last=True
)

with mlflow.start_run():
    # Логирование гиперпараметров
    mlflow.log_params(hyperparams)
    
    global_step = 0
    
    for epoch in (epoch_bar := trange(num_epochs)):
        total_loss = 0.0
        
        # Train epoch
        for batch_idx, (ids, transactions, lengths) in enumerate((pbar := tqdm(dataloader, leave=False))):
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
            
            pbar.set_postfix(loss=loss_value)

            optimizer.step()
        
        avg_epoch_loss = total_loss / len(dataloader)
        mlflow.log_metric("avg_epoch_loss", avg_epoch_loss, step=epoch)
        epoch_bar.set_postfix(avg_epoch_loss=avg_epoch_loss)

        torch.save(encoder.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path=f"models/epoch_{epoch}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            mlflow.log_metric("best_loss", best_loss, step=epoch)
            mlflow.log_artifact(checkpoint_path, artifact_path="models/best_model")

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
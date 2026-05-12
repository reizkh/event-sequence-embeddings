from encoder import LSTMEncoder


import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union, Dict, Any, Optional
import random
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class ClientTransactionsDataset(Dataset):
    """
    Класс датасета для загрузки транзакций клиентов.

    Данные группируются по полю 'cl_id'. Категориальные признаки преобразуются 
    в индексы на основе предоставленных или автоматически сгенерированных словарей.

    Атрибуты:
        - cl_ids (List): Список уникальных идентификаторов клиентов.
        - transactions (List[torch.Tensor]): Список тензоров признаков для каждого клиента.
        - labels (List[int]): Список целевых меток (при наличии).
        - cat_vocabularies (Dict[str, Dict]): Словари соответствия значений категориальных 
          признаков их индексам для каждой колонки.
        - cat_cols (List[str]): Список имен колонок с категориальными признаками.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        cat_cols: List[str],
        sep_events: bool = True
    ):
        """
        Инициализация датасета и предварительная обработка данных.

        :param pd.DataFrame df: DataFrame, содержащий колонки 'cl_id', 'amount', 
                                а также указанные категориальные колонки.
        :param List[str] cat_cols: Список имен колонок, содержащих категориальные признаки.
        """
        super().__init__()
        
        # Валидация входных данных
        required_columns = {'cl_id', 'amount'}.union(set(cat_cols))

        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"DataFrame должен содержать колонки: {missing}")
        
        self.label = "target_flag" in df.columns
        self.cat_cols = cat_cols
        
        self.cl_ids: List[Union[int, str]] = []
        self.transactions: List[torch.Tensor] = []
        self.labels: List[int] = []
        
        # Инициализация словарей для категориальных признаков
        self.cat_vocabularies: Dict[str, Dict[Union[int, str], int]] = {}
        self.cat_vocab_sizes: Dict[str, int] = {}
        
        for col in cat_cols:
            # Автоматическое построение словаря на основе данных
            # Сортировка обеспечивает детерминированность порядка классов
            unique_values = sorted(df[col].unique())
            self.cat_vocabularies[col] = {val: idx for idx, val in enumerate(unique_values)}
            self.cat_vocab_sizes[col] = len(unique_values)
        
        # Группировка данных по cl_id
        grouped = df.groupby('cl_id', sort=True)
        
        for cl_id, group in grouped:
            self.cl_ids.append(cl_id) # type: ignore

            # Обработка числовых признаков (amount)
            # Приведение к тензору и добавление размерности [N, 1]
            log_amounts = torch.log(torch.tensor(group['amount'].values, dtype=torch.float32).unsqueeze(1))
            
            # Обработка категориальных признаков
            cat_tensors = []
            for col in cat_cols:
                # Маппинг значений колонки в индексы классов
                indices = [self.cat_vocabularies[col][m] for m in group[col]]
                cat_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(1)
                cat_tensors.append(cat_tensor)
            
            if sep_events:
                cat_tensors.append(torch.tensor(group["is_sep"].to_numpy()).unsqueeze(1))
                        
            # Конкатенация признаков: [amount, cat_col_1, cat_col_2, ...]
            # Итоговая размерность последовательности: [N, 1 + len(cat_cols)]
            if cat_tensors:
                features = torch.cat([log_amounts] + cat_tensors, dim=1)
            else:
                features = log_amounts
            
            self.transactions.append(features)
            
            if self.label:
                self.labels.append(group["target_flag"].iloc[0])

    def __len__(self) -> int:
        """
        Возвращает количество уникальных клиентов в датасете.

        :returns: Количество элементов (уникальных ``cl_id``).
        :rtype: int
        """
        return len(self.cl_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Union[int, str], torch.Tensor, Optional[int]]:
        """
        Возвращает данные для клиента по индексу.

        :param int idx: Индекс клиента в диапазоне ``[0, len(self) - 1]``.
            
        :rtype: Tuple[Union[int, str], torch.Tensor, Optional[int]]
        :returns:
        Кортеж ``(cl_id, features, label)`` содержащий:

        * ``cl_id`` — идентификатор клиента.
        * ``features`` — тензор размера ``[N_transactions, 1 + len(cat_cols)]``.
        * ``label`` — целевая метка (или None, если отсутствует).
        """
        if idx >= len(self):
            raise IndexError(f"Индекс {idx} выходит за границы датасета (размер: {len(self)})")
            
        cl_id = self.cl_ids[idx]
        features = self.transactions[idx]

        if self.label:
            return cl_id, features, self.labels[idx]
        return cl_id, features, None
    
def random_slices_collate_fn(
    batch: List[Tuple[Any, torch.Tensor, Any]], 
    m: int, 
    M: int, 
    k: int,
    pad_value: float = 0.0
) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
    """
    Collate function, реализующая стратегию случайной выборки подпоследовательностей.
    
    :param List[Tuple[Any, torch.Tensor]] batch:
        Список кортежей (cl_id, transactions), где transactions имеет форму [T, feature_dim]
    :param int m:
        Минимальная длина подпоследовательности
    :param int M:
        Максимальная длина подпоследовательности
    :param int k:
        Количество сэмплов для генерации из каждой последовательности
    :param float pad_value:
        Значение для padding
        
    :rtype: Tuple[List[Any], torch.Tensor, torch.Tensor]
    :returns:
    Кортеж ``(cl_ids, padded_subsequences, lengths)``, где:

    * ``cl_ids`` — список идентификаторов клиентов повторяется k раз для каждого клиента)
    * ``padded_subsequences`` — тензор формы ``[batch_size * k, max_len, feature_dim]``.
    * ``lengths`` — тензор формы ``[batch_size * k]`` с фактическими длинами подпоследовательностей.
    """
    cl_ids = []
    subsequences = []
    lengths = []
    
    for client_id, transactions, _ in batch:
        T = transactions.shape[0]  # Длина исходной последовательности
        
        # Генерация k случайных подпоследовательностей
        for _ in range(k):
            # Генерация случайной длины T_i равномерно из [m, M]
            # Ограничиваем сверху длиной последовательности T
            T_i = min(random.randint(m, M), T)
            
            # Генерация случайной начальной позиции s из [0, T - T_i)
            if T - T_i > 0:
                s = random.randint(0, T - T_i)
            else:
                s = 0
            
            # Извлечение подпоследовательности Ŝ_i := {z_{s+j}}_{j=0}^{T_i-1}
            subseq = transactions[s:s + T_i]
            
            cl_ids.append(client_id)
            subsequences.append(subseq)
            lengths.append(T_i)
    
    # Padding до максимальной длины
    max_len = max(lengths) if lengths else 0
    
    if max_len == 0:
        # Обработка случая пустого батча
        return [], torch.empty(0), torch.empty(0)
    
    padded_subsequences = []
    for subseq in subsequences:
        current_len = subseq.shape[0]
        if current_len < max_len:
            # Добавление padding
            pad_size = max_len - current_len
            if subseq.dim() == 1:
                # Для одномерных тензоров
                padded = F.pad(subseq, (0, pad_size), value=pad_value)
            else:
                # Для многомерных тензоров [seq_len, feature_dim]
                padded = F.pad(subseq, (0, 0, 0, pad_size), value=pad_value)
        else:
            padded = subseq
        padded_subsequences.append(padded)
    
    # Стек всех подпоследовательностей в один батч
    padded_subsequences = torch.stack(padded_subsequences, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return cl_ids, padded_subsequences, lengths

def add_sep_events(
    df: pd.DataFrame,
    cl_id_column: str = "cl_id",
    date_column: str = "date",
    is_sep_column: str = "is_sep"
):
    df = df.copy()
    df[is_sep_column] = False
    sep_rows = {
        cl_id_column: [],
        date_column: [],
    }
        
    grouped = df.groupby(cl_id_column, sort=False)
    for cl_id, group in grouped:
        group = group.sort_values(date_column).reset_index(drop=True)
        
        begin = group[date_column].min()
        end = group[date_column].max()
        x = begin
        while x < end:
            sep_rows[cl_id_column].append(cl_id)
            sep_rows[date_column].append(x)
            x += pd.Timedelta("1d")
    
    df_separators = pd.DataFrame(sep_rows)
    df_separators[is_sep_column] = True
    df_separators["amount"] = 1.0
    for col in df.columns:
        if col not in df_separators.columns:
            df_separators[col] = "rare"
    df_result = pd.concat([df, df_separators], ignore_index=True)
    
    df_result = df_result.sort_values([cl_id_column, date_column, is_sep_column]).reset_index(drop=True)
    
    return df_result

def load_and_split_data(
    labeled_dataset_path: str,
    unlabeled_dataset_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: Optional[int] = 0,
    cat_features: List = ["MCC"],
    cat_coverage: float = 0.9,
    add_sep: bool = False
) -> Tuple[ClientTransactionsDataset, ClientTransactionsDataset, ClientTransactionsDataset, ClientTransactionsDataset, List]:
    """
    Загружает датасеты, объединяет размеченные и неразмеченные данные,
    выполняет разбиение по клиентам и фильтрацию редких MCC-кодов.
    """
    ds_labeled = load_dataset(labeled_dataset_path, "train")
    labeled_df: pd.DataFrame = ds_labeled["train"].to_pandas().drop(["target_sum"], axis=1)  # type: ignore

    ds_unlabeled = load_dataset(unlabeled_dataset_path, "test")
    unlabeled_df: pd.DataFrame = ds_unlabeled["train"].to_pandas()  # type: ignore
    unlabeled_df["target_flag"] = None

    full_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    full_df["cl_id"] = pd.factorize(full_df.cl_id)[0]

    labeled_clients = full_df.loc[~full_df["target_flag"].isna(), "cl_id"].unique()
    enc_train_clients_A = full_df.loc[full_df["target_flag"].isna(), "cl_id"].unique()
    
    crossval_clients, test_clients = train_test_split(
        labeled_clients, test_size=test_size, random_state=random_state
    )
    enc_train_clients_B, val_clients = train_test_split(
        crossval_clients, test_size=val_size / (1 - test_size), random_state=random_state
    )
    
    enc_train_clients = np.concatenate([enc_train_clients_A, enc_train_clients_B])

    mask_train = full_df["cl_id"].isin(enc_train_clients)
    vocab_sizes = []
    for cat_feature in cat_features:
        feature_value_counts = full_df.loc[mask_train, cat_feature].value_counts(normalize=True)
        most_frequent_values = feature_value_counts[feature_value_counts.cumsum() < cat_coverage].index
        vocab_sizes.append(len(most_frequent_values) + 1)
        
        full_df[cat_feature] = full_df[cat_feature].astype(str).where(full_df[cat_feature].isin(most_frequent_values), "rare")

    full_df['TRDATETIME'] = pd.to_datetime(full_df['TRDATETIME'], format="%d%b%y:%X")
    full_df["date"] = full_df["TRDATETIME"].dt.date

    if add_sep:
        full_df = add_sep_events(full_df)

    enc_train_df = full_df[full_df["cl_id"].isin(enc_train_clients)]
    enc_val_df = full_df[full_df["cl_id"].isin(val_clients)]
    crossval_df = full_df[full_df["cl_id"].isin(crossval_clients)]
    test_df = full_df[full_df["cl_id"].isin(test_clients)]

    return (
        ClientTransactionsDataset(enc_train_df, cat_features, sep_events=add_sep),
        ClientTransactionsDataset(enc_val_df, cat_features, sep_events=add_sep),
        ClientTransactionsDataset(crossval_df, cat_features, sep_events=add_sep),
        ClientTransactionsDataset(test_df, cat_features, sep_events=add_sep),
        vocab_sizes
    )

@torch.no_grad()
def create_global_dataset(
    model: LSTMEncoder,
    dataset: ClientTransactionsDataset, 
    device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    vector_dataset = []
    labels = np.array(dataset.labels, dtype=np.long)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=64,
        collate_fn=lambda batch: random_slices_collate_fn(batch, 10000, 10000, 1),
        drop_last=False
    )
    pbar = tqdm(dataloader, desc="Creating vector dataset")
    for ids, transactions, lengths in pbar:
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            transactions,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        ).to(device)
        vector_dataset.append(model.global_embed(packed_inputs).detach().cpu().numpy())
    vector_dataset = np.concat(vector_dataset)
    return vector_dataset, labels

@torch.no_grad()
def create_local_dataset(
    enc: LSTMEncoder,
    data: ClientTransactionsDataset,
    device,
    window_len: int = 32,
    window_stride: int = 32,
    sep_events: bool = False,
    global_embed: bool = False
):
    enc.eval()
    x = []
    targets = []
    dl = DataLoader(
        data,
        shuffle=False
    )
    for _, seq, _ in tqdm(dl): # type: ignore
        seq = seq[0].to(device)

        if sep_events:
            normal_mask = seq[:, -1] == 0
        else:
            normal_mask = torch.ones_like(seq[:, -1])
        real_events = torch.where(normal_mask)[0]

        for idx in range(0, real_events.shape[0] - window_len, window_stride):
            i = real_events[idx]
            j = real_events[idx + window_len]

            target_log_amount = seq[j, 0]
            target_mcc = seq[j, 1].long()
            if global_embed:
                emb = enc.global_embed(seq[i:j])
            else:
                emb = enc.local_embed(seq[i:j])
            x.append(emb)
            targets.append([target_log_amount, target_mcc])

    x = torch.stack(x).detach().cpu().numpy()
    targets = torch.tensor(targets).detach().cpu().numpy()
    return x, targets
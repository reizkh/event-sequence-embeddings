import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Dict, Any
import random

class ClientTransactionsDataset(Dataset):
    """
    Класс датасета для загрузки транзакций клиентов.
    
    Данные группируются по полю 'cl_id'. Поле 'MCC' преобразуется в индекс по словарю.
    
    Атрибуты:
        - cl_ids (List): Список уникальных идентификаторов клиентов.
        - transactions (List[torch.Tensor]): Список тензоров признаков для каждого клиента.
        - MCC_vocab (Dict): Словарь соответствия значений MCC их индексам.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация датасета и предварительная обработка данных.
        
        :param pd.DataFrame df: DataFrame, содержащий колонки 'cl_id', 'amount', 'MCC'.
        """
        super().__init__()
        
        # Валидация входных данных
        required_columns = {'cl_id', 'amount', 'MCC'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required_columns}")
        
        self.cl_ids: List[int] = []
        self.transactions: List[torch.Tensor] = []
        
        # Сортировка обеспечивает детерминированность порядка классов
        unique_MCCs = sorted(df['MCC'].unique())
        self.MCC_vocab: Dict[Union[int, str], int] = {val: idx for idx, val in enumerate(unique_MCCs)}
        self.MCC_vocab_size: int = len(unique_MCCs)
        
        # Группировка данных по cl_id
        grouped = df.groupby('cl_id', sort=True)
        
        for cl_id, group in grouped:
            self.cl_ids.append(cl_id)
            
            # Обработка поля amount
            # Приведение к тензору и добавление размерности [N, 1]
            amounts = torch.tensor(group['amount'].values, dtype=torch.float32).unsqueeze(1)
            
            # Обработка поля MCC
            # Маппинг значений MCC в индексы классов
            MCC_indices = [self.MCC_vocab[m] for m in group['MCC']]
            MCC_tensor = torch.tensor(MCC_indices, dtype=torch.long).unsqueeze(1)
                        
            # Конкатенация признаков: [amount, MCC_one_hot...]
            # Итоговая размерность транзакции: [1 + MCC_vocab_size]
            features = torch.cat([amounts, MCC_tensor], dim=1)
            
            self.transactions.append(features)
            
    def __len__(self) -> int:
        """
        Возвращает количество уникальных клиентов в датасете.
        
        :returns: Количество элементов (уникальных ``cl_id``).
        :rtype: int
        """
        return len(self.cl_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Union[int, str], torch.Tensor]:
        """
        Возвращает данные для клиента по индексу.
        
        :param int idx: Индекс клиента в диапазоне ``[0, len(self) - 1]``.
            
        :rtype: Tuple[Union[int, str], torch.Tensor]
        :returns:
        Кортеж ``(cl_id, features)`` содержащий:

        * ``cl_id`` — (идентификатор клиента).
        * ``features`` — (тензор размера ``[N_transactions, 2]``).
        """
        if idx >= len(self):
            raise IndexError(f"Индекс {idx} выходит за границы датасета (размер: {len(self)})")
            
        cl_id = self.cl_ids[idx]
        features = self.transactions[idx]
        
        return cl_id, features
    
def random_slices_collate_fn(
    batch: List[Tuple[Any, torch.Tensor]], 
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
    
    for client_id, transactions in batch:
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
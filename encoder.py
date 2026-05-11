import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import List, Dict, Any

class LSTMEncoder(nn.Module):
    """
    Модуль кодирования последовательностей на основе архитектуры LSTM.
    
    Данный модуль принимает на вход упакованную последовательность (PackedSequence), 
    содержащую смешанные числовые и категориальные признаки, и возвращает 
    векторное представление последовательности на основе финального скрытого состояния.
    
    Поддерживается динамическая конфигурация входных признаков:
    - Произвольное количество числовых признаков (подвергаются нормализации).
    - Произвольное количество категориальных признаков (подвергаются эмбеддингу).
    
    Атрибуты:
        - numerical_bn (nn.Module): Слой нормализации для числовых признаков.
        - categorical_embeddings (nn.ModuleList): Список слоев эмбеддинга для категориальных признаков.
        - lstm (nn.LSTM): Основной слой долгой краткосрочной памяти.
    """
    
    def __init__(
        self, 
        cat_vocab_sizes: List[int],
        hidden_size: int, 
        batch_first: bool = True,
        num_numerical_features: int = 1,
        mask_pr: float = 0.02,
        club_pr: float = 0.02,
        sep_tokens: bool = False
    ):
        """
        Инициализация компонентов модели.

        :param int num_numerical_features: Количество числовых признаков во входных данных 
                                           (например, 'amount').
        :param List[int] cat_vocab_sizes: Список размеров словарей для каждого категориального признака.
        :param List[int] cat_embedding_dims: Список размерностей эмбеддинга для каждого 
                                             категориального признака.
        :param int hidden_size: Размерность скрытого состояния LSTM.
        :param bool batch_first: Флаг формата входных данных (рекомендуется True).
        """
        super(LSTMEncoder, self).__init__()
        
        self.num_numerical_features = num_numerical_features
        self.cat_vocab_sizes = cat_vocab_sizes
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.mask_pr = mask_pr
        self.club_pr = club_pr
        self.sep_tokens = sep_tokens

        # Инициализация нормализации для числовых признаков
        if num_numerical_features > 0:
            self.numerical_bn = nn.BatchNorm1d(num_numerical_features)
        else:
            self.numerical_bn = nn.Identity()

        intermediate_dim = num_numerical_features + sum(cat_vocab_sizes)
        self.linear = nn.Linear(in_features=intermediate_dim, out_features=hidden_size)

        self.global_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.local_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.sep_vector = nn.Parameter(torch.empty([self.hidden_size]))
        self.mask_vector = nn.Parameter(torch.empty([self.hidden_size]))

        torch.nn.init.normal_(self.sep_vector, std=1/self.hidden_size**0.5)
        torch.nn.init.normal_(self.mask_vector, std=1/self.hidden_size**0.5)
        
        # Инициализация LSTM слоя
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=False
        )

    def embed_events(self, data: torch.Tensor) -> torch.Tensor:
        processed_features = []
        
        # Обработка числовых признаков
        if self.num_numerical_features > 0:
            numerical_data = data[:, :self.num_numerical_features]
            numerical_normalized = self.numerical_bn(numerical_data)
            processed_features.append(numerical_normalized)
        
        # Обработка категориальных признаков
        cat_start_idx = self.num_numerical_features
        for i, num_classes in enumerate(self.cat_vocab_sizes):
            cat_indices = data[:, cat_start_idx + i].long()
            # Применение OHE
            encoded_category = F.one_hot(cat_indices, num_classes)
            processed_features.append(encoded_category)

        intermediate_embedding = torch.cat(processed_features, dim=1)
        event_embeddings = self.linear(intermediate_embedding)
        return event_embeddings

    def apply_special_tokens(self, data: torch.Tensor, sep_idx: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        data = torch.where(sep_idx.unsqueeze(-1), self.sep_vector.unsqueeze(0), data)
        data = torch.where(mask_idx.unsqueeze(-1), self.mask_vector.unsqueeze(0), data)
        return data
        
    def forward(self, packed_input: PackedSequence) -> Dict[Any, torch.Tensor]:
        data = packed_input.data

        event_embeddings = self.embed_events(data)

        sep_idx = data[:, -1].bool()
        if not self.sep_tokens:
            sep_idx = torch.zeros_like(sep_idx)
        mask_idx = torch.rand(data.shape[0], device=data.device) < self.mask_pr
        masked_event_embeddings = self.apply_special_tokens(event_embeddings, sep_idx, mask_idx)
        
        lstm_input = packed_input._replace(data=masked_event_embeddings)
        h_t, (h_n, _) = self.lstm(lstm_input)
        h_t = h_t.data
        
        coles_vectors = self.global_proj(h_n[-1])
        cmlm_queries = self.local_proj(h_t[mask_idx])
        cmlm_targets = event_embeddings[mask_idx]

        rand_idx = torch.rand(data.shape[0], device=data.device) < self.club_pr
        club_z1 = self.global_proj(h_t[rand_idx])
        club_z2 = self.local_proj(h_t[rand_idx])

        return {
            "coles_vectors": coles_vectors,
            "cmlm_queries": cmlm_queries,
            "cmlm_targets": cmlm_targets,
            "club_z1": club_z1,
            "club_z2": club_z2
        }

    def local_embed(self, data: torch.Tensor) -> torch.Tensor:
        event_embeddings = self.embed_events(data)
        event_embeddings = torch.concat([event_embeddings, self.mask_vector.unsqueeze(0)])
        _, (h_n, _) = self.lstm(event_embeddings)
        return self.local_proj(h_n[-1])
    
    def global_embed(self, packed_input: PackedSequence) -> torch.Tensor:
        data = packed_input.data
        event_embeddings = self.embed_events(data)
        _, (h_n, _) = self.lstm(packed_input._replace(data=event_embeddings))
        return self.global_proj(h_n[-1])
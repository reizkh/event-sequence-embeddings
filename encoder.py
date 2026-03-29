import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import List

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
        cat_embedding_dims: List[int],
        hidden_size: int, 
        batch_first: bool = True,
        num_numerical_features: int = 1
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
        self.cat_embedding_dims = cat_embedding_dims
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        # Валидация входных параметров
        if len(cat_vocab_sizes) != len(cat_embedding_dims):
            raise ValueError("Списки cat_vocab_sizes и cat_embedding_dims должны иметь одинаковую длину.")
        
        # Инициализация нормализации для числовых признаков
        if num_numerical_features > 0:
            self.numerical_bn = nn.BatchNorm1d(num_numerical_features)
        else:
            self.numerical_bn = nn.Identity()
        
        # Инициализация слоев эмбеддинга для категориальных признаков
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            for vocab_size, embed_dim in zip(cat_vocab_sizes, cat_embedding_dims)
        ])
        
        total_embedding_dim = sum(cat_embedding_dims)
        self.input_size = num_numerical_features + hidden_size
        
        self.linear = nn.Linear(in_features=total_embedding_dim, out_features=hidden_size)

        self.sep_vector = nn.Parameter(torch.zeros([self.input_size]))
        
        # Инициализация LSTM слоя
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=False
        )
        
    def forward(self, packed_input: PackedSequence) -> torch.Tensor:
        """
        Прямой проход через энкодер.
        
        Осуществляет разделение входных данных на числовые и категориальные части,
        применяет соответствующие преобразования (нормализация, эмбеддинг) и передает
        результат в LSTM.

        :param PackedSequence packed_input: Упакованная последовательность входных данных. 
                                            Данные должны иметь структуру [TotalSteps, TotalFeatures],
                                            где первые колонки — числовые признаки, последующие — 
                                            индексы категориальных признаков.
        
        :rtype: torch.Tensor
        :return: Финальное скрытое состояние размера ``[batch_size, hidden_size]``.
        """
        data = packed_input.data
        device = data.device
        combined_input = torch.zeros([data.shape[0], self.input_size], device=device)

        sep_idx = data[:,-1] == 1
        real_data = data[~sep_idx] # Данные с флагом is_sep обрабатываются независимо от остальных

        processed_features = []
        embedded_features = []
        
        # Обработка числовых признаков
        if self.num_numerical_features > 0:
            numerical_data = real_data[:, :self.num_numerical_features]
            # BatchNorm1d ожидает размерность (N, C), где C — количество признаков
            numerical_normalized = self.numerical_bn(numerical_data)
            processed_features.append(numerical_normalized)
        
        # Обработка категориальных признаков
        cat_start_idx = self.num_numerical_features
        for i, embedding_layer in enumerate(self.categorical_embeddings):
            # Извлечение колонки соответствующего категориального признака
            cat_indices = real_data[:, cat_start_idx + i].long()
            # Применение эмбеддинга
            cat_embedded = embedding_layer(cat_indices)
            embedded_features.append(cat_embedded)

        combined_embeddings = torch.cat(embedded_features, dim=1)
        combined_embeddings = self.linear(combined_embeddings)

        processed_features.append(combined_embeddings)
        
        # Конкатенация всех обработанных признаков
        combined_input[~sep_idx] = torch.cat(processed_features, dim=1)
        combined_input[sep_idx] = self.sep_vector
        
        # Формирование обновленной PackedSequence
        lstm_input = packed_input._replace(data=combined_input)
        
        # Прямой проход через LSTM
        # LSTM автоматически обрабатывает упакованную последовательность
        _, (h_n, _) = self.lstm(lstm_input)
        
        # Возврат скрытого состояния последнего слоя
        return h_n[-1, :, :]


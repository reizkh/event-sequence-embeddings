import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Инициализация нормализации для числовых признаков
        if num_numerical_features > 0:
            self.numerical_bn = nn.BatchNorm1d(num_numerical_features)
        else:
            self.numerical_bn = nn.Identity()

        intermediate_dim = num_numerical_features + sum(cat_vocab_sizes)
        self.linear = nn.Linear(in_features=intermediate_dim, out_features=hidden_size)
        
        # Инициализация LSTM слоя
        self.lstm = nn.LSTM(
            input_size=hidden_size,
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
        
        # Формирование обновленной PackedSequence
        lstm_input = packed_input._replace(data=event_embeddings)
        
        # Прямой проход через LSTM
        # LSTM автоматически обрабатывает упакованную последовательность
        _, (h_n, _) = self.lstm(lstm_input)
        
        # Возврат скрытого состояния последнего слоя
        return h_n[-1, :, :]


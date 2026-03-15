import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import Tuple

class LSTMEncoder(nn.Module):
    """   
    Данный модуль принимает на вход упакованную последовательность и возвращает
    финальные состояния скрытого слоя и ячейки памяти, которые могут быть использованы
    как векторное представление исходной последовательности.
    
    :param input_size: Размерность входных признаков (количество_FEATURES в одном шаге времени).
    :type input_size: int
    :param hidden_size: Размерность скрытого состояния (количество нейронов в скрытом слое).
    :type hidden_size: int
    :param batch_first: Если True, то входные и выходные тензоры имеют размерность 
                        (batch, seq, feature). Рекомендуется устанавливать True для 
                        совместимости с DataLoader.
    :type batch_first: bool, optional
    """
    
    def __init__(
        self, 
        vocab_size: int,
        embedding_size: int,
        hidden_size: int, 
        batch_first: bool = True
    ):
        super(LSTMEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first


        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size
        )        
        self.bn = nn.BatchNorm1d(1)

        # Инициализация LSTM слоя
        self.lstm = nn.LSTM(
            input_size=embedding_size+1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=False
        )
        
    def forward(self, packed_input: PackedSequence) -> torch.Tensor:
        """
        Прямой проход через энкодер.
        
        :param packed_input: Упакованная последовательность входных данных. 
                             Объект типа ``PackedSequence``, содержащий данные формы 
                             ``[batch, seq, feature]``.
        :type packed_input: PackedSequence
        
        :rtype: torch.Tensor
        :return: Финальное скрытое состояние размера ``[batch, hidden_size]``.
        """
        amounts = packed_input.data[:,0]
        MCC = packed_input.data[:,1].long()

        amounts = self.bn(amounts.unsqueeze(1))
        MCC_embeddings = self.embedding(MCC)
        input = packed_input._replace(data=torch.cat([amounts, MCC_embeddings], dim=1))

        # LSTM автоматически обрабатывает PackedSequence, игнорируя padding
        _, (h_n, c_n) = self.lstm(input)
        
        return h_n[-1, :, :]
    
    def get_encoded_vector(self, packed_input: PackedSequence) -> torch.Tensor:
        """
        Вспомогательный метод для получения единого векторного представления последовательности.
        
        Извлекает финальное скрытое состояние из последнего слоя и удаляет размерность слоя,
        возвращая тензор размерности (batch, hidden_size).
        
        :param packed_input: Упакованная последовательность входных данных.
        :type packed_input: PackedSequence
        
        :return: Тензор кодированных представлений для каждого элемента в батче.
        :rtype: torch.Tensor
        """
        h_n, _ = self.forward(packed_input)
        
        # h_n имеет размерность (num_layers, batch, hidden_size)
        # Для однослойной модели выбираем последний слой (индекс -1 или 0)
        encoded_vector = h_n[-1, :, :]
        
        return encoded_vector
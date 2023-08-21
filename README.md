# BiLSTM/GRU


## Архитектура модели
### Conv1dModule
Свёрточный слой: nn.Conv1d с n_in фильтрами на входе и n_out фильтрами на выходе. Размер ядра kernel.

### FlattenModule
Линейный слой: nn.Linear с n_in нейронами на входе и n_out нейронами на выходе.

### Conv1dModel
Последовательность свёрточных модулей Conv1dModule, ConformerBlock, слоя Flatten и полносвязных слоёв FlattenModule.

### CNNLayerNorm
Слой нормализации, специализированный для свёрточных слоёв, использующий nn.LayerNorm.

### ResidualCNN
Остаточные свёрточные слои: Два свёрточных слоя nn.Conv2d с дропаутом и нормализацией.

### BidirectionalGRU
Двунаправленный GRU: nn.GRU с размером входа rnn_dim и размером скрытого состояния hidden_size.
Нормализация: nn.LayerNorm с размером rnn_dim.

### SpeechRecognitionModel
Свёрточный слой: nn.Conv2d для извлечения иерархических признаков.
Остаточные свёрточные слои: Последовательность слоёв ResidualCNN.
Полносвязный слой: nn.Linear.
Двунаправленные GRU: Последовательность слоёв BidirectionalGRU.
Классификатор: Последовательность линейных слоёв и активации GELU.
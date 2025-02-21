from scipy.io import wavfile
from torch.utils.mobile_optimizer import optimize_for_mobile
from pytorch_metric_learning import losses
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torchaudio.functional as TAF
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from typing import List, Tuple
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import functools
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
import sys
from audiomentations import Compose, SevenBandParametricEQ, RoomSimulator, AirAbsorption, TanhDistortion, TimeStretch, \
    PitchShift, AddGaussianNoise, Gain, Shift, BandStopFilter, AddBackgroundNoise, PolarityInversion

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


torch.cuda.empty_cache()

root_dir = "e:\\dvc"
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 321
hparams = {
    "n_cnn_layers": 2,
    "n_rnn_layers": 2,
    "rnn_dim": 512,
    "n_class": 28,
    "n_feats": 80,
    "stride": 2,
    "dropout": 0.1,
}
# Hyperparameters
# sequence_length = 28
input_size = 40
hidden_size = 512
num_layers = 3
num_classes = 2


class Conv1dModule(nn.Module):
    """
    Простой кирпичик для свёрточной модели.
    n_in -- число фильтров на входе
    n_out -- число фильтров на выходе
    kernel -- размер ядра
    pooling -- размер ядра пулинга
    batchnorm -- флаг отвечающий за использование батч нормализации
    relu -- флаг отвечающий за использование нелинейности
    """

    def __init__(self, n_in, n_out, kernel, pooling, batchnorm=False, relu=True):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        modules = [nn.Conv1d(n_in, n_out, kernel, padding=pad)]
        if batchnorm:
            modules.append(nn.BatchNorm1d(n_out))
        if pooling > 1:
            modules.append(nn.MaxPool1d(pooling))
        if relu:
            modules.append(nn.ReLU())
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)


class Flatten(nn.Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)


class FlattenModule(nn.Module):
    """
    Простой кирпичик полносвязной части свёрточной модели.
    n_in -- число нейронов на входе
    n_out -- число нейроново на выходе
    batchnorm -- флаг отвечающий за использование батч нормализации
    relu -- флаг отвечающий за использование нелинейности
    """

    def __init__(self, n_in, n_out, batchnorm=False, relu=True):
        super().__init__()
        modules = [nn.Linear(n_in, n_out)]
        if batchnorm:
            modules.append(nn.BatchNorm1d(n_out))
        if relu:
            modules.append(nn.ReLU())
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)


block = ConformerBlock(
    dim=40,
    dim_head=64,
    heads=8,
    ff_mult=4,
    conv_expansion_factor=2,
    conv_kernel_size=31,
    attn_dropout=0.,
    ff_dropout=0.,
    conv_dropout=0.
)


class Conv1dModel(nn.Module):
    """
    shapes -- число филтров в свёрточных слоях. На нулевом индексе число фильтров на входе.
        Далее число фильтров после каждого свёрточного слоя
    flatten_shapes -- число нейронов после применения линейных слоёв.
    kernels -- размеры ядер свёрточных слоёв
    poolings -- параметры пулинга после свёрточных слоёв
    batchnorm -- флаг отвечающий за использовать или нет нормализацию
    """

    def __init__(
            self, shapes: List[int], flatten_shapes: List[int], kernels: List[int],
            poolings: List[int], batchnorm=False
    ):
        super().__init__()
        assert len(kernels) + 1 == len(shapes)
        assert len(poolings) == len(kernels)
        modules = []
        start_flatten_shape = MAX_LEN
        for i in range(len(kernels)):
            modules.append(Conv1dModule(
                shapes[i], shapes[i + 1], kernels[i], poolings[i],
                batchnorm=batchnorm
            ))
            start_flatten_shape //= poolings[i]
        modules.append(block)
        modules.append(Flatten())
        flatten_shapes = [start_flatten_shape * shapes[-1]] + flatten_shapes
        for i in range(len(flatten_shapes) - 1):
            modules.append(FlattenModule(
                flatten_shapes[i], flatten_shapes[i + 1], batchnorm=batchnorm, relu=i + 2 == len(flatten_shapes)
            ))
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
        self.lstm0 = nn.GRU(rnn_dim * 2, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, len(CLASSES))

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x0 = self.birnn_layers(x)
        out, _ = self.lstm0(x0)
        out = self.fc(out[:, -1, :])
        x = self.classifier(x0)
        return x, out


class DueModel(nn.Module):
    def __init__(self):
        super(DueModel, self).__init__()
        self.conv1d_clean_model = Conv1dModel([80] + [256] * 7, [1024, 256, len(CLASSES)], [5] * 7, [1, 2] * 3 + [1],
                                              batchnorm=True)
        self.dp2 = SpeechRecognitionModel(
            hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout'])
        self.ob = nn.Linear(len(CLASSES) * 2, len(CLASSES))

    #         self.ob1=nn.Linear(hparams['rnn_dim'], hparams['n_class'])
    def forward(self, x):
#        x1 = x
#        x2 = torch.cat((x1, x), dim=1)
        x21 = x.unsqueeze(1)
        x4, x5 = self.dp2(x21)
        x3 = self.conv1d_clean_model(x2)
        x6 = torch.cat((x3, x5), dim=-1)
        x7 = self.ob(x6)
        #         x7=self.ob1(x7)
        return x7, x3, x5, x1, x4


class CommandDataset(Dataset):

    def __init__(self, meta, root_dir, sample_rate, labelmap, augment=True):
        self.meta = meta
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.labelmap = labelmap
        self.augment = augment
        if self.augment:
            self.augmentations = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.1),
                PitchShift(min_semitones=-6, max_semitones=6, p=0.1),
                #                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.030, p=0.1),
                #                Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.1),
                #                BandStopFilter(min_bandwidth_fraction=0.01, max_bandwidth_fraction=0.25, p=0.1),
                #                PolarityInversion(p=0.1),
                Shift(min_fraction=-0.1, max_fraction=0.1, p=0.1),
                AirAbsorption(p=0.4),
                TanhDistortion(p=0.1),
                #                SevenBandParametricEQ(p=0.3)
            ])
        n_mfcc = 64
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc, log_mels=True,
                                       melkwargs={'n_fft': 1024, 'hop_length': 160, 'n_mels': n_mfcc}),
            #            torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * 16000/160)),
            #            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )

    #        self.transform_spec = torchaudio.transforms.Spectrogram(n_fft=320, hop_length=160, power=None)
    #        self.transform_inverse = torchaudio.transforms.InverseSpectrogram(n_fft=320, hop_length=160)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.meta['path'].iloc[idx]
        signal, sample_rate = torchaudio.load(file_name)

        if signal.shape[1] < 16000:
            padding_size = 16000 - signal.shape[1]
            signal = F.pad(signal, (0, padding_size))

        # Обрезаем сигнал до первых 16000 сэмплов
        signal = signal[:16000]

        #        signal = self.transform_spec(signal)
        #        signal = self.transform_inverse(signal)

        if self.augment:
            signal = self.augmentations(samples=signal.numpy(), sample_rate=16000)
            signal = torch.from_numpy(signal)

        spec = self.transform(signal)
        label = self.meta['label'].iloc[idx]

        return spec, self.labelmap[label]


labels = {
    'movix': 0,
    'other': 1
}

data = pd.DataFrame([
    {'label': i[0].split("\\")[-1], 'path': i[0] + "\\" + j}
    for i in os.walk(root_dir)
    for j in i[2]
])

# print(data.label.value_counts())
train, val, _, _ = train_test_split(data, data['label'], test_size=0.2)

train_dataset = CommandDataset(
    meta=train, root_dir=root_dir, sample_rate=16000, labelmap=labels, augment=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

val_dataset = CommandDataset(
    meta=val, root_dir=root_dir, sample_rate=16000, labelmap=labels, augment=False)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


model = M5()
model.to(device)

EPOCHS = 150
lr = 0.001
best_val_loss = float('inf')
epochs_without_improvement = 0

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if __name__ == '__main__':
    for epoch in range(EPOCHS):

        model.train()

        train_loss = []
        for batch, targets in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            batch = batch.to(device)
            targets = targets.to(device)
            predictions = model(batch)

            loss = F.nll_loss(predictions, targets)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print('Training loss:', np.mean(train_loss))

        model.eval()

        val_loss = []
        correct = 0
        all_preds = []
        all_targets = []

        for batch, targets in tqdm(val_dataloader, desc=f"Epoch: {epoch}"):
            with torch.no_grad():
                batch = batch.to(device)
                targets = targets.to(device)
                input = batch
                predictions = model(batch)

                loss = F.nll_loss(predictions, targets)

                pred = get_likely_index(predictions).to(device)
                correct += number_of_correct(pred, targets)

                val_loss.append(loss.item())

                # Сохраняем предсказания и метки
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Подсчет F-меры
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 score: {f1:.2f}')

        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            epochs_without_improvement = 0
            #            torch.save(model.state_dict(), 'best_model.pth')
            traced_model = torch.jit.trace(model, input)
            traced_model.save('model_traced.pt')

        #            model_dynamic_quantized = torch.quantization.quantize_dynamic(
        #                traced_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        #            traced_quantized_model = torch.jit.trace(
        #                model_dynamic_quantized, input, strict=False)
        #            optimized_traced_quantized_model = optimize_for_mobile(
        #                traced_quantized_model)
        #            optimized_traced_quantized_model.save('model_traced.pt')
        #            optimized_traced_quantized_model._save_for_lite_interpreter("best_model.ptl")
        else:
            epochs_without_improvement += 1

        # Ранняя остановка, если количество эпох без улучшений превысило patience
        if epochs_without_improvement >= 20:
            print(f'Early stopping at epoch {epoch + 1}')
        #            break

        print(
            f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_dataloader.dataset)} ({100. * correct / len(val_dataloader.dataset):.0f}%)\n")
        print('Val loss:', np.mean(val_loss))

        scheduler.step()

    torch.save(model.state_dict(), 'model.pth')

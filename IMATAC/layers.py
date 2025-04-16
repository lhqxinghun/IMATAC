import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.utils


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.hidden = []
        for i in hidden_feature:
            self.hidden.append(nn.Linear(in_feature, i))
            self.hidden.append(nn.Dropout(0.1))
            self.hidden.append(activation)
            in_feature = i        

        self.hidden.append(nn.Linear(in_feature, out_feature))
        self.hidden = nn.Sequential(*self.hidden)
    def forward(self, input):
        return self.hidden(input)


class Classifier(nn.Module):
    def __init__(self, in_feature, hidden_feature, n_class):
        super().__init__()       
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden_feature),
            nn.BatchNorm1d(hidden_feature),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature, n_class)
        )
    def forward(self, input):
        out = self.dense(input)
        out = F.softmax(out, dim=1)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [
                nn.Conv1d(in_channel, channel, 4, stride=2, padding=1),
                nn.Dropout(0.1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.Dropout(0.1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // 2, channel, 4, stride=2, padding=1),
                nn.Dropout(0.1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 3, padding=1),
            ]
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride):
        super(Decoder, self).__init__()
        blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]
        if stride == 2:
            blocks.extend(
                [
                    nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                    nn.Dropout(0.1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel, out_channel, 4, stride=2, padding=1),
                    nn.Dropout(0.1),
                    nn.ReLU(inplace=True),
                ]
            )
        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose1d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.Dropout(0.1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(channel // 2, out_channel, 3, padding=1),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
    def forward(self, input):
        return self.blocks(input)
import torch
from torch import nn
import torch.nn.functional as F
from .layers import Encoder, Decoder, Classifier, MLP
from .loss import UncertainLoss, WeightedMSELoss

import pandas as pd
from tqdm import tqdm



class IMATAC(nn.Module):
    def __init__(
        self,
        in_feature=0,
        num_components=30,
        dropout_rate=0.1,
        params={},
        dropout=True,
    ):
        super().__init__()
        self.num_components = num_components
        hidden_feature = int(in_feature / 2)
        self.dropout = dropout

        # MLP and Classifier
        self.mlp = MLP(hidden_feature, params['fully_connect_layer']['mlp']['hidden_dim'], params['fully_connect_layer']['mlp']['out_dim'], activation=nn.ReLU())
        self.classifier = Classifier(params['fully_connect_layer']['mlp']['out_dim'], params['fully_connect_layer']['classifier']['hidden_dim'], num_components)

        # Upsample MLP
        self.upsample = MLP(params['fully_connect_layer']['mlp']['out_dim'], params['fully_connect_layer']['upsample']['hidden_dim'], hidden_feature, activation=nn.ReLU())

        # Encoders and Decoders
        self.encoder_t = Encoder(params['conv_layer']['top_encoder']['hidden_channel'], params['conv_layer']['top_encoder']['out_channel'], stride=2)
        self.encoder_b = Encoder(params['conv_layer']['bottom_encoder']['in_channel'], params['conv_layer']['bottom_encoder']['hidden_channel'], stride=4)
        self.decoder_t = Decoder(params['conv_layer']['top_encoder']['out_channel'], params['conv_layer']['top_decoder']['hidden_channel'], params['conv_layer']['top_decoder']['out_channel'], stride=2)
        self.decoder = Decoder(params['conv_layer']['top_encoder']['out_channel'] + params['conv_layer']['top_decoder']['hidden_channel'], params['conv_layer']['bottom_decoder']['hidden_channel'], params['conv_layer']['bottom_encoder']['in_channel'], stride=4)

        # Upsample layers
        self.upsample_t = nn.Sequential(
            nn.Conv1d(params['conv_layer']['bottom_encoder']['in_channel'], params['conv_layer']['upsample_layer']['channel'], 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(params['conv_layer']['upsample_layer']['channel'], params['conv_layer']['upsample_layer']['channel'], 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(params['conv_layer']['bottom_encoder']['hidden_channel'] + params['conv_layer']['top_decoder']['out_channel'], params['conv_layer']['conv_layer']['channel'], 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(params['conv_layer']['conv_layer']['channel'], params['conv_layer']['conv_layer']['channel'] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(params['conv_layer']['conv_layer']['channel'] // 2, params['conv_layer']['conv_layer']['channel'] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(params['conv_layer']['conv_layer']['channel'] // 2, 1, 1),
        )

        self.dropout_rate = dropout_rate
        if not self.dropout:
            self.shut_dropout()

    def random_noise(self, input, p=0.1):
        x = torch.flatten(input, 1)
        mask = torch.bernoulli(torch.ones_like(x) * (1 - (p)))
        x = x * mask
        return x.reshape(input.shape)

    def shut_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def forward(self, input):
        if self.dropout:
            input = self.random_noise(input, self.dropout_rate)
        input = input.unsqueeze(1)
        enc_b, enc_t, dec_t = self.encode(input)
        enc_t = self.conv(torch.cat([dec_t, enc_b], 1))
        z = self.mlp(enc_t)
        predict = self.classifier(z.squeeze(1))
        enc_t = self.upsample(z)
        dec = self.decode(enc_t, enc_b).squeeze(1)
        return dec, predict

    def encode(self, input):
        enc_b = self.encoder_b(input)
        enc_t = self.encoder_t(enc_b)
        dec_t = self.decoder_t(enc_t)
        return enc_b, enc_t, dec_t

    def encode_code(self, input):
        input = input.unsqueeze(1)
        enc_b, enc_t, dec_t = self.encode(input)
        z = self.conv(torch.cat([dec_t, enc_b], 1))
        return self.mlp(z)

    def decode(self, enc_t, enc_b):
        upsample_t = self.upsample_t(enc_t)
        enc = torch.cat([upsample_t, enc_b], 1)
        return self.decoder(enc)

    def imputation(self, dataloader, device):
        output = pd.DataFrame()
        print('Processing...')
        for x, _, b, _ in dataloader:
            x = x.float().to(device)
            x = x.unsqueeze(1)
            enc_b, enc_t, dec_t = self.encode(x)
            enc_t = self.conv(torch.cat([dec_t, enc_b], 1))
            z = self.mlp(enc_t)
            enc_t = self.upsample(z)
            enc_t = nn.ReLU()(enc_t)
            dec = self.decode(enc_t, enc_b)
            dec = dec.squeeze(1)
            output = pd.concat([output, pd.DataFrame(dict(zip(b, dec.squeeze().detach().cpu().numpy())))], axis=1)
        print('Imputation finished!')
        return output

    def get_latent(self, dataloader, device):
        output = pd.DataFrame()
        for x, _, b, _ in dataloader:
            x = x.float().to(device)
            z = self.encode_code(x)
            output = pd.concat([output, pd.DataFrame(dict(zip(b, z.squeeze().detach().cpu().numpy())))], axis=1)
        return output

    def fit(self, dataloader, lr=0.0001, epochs=100, device='cuda:1'):
        self.to(device)
        weighted_loss_func = UncertainLoss(2).to(device)
        optimizer = torch.optim.Adamax(filter(lambda x: x.requires_grad, self.parameters()), lr=lr)

        for epoch in range(epochs):
            epoch_recon_loss, epoch_clas_loss = 0, 0
            tk0 = tqdm(dataloader, total=len(dataloader))
            for i, (x, label, _, label_onehot) in enumerate(tk0):
                x = x.float().to(device)
                optimizer.zero_grad()
                recon_x, predict = self(x)
                label = label_onehot
                label_onehot = torch.zeros(label_onehot.size(0), self.num_components).to(device)
                label = label.long().to(device)
                label_onehot.scatter_(1, label.view(-1, 1), 1)
                recon_loss = nn.MSELoss()(recon_x, x)
                clas_loss = nn.CrossEntropyLoss()(predict, label_onehot)
                loss = weighted_loss_func(recon_loss, clas_loss)
                loss.backward()
                optimizer.step()
                epoch_recon_loss += recon_loss.item()
                epoch_clas_loss += clas_loss.item()
                tk0.set_postfix_str(f'loss={loss:.3f} recon_loss={recon_loss:.5f} clas_loss={clas_loss:.3f}')
                tk0.update(1)
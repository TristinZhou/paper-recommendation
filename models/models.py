import torch
import torch.nn as nn


class BiLSTMPool(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pre_trained=None):
        super(BiLSTMPool, self).__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        if pre_trained is not None:
            self.token_embedding.weight.data.copy_(pre_trained.vectors)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.mean_pool = nn.AdaptiveAvgPool2d((1, None))
        self.linear = nn.Linear(embedding_dim*2, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def embedding(self, feat):
        h = self.token_embedding(feat)
        output = self.lstm(h)[0]
        output = self.linear(output)
        emb_max = self.max_pool(output).squeeze(dim=1)
        emb_mean = self.mean_pool(output).squeeze(dim=1)
        output = torch.cat([emb_mean, emb_max], axis=1)
        output = self.fc(output)
        return output

    def forward(self, inputs):
        if isinstance(inputs, list):
            return list(map(self.embedding, inputs))
        return self.embedding(inputs)


class TransformerPool(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pre_trained=None):
        super(TransformerPool, self).__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        if pre_trained is not None:
            self.token_embedding.weight.data.copy_(pre_trained.vectors)
        # self.lstm = nn.LSTM(embedding_dim,
        #                     hidden_dim,
        #                     batch_first=True,
        #                     bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=8)
        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.mean_pool = nn.AdaptiveAvgPool2d((1, None))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def embedding(self, feat):
        h = self.token_embedding(feat)
        h = h.reshape(h.shape[1], h.shape[0], -1)
        output = self.transformer_encoder(h)
        output = output.reshape(output.shape[1], output.shape[0], -1)
        emb_max = self.max_pool(output).squeeze(dim=1)
        emb_mean = self.mean_pool(output).squeeze(dim=1)
        output = torch.cat([emb_mean, emb_max], axis=1)
        output = self.fc(output)
        return output

    def forward(self, inputs):
        if isinstance(inputs, list):
            return list(map(self.embedding, inputs))
        return self.embedding(inputs)

from random import random
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, config, device):
        super(Encoder, self).__init__()
        rnn_config = config.get('rnn')
        dropout = rnn_config.get('dropout')
        source_vocabulary_size = config.get('source_vocabulary_size')
        self.device = device
        self.hidden_size = rnn_config.get('hidden_size')
        self.num_layers = rnn_config.get('num_layers')

        self.embedding = nn.Embedding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )

    def forward(self, batch):
        source_batch, source_lengths = batch.src
        _, batch_size = source_batch.shape
        embedded = self.embedding(source_batch)
        output, hidden = self.lstm(embedded)
        context = torch.empty((1, batch_size, self.hidden_size), device=self.device)
        for i in range(batch_size):
            index = source_lengths[i] - 1
            context[0, i] = output[index, i]
        return context, hidden


class Decoder(nn.Module):

    def __init__(self, config, device):
        super(Decoder, self).__init__()
        attention_config = config.get('attention')
        rnn_config = config.get('rnn')
        target_vocabulary_size = config.get('target_vocabulary_size')
        dropout = rnn_config.get('dropout')
        window_size = attention_config.get('window_size')
        self.device = device
        self.hidden_size = rnn_config.get('hidden_size')
        self.num_layers = rnn_config.get('num_layers')

        self.embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=2 * self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(
            in_features=self.hidden_size,
            out_features=target_vocabulary_size,
        )

    def forward(self, input, context, hidden):
        output = self.embedding(input)
        output = torch.cat((output, context), 2)
        output, hidden = self.lstm(output, hidden)
        output = self.fc1(output)
        return output, hidden


class ModelWithoutAttention(nn.Module):

    def __init__(self, config, device):
        super(ModelWithoutAttention, self).__init__()
        self.device = device
        self.encoder = Encoder(config, device)
        self.decoder = Decoder(config, device)
        self.target_vocabulary_size = config.get('target_vocabulary_size')
        self.teacher_forcing = config.get('teacher_forcing')
        self.eos = config.get('EOS')
        self.sos = config.get('SOS')

    def decode(self, input, context, hidden, batch_size):
        y, hidden = self.decoder(input, context, hidden)
        _, topi = y.topk(1)
        input = topi.detach().view(1, batch_size)
        y = y.view(batch_size, -1)
        return y, input, hidden

    def forward(self, batch, **kwargs):
        training = kwargs.get('training', True)
        sample = kwargs.get('sample', False)

        target_batch, _ = batch.trg
        T, batch_size = target_batch.shape

        context, hidden = self.encoder(batch)
        ys = torch.empty(T, batch_size, self.target_vocabulary_size, dtype=torch.float, device=self.device)
        if training:
            input = target_batch[0].unsqueeze(0)
            for i in range(T):
                if i != 0 and random() <= self.teacher_forcing:
                    input = target_batch[i].unsqueeze(0)
                y, input, hidden = self.decode(input, context, hidden, batch_size)
                ys[i] = y
            return ys
        else:
            input = torch.tensor([[self.sos] * batch_size], device=self.device, dtype=torch.long)
            translations = [[] for _ in range(batch_size)]
            for i in range(T):
                y, input, hidden = self.decode(input, context, hidden, batch_size)
                ys[i] = y

                for j in range(batch_size):
                    translations[j].append(input[0, j].item())

            if sample:
                return ys, translations, None
            else:
                return ys, translations

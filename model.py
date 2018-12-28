import math
from random import random
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, config, device):
        super(Encoder, self).__init__()
        rnn_config = config.get('rnn')
        source_vocabulary_size = config.get('source_vocabulary_size')
        dropout = rnn_config.get('dropout')
        self.window_size = config.get('window_size')
        self.pad = config.get('PAD_src')
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
        target_batch, _ = batch.trg
        batch_size = source_batch.shape[1]
        S = source_batch.size(0)
        T = target_batch.size(0)
        input = self.pad_with_window_size(source_batch)
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)
        context_indices = self.window_size + source_lengths - 1
        # select last word of encoded output
        context = torch.empty((1, batch_size, hidden[0].shape[2]), device=self.device)
        for i in range(batch_size):
            index = context_indices[i]
            context[0, i] = output[index, i]
        return output, hidden, context, S, T, batch_size

    def pad_with_window_size(self, batch):
        size = batch.size()
        n = len(size)
        if n == 2:
            length, batch_size = size
            padded_length = length + (2 * self.window_size + 1)
            padded = torch.empty((padded_length, batch_size), dtype=torch.long, device=self.device)
            padded[:self.window_size, :] = self.pad
            padded[self.window_size:self.window_size+length, :] = batch
            padded[-(self.window_size+1):, :] = self.pad
        elif n == 3:
            length, batch_size, hidden = size
            padded_length = length + (2 * self.window_size + 1)
            padded = torch.empty((padded_length, batch_size, hidden), dtype=torch.long, device=self.device)
            padded[:self.window_size, :, :] = self.pad
            padded[self.window_size:self.window_size+length, :, :] = batch
            padded[-(self.window_size+1):, :, :] = self.pad
        else:
            raise Exception(f'Cannot pad batch with {n} dimensions.')
        return padded


class Decoder(nn.Module):

    def __init__(self, config, device):
        super(Decoder, self).__init__()
        attention_config = config.get('attention')
        rnn_config = config.get('rnn')
        target_vocabulary_size = config.get('target_vocabulary_size')
        dropout = rnn_config.get('dropout')
        window_size = attention_config.get('window_size')
        self.device = device
        self.input_feeding = config.get('input_feeding')
        self.hidden_size = rnn_config.get('hidden_size')
        self.num_layers = rnn_config.get('num_layers')

        self.attention = Attention(window_size, self.hidden_size, device)
        self.embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        lstm_input_size = 2 * self.hidden_size if self.input_feeding else self.hidden_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
        )
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=target_vocabulary_size,
        )

    def forward(self, encoder_output, target_words, hidden, context, lengths, output_weights=False):
        T, batch_size = target_words.shape
        embedded = self.embedding(target_words)
        if self.input_feeding:
            input = torch.cat((embedded, context), 2)
        else:
            input = embedded
        output, hidden = self.lstm(input, hidden)
        attention = self.attention(encoder_output, output, lengths, T, batch_size, output_weights)
        if output_weights:
            c, weights = attention
        else:
            c = attention
        output = torch.cat((c, output), 2)
        output = self.relu(self.fc1(output))
        y = self.fc2(output)
        if output_weights:
            return y, hidden, c, weights
        else:
            return y, hidden, c


class Attention(nn.Module):

    def __init__(self, window_size, hidden_size, device):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.std_squared = (self.window_size / 2) ** 2
        self.hidden_size = hidden_size
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=math.ceil(hidden_size / 2))
        self.fc2 = nn.Linear(in_features=math.ceil(hidden_size / 2), out_features=1)

    def forward(self, encoder_output, decoder_output, lengths, T, batch_size, output_weights):
        s0 = lengths[0].item()
        lengths = lengths.view(batch_size, 1)
        window_length = 2 * self.window_size + 1
        # h_s: batch_size x (window_size + S + window_size) x hidden
        h_s = encoder_output
        h_s = h_s.permute(1, 0, 2)
        # h_t: batch_size x T x hidden
        h_t = decoder_output
        h_t = h_t.permute(1, 0, 2)

        # batch_size x T x 1
        p = self.tanh(self.fc1(h_t))
        p = self.sigmoid(self.fc2(p))
        p = p.view(batch_size, T)
        p = self.window_size + lengths.float() * p
        p = p.unsqueeze(2)

        window_start = torch.round(p - self.window_size).int()
        window_end = window_start + window_length
        positions = torch.empty((batch_size, T, window_length), device=self.device, dtype=torch.float)
        selection = torch.empty((batch_size, window_length, self.hidden_size), device=self.device, dtype=torch.float)
        for i in range(batch_size):
            for j in range(T):
                start = window_start[i, j].item()
                end = window_end[i, j].item()
                positions[i, j] = torch.arange(start, end, device=self.device, dtype=torch.float)
                selection[i] = h_s[i, start:end]

        # batch_size x T x window_length
        gaussian = torch.exp(-(positions - p) ** 2 / (2 * self.std_squared))
        gaussian = gaussian.view(batch_size, T, window_length)

        # batch_size x T x window_length
        epsilon = 1e-14
        score = self.score(selection, h_t)
        for i in range(batch_size):
            li = lengths[i].item()
            for j in range(T):
                start = window_start[i, j].item()
                end = window_end[i, j].item()
                if start < self.window_size:
                    d = self.window_size - start
                    score[i, j, :d] = epsilon
                if end > li + self.window_size:
                    d = (li + self.window_size) - end
                    score[i, j, d:] = epsilon

        # batch_size x T x window_length
        a = self.softmax(score)
        a = a * gaussian

        # batch_size x T x hidden_size
        c = torch.bmm(a, selection)

        # T x batch_size x hidden_size
        c = c.permute(1, 0, 2)

        if not output_weights:
            return c

        # insert weights of first sentence for eventual visualiation
        weights = torch.zeros((T, s0), device=self.device, dtype=torch.float)
        for j in range(T):
            start = window_start[0, j].item()
            end = window_end[0, j].item()
            if start < self.window_size and end > self.window_size + s0:
                # overflow in both ends
                weights_start = 0
                weights_end = s0
                a_start = self.window_size - start
                a_end = a_start + s0
            elif start < self.window_size:
                # overflow in left side only
                weights_start = 0
                weights_end = end - self.window_size
                a_start = self.window_size - start
                a_end = window_length
            elif end > self.window_size + s0:
                # overflow in right side only
                weights_start = start - self.window_size
                weights_end = s0
                a_start = 0
                a_end = a_start + (weights_end - weights_start)
            else:
                # a is contained in sentence
                weights_start = start - self.window_size
                weights_end = end - self.window_size
                a_start = 0
                a_end = window_length
            weights[j, weights_start:weights_end] = a[0, j, a_start:a_end]

        return c, weights

    def score(self, h_s, h_t):
        # h_s : batch x length x hidden
        # h_t : batch x T x hidden
        h_s = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s)


class Model(nn.Module):

    def __init__(self, config, device):
        super(Model, self).__init__()
        self.device = device
        self.encoder = Encoder(config, device)
        self.decoder = Decoder(config, device)
        self.teacher_forcing = config.get('teacher_forcing')
        self.eos = config.get('EOS')
        self.sos = config.get('SOS')
        self.pad_trg = config.get('PAD_trg')
        self.target_vocabulary_size = config.get('target_vocabulary_size')

    def decode(self, encoder_output, input, hidden, context, lengths, batch_size, output_weights):
        decoded = self.decoder(encoder_output, input, hidden, context, lengths, output_weights)
        if output_weights:
            y, hidden, context, attention = decoded
        else:
            y, hidden, context = decoded
        _, topi = y.topk(1)
        input = topi.detach().view(1, batch_size)
        context = context.detach()
        y = y.view(batch_size, -1)
        if output_weights:
            return y, input, hidden, context, attention
        else:
            return y, input, hidden, context

    def forward(self, batch, **kwargs):
        training = kwargs.get('training', True)
        sample = kwargs.get('sample', False)
        encoder_output, hidden, context, S, T, batch_size = self.encoder(batch)
        _, source_lengths = batch.src
        target_batch, _ = batch.trg

        ys = torch.empty(T, batch_size, self.target_vocabulary_size, dtype=torch.float, device=self.device)
        if training:
            input = target_batch[0].unsqueeze(0)
            for i in range(T):
                if i != 0 and random() <= self.teacher_forcing:
                    input = target_batch[i-1].unsqueeze(0)
                y, input, hidden, context = self.decode(encoder_output, input, hidden, context, source_lengths, batch_size, False)
                ys[i] = y
            return ys
        else:
            _, source_lengths = batch.src
            input = torch.tensor([[self.sos] * batch_size], device=self.device, dtype=torch.long)
            translations = [[] for _ in range(batch_size)]
            if sample:
                first_sentence_has_reached_end = False
                attention_weights = torch.zeros(0, source_lengths[0], device=self.device)
            for i in range(T):
                decoded = self.decode(encoder_output, input, hidden, context, source_lengths, batch_size, sample)
                if sample:
                    y, input, hidden, context, attention = decoded
                else:
                    y, input, hidden, context = decoded
                ys[i] = y

                for j in range(batch_size):
                    translations[j].append(input[0, j].item())

                if sample:
                    # don't add padding to attention visualiation
                    decoded_word = translations[0][i]
                    if not first_sentence_has_reached_end and decoded_word != self.pad_trg:
                        # add attention weights of first sentence in batch
                        attention_weights = torch.cat((attention_weights, attention))
                        first_sentence_has_reached_end = decoded_word == self.eos

            if sample:
                return ys, translations, attention_weights
            else:
                return ys, translations

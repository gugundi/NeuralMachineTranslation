import math
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, config, device):
        super(Encoder, self).__init__()
        rnn_config = config.get('rnn')
        source_vocabulary_size = config.get('source_vocabulary_size')
        dropout = rnn_config.get('dropout')
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

    def forward(self, source_sentence, hidden):
        embedded = self.embedding(source_sentence)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return h, c


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

        self.attention = Attention(window_size, self.hidden_size, device)
        self.embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
        )
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=target_vocabulary_size,
        )

    def forward(self, source_sentence_length, encoder_output, word, hidden, batch_size, lengths):
        embedded = self.embedding(word)
        output, hidden = self.lstm(embedded, hidden)
        c, a = self.attention(source_sentence_length, encoder_output, output, batch_size, lengths)
        output = torch.cat((c, output), 2)
        output = self.tanh(self.fc1(output))
        y = self.fc2(output)
        y = y.view(batch_size, -1)
        return y, output, hidden, a

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return h, c


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

    def forward(self, S, encoder_output, decoder_output, batch_size, lengths):
        s0 = lengths[0].item()
        lengths = lengths.view(batch_size, 1)
        window_length = 2 * self.window_size + 1
        # h_s: batch_size x (window_size + S + window_size) x hidden
        h_s = encoder_output
        h_s = h_s.permute(1, 0, 2)
        # h_t: batch_size x 1 x hidden
        h_t = decoder_output
        h_t = h_t.permute(1, 0, 2)

        # batch_size x 1
        p = self.tanh(self.fc1(h_t))
        p = self.sigmoid(self.fc2(p))
        p = p.view(batch_size, 1)
        p = self.window_size + lengths.float() * p

        window_start = torch.round(p - self.window_size).int()
        window_end = window_start + window_length
        positions = torch.empty((batch_size, window_length), device=self.device, dtype=torch.float)
        selection = torch.empty((batch_size, window_length, self.hidden_size), device=self.device, dtype=torch.float)
        for i in range(batch_size):
            start = window_start[i].item()
            end = window_end[i].item()
            positions[i] = torch.arange(start, end, device=self.device, dtype=torch.float)
            selection[i] = h_s[i, start:end]

        # batch_size x 1 x window_length
        gaussian = torch.exp(-(positions - p) ** 2 / (2 * self.std_squared))
        gaussian = gaussian.view(batch_size, 1, window_length)

        # batch_size x 1 x window_length
        epsilon = 1e-14
        score = self.score(selection, h_t)
        for i in range(batch_size):
            li = lengths[i].item()
            start = window_start[i].item()
            end = window_end[i].item()
            if start < self.window_size:
                d = self.window_size - start
                score[i, 0, :d] = epsilon
            if end > li + self.window_size:
                d = (li + self.window_size) - end
                score[i, 0, d:] = epsilon

        # batch_size x 1 x window_length
        a = self.softmax(score)
        a = a * gaussian

        # batch_size x 1 x hidden_size
        c = torch.bmm(a, selection)

        # 1 x batch_size x hidden_size
        c = c.permute(1, 0, 2)

        # insert weights of first sentence for eventual visualiation
        weights = torch.zeros((1, s0), device=self.device, dtype=torch.float)
        start = window_start[0].item()
        end = window_end[0].item()
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
        weights[0, weights_start:weights_end] = a[0, 0, a_start:a_end]

        return c, weights

    def score(self, h_s, h_t):
        # h_s : batch x length x hidden
        # h_t : batch x 1 x hidden
        h_s = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s)

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

    def init_hidden(self):
        h = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
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
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.fc1 = nn.Linear(
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
        )
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=target_vocabulary_size,
        )

    def forward(self, source_sentence_length, encoder_output, word, hidden):
        embedded = self.embedding(word)
        output, hidden = self.lstm(embedded, hidden)
        c, a = self.attention(source_sentence_length, encoder_output, output)
        output = torch.cat((c, output), 2)
        output = self.tanh(self.fc1(output))
        y = self.log_softmax(self.fc2(output))
        return y, output, hidden, a

    def init_hidden(self):
        h = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
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

    def forward(self, source_sentence_length, encoder_output, decoder_output):
        S = source_sentence_length
        h_s = encoder_output
        h_t = decoder_output

        p = self.tanh(self.fc1(h_t))
        p = S * self.sigmoid(self.fc2(p))
        window_start = torch.clamp(p - self.window_size, min=0)
        window_end = torch.clamp(p + self.window_size, max=S - 1)
        window_start = torch.round(window_start).int().item()
        window_end = torch.round(window_end).int().item()
        h_s = h_s[window_start:window_end+1]

        # batch first
        h_s = h_s.permute(1, 2, 0)
        h_t = h_t.permute(1, 0, 2)

        positions = torch.arange(window_start, window_end + 1, device=self.device, dtype=torch.float)
        gaussian = torch.exp((positions - p) / (2 * self.std_squared))
        a = self.softmax(self.score(h_s, h_t))
        a = a * gaussian

        # sequence before hidden size
        h_s = h_s.permute(0, 2, 1)

        c = torch.bmm(a, h_s)

        return c, (a, window_start, window_end)

    def score(self, h_s, h_t):
        return torch.bmm(h_t, h_s)

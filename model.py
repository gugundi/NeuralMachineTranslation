import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        rnn_config = config.get('rnn')
        source_vocabulary_size = config.get('source_vocabulary_size')
        dropout = rnn_config.get('dropout')
        self.batch_size = config.get('batch_size')
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
            batch_first=True,
        )

    def forward(self, source_sentence, hidden):
        embedded = self.embedding(source_sentence).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h, c


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        attention_config = config.get('attention')
        rnn_config = config.get('rnn')
        target_vocabulary_size = config.get('target_vocabulary_size')
        dropout = rnn_config.get('dropout')
        window_size = attention_config.get('window_size')
        self.batch_size = config.get('batch_size')
        self.hidden_size = rnn_config.get('hidden_size')
        self.num_layers = rnn_config.get('num_layers')

        self.attention = Attention(window_size, self.hidden_size)
        self.embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=2 * self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
        )
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=target_vocabulary_size,
        )

    def forward(self, source_sentence_length, source_hiddens, prev_word, prev_c, previous_hidden):
        embedded = self.embedding(prev_word)
        lstm_input = torch.cat((embedded, prev_c), 2)
        _, hidden = self.lstm(lstm_input, previous_hidden)
        target_hidden, _ = hidden
        c = self.attention(source_sentence_length, source_hiddens, target_hidden)
        h_t = target_hidden.view(1, -1)
        h_t = torch.cat((c, h_t), 1)
        h_t = self.tanh(self.fc1(h_t))
        y = self.log_softmax(self.fc2(h_t))
        return y, h_t, hidden

    def init_hidden(self):
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h, c


class Attention(nn.Module):

    def __init__(self, window_size, hidden_size):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.std_squared = (self.window_size / 2) ** 2
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        n_intermediate_features = min(hidden_size / 2, 4 * window_size)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=n_intermediate_features)
        self.fc2 = nn.Linear(in_features=n_intermediate_features, out_features=1)

    def forward(self, source_sentence_length, source_hiddens, target_hidden):
        S = source_sentence_length
        h_s = source_hiddens
        h_t = target_hidden

        p_t = self.tanh(self.fc1(h_t))
        p_t = S * self.sigmoid(self.fc2(p_t))
        window_start = torch.clamp(torch.ceil(p_t - self.window_size), min=0).int().item()
        window_end = torch.clamp(torch.floor(p_t + self.window_size), max=S - 1).int().item()
        h_s = h_s[window_start:window_end+1]

        e_t = torch.exp((S - p_t) / (2 * self.std_squared))
        a_t = self.softmax(self.score(h_s, h_t)) * e_t
        a_t = a_t.view(1, -1)
        c_t = torch.mm(a_t, h_s)

        return c_t

    def score(self, h_s, h_t):
        h_t = h_t.view(self.hidden_size, 1)
        return torch.mm(h_s, h_t)

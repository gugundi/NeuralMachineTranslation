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
        )

    def forward(self, source_sentence, hidden):
        embedded = self.embedding(source_sentence)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
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
        self.batch_size = config.get('batch_size')
        self.hidden_size = rnn_config.get('hidden_size')
        self.num_layers = rnn_config.get('num_layers')

        self.attention = Attention(window_size, self.hidden_size, self.batch_size, device)
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
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
        return h, c


class Attention(nn.Module):

    def __init__(self, window_size, hidden_size, batch_size, device):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.std_squared = (self.window_size / 2) ** 2
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=math.ceil(hidden_size / 2))
        self.fc2 = nn.Linear(in_features=math.ceil(hidden_size / 2), out_features=1)

    def forward(self, source_sentence_length, encoder_output, decoder_output):
        S = source_sentence_length
        h_s_batch = encoder_output
        h_t_batch = decoder_output

        p_batch = self.tanh(self.fc1(h_t_batch))
        p_batch = S * self.sigmoid(self.fc2(p_batch))

        max_window_size = 2 * self.window_size + 1
        h_s = torch.zeros(max_window_size, self.batch_size, self.hidden_size, device=self.device, dtype=torch.float)
        window_start_batch = torch.zeros(self.batch_size, 1, device=self.device, dtype=torch.int)
        window_end_batch = torch.zeros(self.batch_size, 1, device=self.device, dtype=torch.int)
        gaussian_batch = torch.zeros(self.batch_size, 1, max_window_size, device=self.device, dtype=torch.float)
        for i in range(self.batch_size):
            p = p_batch[0, i]
            window_start = torch.clamp(p - self.window_size, min=0)
            window_end = torch.clamp(p + self.window_size, max=S - 1)
            window_start = torch.round(window_start).int().item()
            window_end = torch.round(window_end).int().item()
            window_start_batch[i] = window_start
            window_end_batch[i] = window_end
            window_size = window_end - window_start + 1
            h_s[:window_size, i] = h_s_batch[window_start:window_end+1, i]
            positions = torch.arange(window_start, window_end + 1, device=self.device, dtype=torch.float)
            print('window size: {window_size}')
            print(torch.exp((positions - p) / (2 * self.std_squared)).shape)
            gaussian_batch[i, :window_size] = torch.exp((positions - p) / (2 * self.std_squared))
        h_s_batch = h_s

        # batch x window x hidden
        h_s_batch = h_s_batch.permute(1, 0, 2)
        h_t_batch = h_t_batch.permute(1, 0, 2)

        # batch x 1 x window
        a_batch = self.softmax(self.score(h_s_batch, h_t_batch))
        a_batch = a_batch * gaussian_batch

        c = torch.bmm(a_batch, h_s_batch)

        # 1 x batch x window
        c = c.permute(1, 0, 2)

        return c, (a_batch, window_start_batch, window_end_batch)

    def score(self, h_s, h_t):
        # h_s : batch x length x hidden
        # h_t : batch x 1 x hidden
        h_s = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s)

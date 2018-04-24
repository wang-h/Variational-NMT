import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Customed stacked LSTM, which we can custom the first layer size.
    """
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.LSTMCell(input_size, hidden_size))
            else:
                self.layers.append(nn.LSTMCell(hidden_size, hidden_size))

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.GRUCell(input_size, hidden_size))
            else:
                self.layers.append(nn.GRUCell(hidden_size, hidden_size))
            

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, (h_1,)

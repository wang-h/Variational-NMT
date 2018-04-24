import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class PackedRNNEncoder(nn.Module):
    def __init__(self, rnn_type, embed_dim, 
                 hidden_size, num_layers=2, dropout=0.0, bidirectional=True):
        super(PackedRNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(embed_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.hidden_size = hidden_size * num_directions
    def fix_final_state(self, final_state):
        def resize(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], -1)
            return h
        if self.bidirectional:
            if self.rnn_type == "GRU":
                final_state = resize(final_state)
            elif self.rnn_type == "LSTM":
                final_state = tuple([resize(h) for h in final_state])
        return final_state
    def forward(self, input, lengths=None, state=None):
        if lengths is not None:
            packed = pack_padded_sequence(input, lengths.view(-1).tolist())    
        output, final_state = self.rnn(packed, state)

        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return output, self.fix_final_state(final_state)

class RNNEncoder(PackedRNNEncoder):
    def __init__(self, *arg, **kwargs):
        super(RNNEncoder, self).__init__(*arg, **kwargs)

    def forward(self, input, lengths=None, state=None):

        if lengths is not None:
            lengths, rank = torch.sort(lengths, dim=0, descending=True)
            input = input.index_select(1, rank)
            output, final_state = super(RNNEncoder, self).forward(input, lengths)
            _, order = torch.sort(rank, dim=0, descending=False)
        if isinstance(final_state, tuple):
            final_state = tuple(x.index_select(1, order) for x in final_state)
        else:
            final_state = final_state.index_select(1, order)
        return output.index_select(1, order), final_state
 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from NMT.Models.Decoders import RNNDecoderState

class NMTModel(nn.Module):
    """
    Core model for NMT.
    {
        Encoder + Decoder.
    }
    """
    def __init__(self, 
            encoder, decoder, 
            src_embedding, trg_embedding,
            trg_vocab_size, 
            config):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.src_embedding = src_embedding

        self.trg_embedding = trg_embedding

        self.decoder = decoder
        self.generator = nn.Linear(config.hidden_size, trg_vocab_size)
        self.config = config


    def encoder2decoder(self, encoder_state):
        if isinstance(encoder_state, tuple):  
            # LSTM: encoder_state = (hidden, state)
            return RNNDecoderState(self.encoder.hidden_size, encoder_state)
        else:  
            # GRU: encoder_state = state
            return RNNDecoderState(self.encoder.hidden_size, encoder_state)
   
    def forward(self, src, lengths, trg, decoder_state=None):
        """
        Forward propagate a `src` and `trg` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): source sequence. [L x B x N]`.
            trg (LongTensor): source sequence. [L x B]`.
            lengths (LongTensor): the src lengths, pre-padding `[batch]`.
            dec_state (`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`nmt.Models.DecoderState`):

                 * decoder output `[trg_len x batch x hidden]`
                 * dictionary attention dists of `[trg_len x batch x src_len]`
                 * final decoder state
        """
        trg = trg[:-1]
        # encoding side
        encoder_outputs, encoder_state = self.encoder(
            self.src_embedding(src), lengths)
        
        # encoder to decoder
        decoder_state = self.encoder2decoder(encoder_state)
        
        decoder_input = self.trg_embedding(trg)
        # decoding side
        decoder_outputs, decoder_state, attns = self.decoder(
            decoder_input, encoder_outputs, lengths, decoder_state)

        return decoder_outputs, decoder_state, attns

  

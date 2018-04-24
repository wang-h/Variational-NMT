import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from NMT.Models import NMTModel


class VRNMTModel(NMTModel):
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
        decoder_outputs, decoder_state, attns, kld = self.decoder(
            decoder_input, encoder_outputs, lengths, decoder_state)

        return decoder_outputs, decoder_state, attns, kld

  

  
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from NMT.Models import NMTModel
from NMT.Models.Decoders import RNNDecoderState

def compute_kld(mu, logvar):
    kld = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
    return kld


class VNMTModel(NMTModel):
    """Recent work has found that VAE + LSTM decoders underperforms vanilla LSTM decoder.
    You should use VAE + GRU.
    see Yang et. al, ICML 2017, `Improved VAE for Text Modeling using Dilated Convolutions`.
    """
    def __init__(self, 
                encoder, decoder, 
                src_embedding, trg_embedding, 
                trg_vocab_size, 
                config):
        super(VNMTModel, self).__init__(
                        encoder, decoder, 
                        src_embedding, trg_embedding, 
                        trg_vocab_size, config)
        self.context_to_mu = nn.Linear(
                        config.hidden_size, 
                        config.latent_size)
        self.context_to_logvar = nn.Linear(
                        config.hidden_size, 
                        config.latent_size)
        self.lstm_state2context = nn.Linear(
                        2*config.hidden_size, 
                        config.latent_size)
    def get_hidden(self, state):
        hidden = None
        if self.encoder.rnn_type == "GRU":
            hidden = state[-1]
        elif self.encoder.rnn_type == "LSTM":
            hidden, context = state[0][-1], state[1][-1]
            hidden = self.lstm_state2context(torch.cat([hidden, context], -1))
        return hidden

    def reparameterize(self, encoder_state):
        """
        context [B x 2H]
        """
        hidden = self.get_hidden(encoder_state)
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z, mu, logvar


    def forward(self, src, src_lengths, trg, trg_lengths=None, decoder_state=None):
        """
        Forward propagate a `src` and `trg` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): source sequence. [L x B x N]`.
            trg (LongTensor): source sequence. [L x B]`.
            src_lengths (LongTensor): the src lengths, pre-padding `[batch]`.
            trg_lengths (LongTensor): the trg lengths, pre-padding `[batch]`.
            dec_state (`DecoderState`, optional): initial decoder state
            z (`FloatTensor`): latent variables
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`nmt.Models.DecoderState`):

                 * decoder output `[trg_len x batch x hidden]`
                 * dictionary attention dists of `[trg_len x batch x src_len]`
                 * final decoder state
        """
        
        # encoding side
        encoder_outputs, encoder_state = self.encoder(
            self.src_embedding(src), src_lengths)

        # re-parameterize
        z, mu, logvar = self.reparameterize(encoder_state)
        # encoder to decoder
        decoder_state = self.encoder2decoder(encoder_state)
        
        trg_feed = trg[:-1]
        decoder_input = torch.cat([
                self.trg_embedding(trg_feed), 
                z.unsqueeze(0).repeat(trg_feed.size(0) ,1, 1)],
                -1)
        
        # decoding side
        decoder_outputs, decoder_state, attns = self.decoder(
            decoder_input, encoder_outputs, src_lengths, decoder_state)

        return decoder_outputs, decoder_state, attns, compute_kld(mu, logvar)

  
    

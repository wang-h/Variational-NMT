import torch
import torch.nn as nn
from torch.autograd import Variable
from NMT.Modules import StackedLSTM
from NMT.Modules import StackedGRU
from NMT.Modules import GlobalAttention


class RNNDecoderState(object):
    def __init__(self, hidden_size, rnn_state):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnn_state: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if isinstance(rnn_state, tuple):
            # LSTM
            self.state = rnn_state
        else:
            # GRU
            self.state = (rnn_state, )
        
        self.coverage = None

        # Init the input feed.
        batch_size = self.state[0].size(1)

        self.input_feed = Variable(
            self.state[0].data.new(batch_size, hidden_size).zero_()).unsqueeze(0)

    @property
    def _all(self):
        return self.state + (self.input_feed,)


    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def update_state(self, rnn_state, input_feed):
        if not isinstance(rnn_state, tuple):
            self.state = (rnn_state,)
        else:
            self.state = rnn_state
        self.input_feed = input_feed

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        
        repeat_func = lambda x: Variable(
            x.data.repeat(1, beam_size, 1), requires_grad=False)
        vars = tuple(repeat_func(h) for h in self._all)
        self.state = vars[:-1]
        self.input_feed = vars[-1]

class RNNDecoderBase(nn.Module):
    def __init__(self, rnn_type, 
                input_size,
                hidden_size, 
                num_layers=2, 
                attn_type="general",
                bidirectional_encoder=True,
                dropout=0.0, 
                embeddings=None):

        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.rnn_type = rnn_type
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

     
        self.attn = GlobalAttention(
            hidden_size, attn_type=attn_type
        )

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        elif rnn_type == "GRU":
            stacked_cell = StackedGRU
        else:
            raise NotImplementedError
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    def forward(self, trg, encoder_outputs, lengths, state):
        """
        Args:
            trg (`LongTensor`): sequences of padded tokens
                                `[L_t x B x D]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[L_s x B x H]`.
            
            lengths (`LongTensor`): the padded source lengths
                `[B]`.
            state (`DecoderState`):
                 decoder state object to initialize the decoder
        Returns:
            (`FloatTensor`,:obj:`nmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[trg_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over source words at each target word
                        `[L_t x B x L_s]`.
        """
        # Run the forward pass of the RNN.
        decoder_outputs, final_state, attns = self.forward_step(
            trg, encoder_outputs, lengths, state)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        state.update_state(final_state, final_output.unsqueeze(0))

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns



class InputFeedRNNDecoder(RNNDecoderBase):
    def __init__(self, rnn_type,
                embedding_size, 
                hidden_size, 
                num_layers=2, 
                attn_type="general",
                bidirectional_encoder=True,
                dropout=0.0):
        super(InputFeedRNNDecoder, self).__init__(rnn_type,
                embedding_size + hidden_size, 
                hidden_size, 
                num_layers, 
                attn_type,
                bidirectional_encoder,
                dropout)

    def forward_step(self, trg, encoder_outputs, lengths, dec_state):
        """
        Input feed concatenates hidden state with input at every time step.

        Args:
            trg (`LongTensor`): sequences of padded tokens
                                `[L_t x B x nfeats]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[L_s x B x H]`.
            state (`DecoderState`):
                 decoder state object to initialize the decoder
            lengths (`LongTensor`): the padded source lengths
                `[B]`.
        Returns:
            (`FloatTensor`,:obj:`nmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[L_t x B x H]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over source words at each target word
                        `[L_t x B x L_s]`.
        """
        
        input_feed = dec_state.input_feed.squeeze(0)         # [B x H]

        decoder_outputs = []
        attns = {"std": []}                         

        rnn_state = dec_state.state
        
        for t, emb_t in enumerate(trg.split(1, dim=0)):
            # iterate over each target word
            emb_t = emb_t.squeeze(0)
            # teacher forcing
            decoder_input = torch.cat([emb_t, input_feed], 1)
            # update state and feed to next RNNCell
            rnn_output, rnn_state = self.rnn(decoder_input, rnn_state)
            
            decoder_output, attn = self.attn(
                rnn_output, encoder_outputs.transpose(0, 1), lengths=lengths)
           
            decoder_output = self.dropout(decoder_output)

            input_feed = decoder_output
            decoder_outputs += [decoder_output]
            attns["std"] += [attn]

        # Return result.
        return decoder_outputs, rnn_state, attns

class VarInputFeedRNNDecoder(RNNDecoderBase):
    def __init__(self, rnn_type,
                embedding_size, 
                hidden_size,
                latent_size, 
                num_layers=2, 
                attn_type="general",
                bidirectional_encoder=True,
                dropout=0.0):
        super(VarInputFeedRNNDecoder, self).__init__(rnn_type,
                embedding_size + hidden_size + latent_size, 
                hidden_size, 
                num_layers, 
                attn_type,
                bidirectional_encoder,
                dropout)

        self.context_to_mu = nn.Linear(
                        hidden_size, 
                        latent_size)
        self.context_to_logvar = nn.Linear(
                        hidden_size, 
                        latent_size)

    def reparameterize(self, state):
        """
        context [B x 2H]
        """
        hidden = self.get_hidden(state)
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z, mu, logvar


    def get_hidden(self, state):
        hidden = None
        if self.rnn_type == "GRU":
            hidden = state[-1]
        elif self.rnn_type == "LSTM":
            hidden = state[0][-1]
        return hidden
    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        return kld
    def forward_step(self, trg, encoder_outputs, lengths, dec_state):
        """
        Input feed concatenates hidden state with input at every time step.

        Args:
            trg (`LongTensor`): sequences of padded tokens
                                `[L_t x B x nfeats]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[L_s x B x H]`.
            state (`DecoderState`):
                 decoder state object to initialize the decoder
            lengths (`LongTensor`): the padded source lengths
                `[B]`.
        Returns:
            (`FloatTensor`,:obj:`nmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[L_t x B x H]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over source words at each target word
                        `[L_t x B x L_s]`.
        """
        
        input_feed = dec_state.input_feed.squeeze(0)         # [B x H]

        decoder_outputs = []
        attns = {"std": []}                         

        rnn_state = dec_state.state
        
        kld = 0.
        for t, emb_t in enumerate(trg.split(1, dim=0)):
            # iterate over each target word
            emb_t = emb_t.squeeze(0)
            # teacher forcing
            z, mu, logvar = self.reparameterize(rnn_state[0])
            decoder_input = torch.cat([emb_t, input_feed, z], -1)
            kld += self.compute_kld(mu, logvar)
            # update state and feed to next RNNCell
            rnn_output, rnn_state = self.rnn(decoder_input, rnn_state)

            decoder_output, attn = self.attn(
                rnn_output, encoder_outputs.transpose(0, 1), lengths=lengths)
           
            decoder_output = self.dropout(decoder_output)

            input_feed = decoder_output
            decoder_outputs += [decoder_output]
            attns["std"] += [attn]

        # Return result.
        return decoder_outputs, rnn_state, attns, kld
    def forward(self, trg, encoder_outputs, lengths, state):
        """
        Args:
            trg (`LongTensor`): sequences of padded tokens
                                `[L_t x B x D]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[L_s x B x H]`.
            
            lengths (`LongTensor`): the padded source lengths
                `[B]`.
            state (`DecoderState`):
                 decoder state object to initialize the decoder
        Returns:
            (`FloatTensor`,:obj:`nmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[trg_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over source words at each target word
                        `[L_t x B x L_s]`.
        """
        # Run the forward pass of the RNN.
        decoder_outputs, final_state, attns, kld = self.forward_step(
            trg, encoder_outputs, lengths, state)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        state.update_state(final_state, final_output.unsqueeze(0))

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns, kld

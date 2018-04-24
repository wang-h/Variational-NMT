import torch
from torch.autograd import Variable
import torch.nn.functional as F
from NMT.translate.Beam import Beam

from Utils.DataLoader import PAD_WORD
from Utils.DataLoader import BOS_WORD
from Utils.DataLoader import EOS_WORD


from NMT.Models import RNNEncoder
from NMT.Models import VNMTModel

class BatchTranslator(object):
    """
    Uses a model to translate a batch of sentences.

    Args:
       model (:obj:`nmt.Modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       k_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
    """

    def __init__(self, model, config, trg_vocab, global_scorer):
        self.config = config
        self.vocab = trg_vocab
        self.model = model
        self.k_best = config.k_best
        self.max_length = 100
        self.global_scorer = global_scorer
        self.beam_size = config.beam_size
        self.stepwise_penalty = config.stepwise_penalty
        self.block_ngram_repeat = config.block_ngram_repeat
        self.ignore_when_blocking = set(config.ignore_when_blocking)

        self.PAD_WID = trg_vocab.stoi[PAD_WORD]
        self.BOS_WID = trg_vocab.stoi[BOS_WORD]
        self.EOS_WID = trg_vocab.stoi[EOS_WORD]
       

    def beam_search(self, batch_size, encoder_outputs, decoder_states, src_lengths, z=None):
        """
        beam search. 

        Args:
           batch (`Batch`): a batch from a dataset object
           encoder_outputs (`Variable`): the outputs of encoder hidden layer 
           decoder_states
        """
        beam = [Beam(self.beam_size, self.PAD_WID, self.BOS_WID, self.EOS_WID,
                     self.config, global_scorer=self.global_scorer)
                for _ in range(batch_size)]
        if z is not None:
            z = Variable(z.data.repeat(1, self.beam_size, 1), requires_grad=True)
            
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break
            trg = Variable(torch.stack(
                        [cad.get_current_state() for cad in beam]
                    ).t().contiguous().view(1, -1))
            trg = trg.unsqueeze(2)
            # Run one step.
            
            decoder_input = self.model.trg_embedding(trg)
            if isinstance(self.model, VNMTModel):
                decoder_input = torch.cat([decoder_input, z], -1)
            decoder_output, decoder_states, attn = self.model.decoder(
                    decoder_input, encoder_outputs, src_lengths, decoder_states)[:3]
            
            decoder_output = decoder_output.squeeze(0)  # [b x H]

            # (b) Compute a vector of batch x beam word scores.
            output = F.log_softmax(self.model.generator(decoder_output).data, dim=-1)
            output = output.view(self.beam_size, batch_size, -1)
            beam_attn = attn["std"].view(self.beam_size, batch_size, -1)

            for j, b in enumerate(beam):
                b.advance(output[:, j], beam_attn.data[:, j, :src_lengths[j]])
                decoder_states.beam_update(
                    j, b.get_current_origin(), self.beam_size)
        return beam

    def translate_batch(self, batch, exclusion_tokens=[]):
        """
        Translate a batch of sentences. 

        Args:
           batch (Batch): a batch from a dataset object
        """
        batch_size = batch.batch_size

        # 1. encoding
        src, src_lengths = batch.src, batch.src_Ls
        encoder_output, encoder_state = self.model.encoder(
            self.model.src_embedding(src), src_lengths)
        z = None
        if isinstance(self.model, VNMTModel):
            # re-parameterize
            z, mu, logvar = self.model.reparameterize(encoder_state)
        # encoder to decoder
        decoder_state = self.model.encoder2decoder(encoder_state)
        
        # decoder_input = torch.cat([
        #         self.model.trg_embedding(trg), 
        #         z.unsqueeze(0).repeat(trg.size(0) ,1, 1)],
        #         -1) 

        # 2. repeat source `beam_size` times.
        
        encoder_outputs = Variable(
                    encoder_output.data.repeat(
                        1, self.beam_size, 1), 
                    requires_grad=False)

        src_lengths = src_lengths.repeat(self.beam_size)
        decoder_states = decoder_state
        decoder_states.repeat_beam_size_times(self.beam_size)

        # 3. Generate translations using beam search.
        beam = self.beam_search(
                batch_size, encoder_outputs, decoder_states, src_lengths, z)

        # 4. Extract sentences from beam.
        batch_trans = self.extract_trans_from_beam(beam)

        batch_trans["gold_score"] = self._run_target(batch)
        batch_trans["batch"] = batch
        return batch_trans

    def extract_trans_from_beam(self, beam):
        """
        extract translations from beam.
        """
        ret = {"predictions": [],
               "scores": [],
               "attention": []
               }

        for b in beam:
            scores, ks = b.sort_finished(minimum=self.k_best)
            hypos, attn = [], []
            for i, (times, k) in enumerate(ks[:self.k_best]):
                hypo, att = b.get_hypo(times, k)
                hypos.append(hypo)
                attn.append(att)
            ret["predictions"].append(hypos)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch):
        src_lengths = batch.src_Ls
        src = batch.src
        trg_in = batch.trg[:-1]
        trg_out = batch.trg[1:]
        #  (1) run the encoder on the src
        
        encoder_outputs, encoder_state = self.model.encoder(
            self.model.src_embedding(src), src_lengths)
        
        z = None
        if isinstance(self.model, VNMTModel):
            z, mu, logvar = self.model.reparameterize(encoder_state)
        
        decoder_state = self.model.encoder2decoder(encoder_state)
        
        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        gold_scores = torch.FloatTensor(batch.batch_size).fill_(0).cuda()
        decoder_input = self.model.trg_embedding(trg_in)
        if isinstance(self.model, VNMTModel):
            decoder_input = torch.cat(
                [decoder_input, 
                z.unsqueeze(0).repeat(trg_in.size(0),1, 1)],
                -1)    
        decoder_output, decoder_states, attn = self.model.decoder(
                    decoder_input, encoder_outputs, src_lengths, decoder_state)[:3]
       

        trg_pad = self.vocab.stoi[PAD_WORD]
        for dec, trg in zip(decoder_output, trg_out.data):
            # Log prob of each word.
            out = F.log_softmax(self.model.generator(dec).data, dim=-1)
            #trg = trg.unsqueeze(1)
            #print(trg.size(),  out.size())
            scores = out.data.gather(1, trg)
            scores.masked_fill_(trg.eq(trg_pad), 0)
            gold_scores += scores.squeeze().float()
        return gold_scores

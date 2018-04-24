import torch
from Utils.DataLoader import EOS_WORD
from Utils.DataLoader import UNK_WORD
from Utils.utils import trace


class TranslationBuilder(object):
    """
    Luong et al, 2015. Addressing the Rare Word Problem in Neural Machine Translation.
    """
    def __init__(self, src_vocab, trg_vocab, config):
        """
        Args:
        src_vocab (Vocab): source vocabulary
        trg_vocab (Vocab): source vocabulary
        replace_unk (bool): replace unknown words using attention
        """
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.replace_unk = config.replace_unk
        self.k_best = config.k_best

    def _build_sentence(self, src, pred, vocab, attn):
        """
        build sentence using predicted output with the given vocabulary.
        """
        tokens = []
        for wid in pred:
            token = vocab.itos[int(wid)]
            if token == EOS_WORD:
                break
            tokens.append(token)
            
        if self.replace_unk and (attn is not None) and (src is not None):
            for i in range(len(tokens)):
                if tokens[i] == UNK_WORD:
                    _, max_ = attn[i].max(0)
                    tokens[i] = src[int(max_[0])]
        return tokens

    def from_batch_translator_output(self, outputs):
        """
        build translation from batch output 
        """
        batch = outputs["batch"]
        batch_size = batch.batch_size
        preds, pred_score, attns, gold_score = list(zip(*zip(
            outputs["predictions"],
            outputs["scores"],
            outputs["attention"],
            outputs["gold_score"])))
       
        src = batch.src.data
        trg = batch.trg.data

        translations = []
        for b in range(batch_size):
            pred_sents = [self._build_sentence(
                src[:,b:,0], preds[b][n], self.trg_vocab, attns[b][n]) for n in range(self.k_best)]
            gold = trg[1:,b:,0].squeeze().cpu().numpy()
            input = src[:,b:,0].squeeze().cpu().numpy()

            input_sent = self._build_sentence(src[:,b:,0], input, self.src_vocab, None)
            gold_sent = self._build_sentence(src[:,b:,0], gold, self.trg_vocab, None)
            
            translation = Translation(input_sent[1:], pred_sents,
                                      attns[b], pred_score[b], gold_sent,
                                      gold_score[b])
            translations.append(translation)
        
        return translations


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention distributions for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """
    def __init__(self, src_sent, pred_sents,
                 attns, pred_scores, trg_sent, gold_score):
        self.src_sent = src_sent
        self.pred_sents = pred_sents
        self.attns = attns
        self.pred_scores = pred_scores
        self.gold_sent = trg_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation to stdout.
        """
        output = '\nINPUT {}: {}\n'.format(sent_number, " ".join(self.src_sent))
        
        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        print("PRED SCORE: {:.4f}".format(best_score))

        if self.gold_sent is not None:
            trg_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, trg_sent)
            # output += ("GOLD SCORE: {:.4f}".format(self.gold_score))
            trace("GOLD SCORE: {:.4f}".format(self.gold_score))
        if len(self.pred_sents) > 1:
            trace('\nBEST HYP:')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output

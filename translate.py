#!/usr/bin/env python
import os
import argparse
import math
import codecs
import torch

from itertools import count

from Utils.args import parse_args
from Utils.utils import trace
from Utils.config import read_config
from Utils.DataLoader import DataBatchIterator
from NMT.ModelConstructor import model_factory
from NMT.Optimizer import Optimizer

from NMT.Loss import NMTLoss
from NMT.Trainer import Trainer
from NMT.Trainer import Statistics
from NMT.translate import BatchTranslator
from NMT.translate import TranslationBuilder
from NMT.translate import GNMTGlobalScorer
from Utils.plot import plot_attn
from Utils.utils import report_bleu
from Utils.utils import report_rouge

def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def main():
    args, parser = parse_args("translate")
    config = read_config(args, parser, args.config)
    config.batch_size = 1
    test_data_iter = DataBatchIterator(
        config=config, is_train=False, dataset="test", batch_size=config.batch_size)
    
    src_vocab = torch.load(config.save_vocab + "." + config.src_lang)
    trg_vocab = torch.load(config.save_vocab + "." + config.trg_lang)

    test_data_iter.set_vocab(src_vocab, trg_vocab)
    test_data_iter.load()

    checkpoint = torch.load(config.save_model+".pt")
    # Load the model.
    model =  model_factory(
        config, src_vocab, trg_vocab, train_mode=False, checkpoint=checkpoint)
    if config.verbose:
        trace(model)
    # File to write sentences to.
    pred_file = codecs.open(config.output+".pred.txt", 'w', 'utf-8')
    ref_file = codecs.open(config.output+".ref.txt", 'w', 'utf-8')
    src_file = codecs.open(config.output+".src.txt", 'w', 'utf-8')
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
   

    # Translator
    scorer = GNMTGlobalScorer(config.alpha, config.beta, config.coverage_penalty,
                                             config.length_penalty)
    translator = BatchTranslator(model, config, trg_vocab, global_scorer=scorer)

    data_iter = iter(test_data_iter)

    builder = TranslationBuilder(src_vocab, trg_vocab, config)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0

    pred_list = []
    gold_list = []
    for batch in data_iter:
        outputs = translator.translate_batch(batch)
        batch_trans = builder.from_batch_translator_output(outputs)
        
        for trans in batch_trans:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            pred_list.append(trans.pred_sents[0])

            gold_score_total += trans.gold_score
            gold_words_total += len(trans.gold_sent) + 1
            gold_list.append(trans.gold_sent)

            k_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:config.k_best]]
            #print(" ".join(trans.gold_sent)                         
            pred_file.write('\n'.join(k_best_preds)+"\n")
            ref_file.write(" ".join(trans.gold_sent)+'\n')
            src_file.write(" ".join(trans.src_sent)+'\n')
            if config.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))

                report_score('PRED', pred_score_total, pred_words_total)
                report_score('GOLD', gold_score_total, gold_words_total)
            if config.plot_attn:
                plot_attn(trans.src_sent, trans.pred_sents[0], trans.attns[0].cpu())
            #break
        #break
    report_bleu(gold_list, pred_list)
    report_rouge(gold_list, pred_list)

    # if config.dump_beam:
    #     import json
    #     json.dump(translator.beam_accum,
    #               codecs.open(config.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()

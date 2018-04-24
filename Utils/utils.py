import os
import sys
import torch
import random
import math
import datetime
import numpy as np
from Utils.bleu import compute_bleu
from Utils.rouge import rouge

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def report_stats(stats, epoch, batch, n_batches, step_time, lr):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
        """
        sys.stderr.flush()
        sys.stderr.write((
            """Epoch {0:d},[{1:d}/{2:d}] Acc: {3:.2f}; PPL: {4:.2f}; Loss: {5:.2f}; CELoss: {6:.2f}, KLDLoss: {7:.2f} \r""").format(
                    epoch, batch, n_batches,
                    stats.accuracy(), stats.ppl(),
                    stats.loss, stats.loss_detail()[0], stats.loss_detail()[1]))
        sys.stderr.flush()


def debug_trace(*args, file=sys.stderr):
    print(datetime.datetime.now().strftime(
        '%Y/%m/%d %H:%M:%S'), '[DEBUG]', *args, file=file, flush=True)


def trace(*args, file=sys.stderr):
    print(datetime.datetime.now().strftime(
        '%Y/%m/%d %H:%M:%S'), *args, file=file, flush=True)

def check_save_path(path):
    save_path = os.path.abspath(path)
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def report_bleu(reference_corpus, translation_corpus):
   
    bleu, precions, bp, ratio, trans_length, ref_length =\
        compute_bleu([[x] for x in reference_corpus], translation_corpus)
    trace("BLEU: %.2f [%.2f/%.2f/%.2f/%.2f] Pred_len:%d, Ref_len:%d"%(
        bleu*100, *precions, trans_length, ref_length))


def report_rouge(reference_corpus, translation_corpus):
   
    scores = rouge([" ".join(x) for x in translation_corpus], 
            [" ".join(x) for x in reference_corpus])

     
    trace("ROUGE-1:%.2f, ROUGE-2:%.2f"%(
        scores["rouge_1/f_score"]*100, scores["rouge_2/f_score"]*100))
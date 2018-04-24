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

# def report_loss(t, num_epochs, lr, *loss, loss_names=["NLLLoss"], prefix="", file=sys.stderr):
    
#     sys.stderr.flush()
#     format_output = prefix + "[{0:d}/{1:d}], lr={2:.4f}"
#     i = 3
#     for l_name in loss_names:
#         format_output += ", %s={%d:.4f}"%(l_name, i)
#         i += 1
#     format_output += "        \r"
#     sys.stderr.write(
#         format_output.format(t, num_epochs, lr, *loss))
#     sys.stderr.flush()
#     file.write(format_output.format(t, num_epochs, lr, *loss))

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

def sort_in_batch(batch, lengths, descending=True, batch_first=False):                                                 
    """
    Sort the batch (N, T/J, D) given input lengths. Some models with RNN
    components require batching different length input by zero padding
    them to the same length., when the input have different lengths.
    torch.nn.Sequenceexpects the sequences to be ordered by length.
    We need to reorder the batch first for passage order and after that
    for question length order, feeded to the RNN and after that reorder
    again with the initial batch ordering to feed the correct combination
    of question and passage.
    """
    
    if batch_first:
        sorted_lengths, sorted_indices = torch.sort(
            lengths, dim=0, descending=descending)
        sorted_batch = batch[sorted_indices]
    else:
        sorted_lengths, sorted_indices = torch.sort(
            lengths, dim=1, descending=descending)
        sorted_batch = batch[:,sorted_indices,:]
    return sorted_batch, sorted_lengths, sorted_indices


def permute_tensor(tensor, indices, dim=0):
    """Given sorted_indices, accept Variables (tensor) to permute fast"""
    if dim == 0:
        permuted_tensor = tensor[indices]
        return permuted_tensor
    elif dim == 1:
        tensor = tensor.transpose(0, 1).contiguous()
        permuted_tensor = tensor[indices]
        return permuted_tensor.transpose(0, 1).contiguous()

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
"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn


from NMT.Models import NMTModel
from NMT.Models import VNMTModel
from NMT.Models import VRNMTModel
from NMT.Models import PackedRNNEncoder
from NMT.Models import RNNEncoder
from NMT.Models import InputFeedRNNDecoder
from NMT.Models import VarInputFeedRNNDecoder
from NMT.Modules import Embeddings
from torch.nn.init import xavier_uniform
from Utils.DataLoader import PAD_WORD
from Utils.utils import trace, aeq

def make_embeddings(vocab_size, embed_dim, dropout, padding_idx):
    """
    Make an Embeddings instance.
    Args:
        config: global configuration settings.
        vocab (Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    
    return Embeddings(embed_dim,
                      position_encoding=False,
                      dropout=dropout,
                      word_padding_idx=padding_idx,
                      word_vocab_size=vocab_size,
                      sparse=True)

def model_factory(config, src_vocab, trg_vocab, train_mode=True, checkpoint=None):
    
    # Make embedding.
    padding_idx = src_vocab.stoi[PAD_WORD]
    src_embeddings = make_embeddings(src_vocab.vocab_size, config.trg_embed_dim, config.dropout, padding_idx)
    
    padding_idx = trg_vocab.stoi[PAD_WORD]
    trg_embeddings = make_embeddings(trg_vocab.vocab_size, config.src_embed_dim, config.dropout, padding_idx)
    
    # Make NMT Model (= encoder + decoder).
    if config.system == "NMT":

        encoder = PackedRNNEncoder(
            config.rnn_type, 
            config.src_embed_dim, 
            config.hidden_size, config.enc_num_layers,
            config.dropout, config.bidirectional)
        decoder = InputFeedRNNDecoder(
            config.rnn_type, config.trg_embed_dim, config.hidden_size,
            config.dec_num_layers, config.attn_type,
            config.bidirectional, config.dropout)
        model = NMTModel(
            encoder, decoder, 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size, config)

    elif config.system == "VNMT":
        encoder = PackedRNNEncoder(
            config.rnn_type, 
            config.src_embed_dim, 
            config.hidden_size, 
            config.enc_num_layers,
            config.dropout, 
            config.bidirectional)
        
        decoder = InputFeedRNNDecoder(
            config.rnn_type, 
            config.trg_embed_dim+config.latent_size, 
            config.hidden_size, 
            config.dec_num_layers, 
            config.attn_type,
            config.bidirectional, 
            config.dropout)
        
        model = VNMTModel(
            encoder, decoder, 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size,  
            config)

    elif config.system == "VRNMT":
        encoder = PackedRNNEncoder(
            config.rnn_type, 
            config.src_embed_dim, 
            config.hidden_size, 
            config.enc_num_layers,
            config.dropout, 
            config.bidirectional)
        
        decoder = VarInputFeedRNNDecoder(
            config.rnn_type, 
            config.trg_embed_dim,
            config.latent_size, 
            config.hidden_size, 
            config.dec_num_layers, 
            config.attn_type,
            config.bidirectional, 
            config.dropout)
        
        model = VRNMTModel(
            encoder, decoder, 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size,  
            config)
    # if config.share_embeddings:
    #     generator[0].weight = decoder.embeddings.word_lut.weight

    if checkpoint is not None:
        trace('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        
    # Load the model states from checkpoint or initialize them.
    if train_mode and config.param_init != 0.0:
        trace("Initializing model parameters.")
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)
       
    

    # if hasattr(model.encoder, 'embeddings'):
    #     model.encoder.embeddings.load_pretrained_vectors(
    #                 config.pre_word_vecs_enc, config.fix_word_vecs_enc)
    # if hasattr(model.decoder, 'embeddings'):
    #     model.decoder.embeddings.load_pretrained_vectors(
    #                 config.pre_word_vecs_dec, config.fix_word_vecs_dec)

    if train_mode:
        model.train()
    else:
        model.eval()
    # Make the whole model leverage GPU if indicated to do so.
    #print(opt)
    if config.gpu_ids is not None:
        model.cuda()
    else:
        model.cpu()

    return model

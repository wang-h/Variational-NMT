import time
from argparse import ArgumentParser

def parse_args(script_name):
    """load arguments from command line. Default values are stored in config.ini"""
    parser = ArgumentParser()

    parse_base_args(parser)
    parse_data_args(parser)
    parse_gpu_args(parser)
    parse_network_args(parser)
    
    parse_embed_args(parser)   

    if script_name == "train":
        parse_optim_args(parser)
        parse_train_args(parser)
        
    elif script_name == "translate":
        parse_translate_args(parser)
        
    parse_loss_args(parser)
    parse_logging_args(parser)
    args = parser.parse_args()
    
    return args, parser


def parse_base_args(parser):
    parser.add_argument('--config', type=str, required=True,
                       help="path to datasets")
    parser.add_argument('--system', type=str,
                       help="use which NMT system")

def parse_data_args(parser):
    group = parser.add_argument_group('Data')
    group.add_argument('--data_path', type=str,
                       help="path to datasets")

    group.add_argument('--save_vocab', type=str,
                       help="path to vocab files")

    group.add_argument('--max_seq_len', type=int, default=50,
                        help="Maximum sequence length")

    group.add_argument('--trg_lang', type=str,
                         help="target language name suffix")

    group.add_argument('--src_lang', type=str,
                        help="source language name suffix")

def parse_gpu_args(parser):
    group = parser.add_argument_group('GPU')
    group.add_argument('--gpu_ids', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")
    group.add_argument('--use_cpu', default=False, action='store_true')

def parse_embed_args(parser):
    group = parser.add_argument_group('Embedding')
    group.add_argument('--min_freq', type=int, default=5,
                        help="min frequency for the prepared data")
    group.add_argument('--src_embed_dim', type=int, default=500,
                       help='Word embedding size for src.')
    group.add_argument('--trg_embed_dim', type=int, default=500,
                       help='Word embedding size for trg.')
    # group.add_argument('-feat_size', type=int, default=-1,
    #                    help="""If specified, feature embedding sizes
    #                    will be set to this. Otherwise, feat_vec_exponent
    #                    will be used.""")
    # group.add_argument('-feat_vec_exponent', type=float, default=0.7,
    #                    help="""If -feat_merge_size is not set, feature
    #                    embedding sizes will be set to N^feat_vec_exponent
    #                    where N is the number of values the feature takes.""")

    # parser.add_argument('--share_decoder_embeddings', action='store_true',
    #                    help="""Use a shared weight matrix for the input and
    #                    output word embeddings in the decoder.""")
    # group.add_argument('-share_embeddings', action='store_true',
    #                    help="""Share the word embeddings between encoder
    #                    and decoder. Need to use shared dictionary for this
    #                    option.""")
    # group.add_argument('-position_encoding', action='store_true',
    #                    help="""Use a sin to mark relative words positions.
    #                    Necessary for non-RNN style models.
    #                    """)
    # group.add_argument('--dynamic_dict', action='store_true',
    #                     help="Create dynamic dictionaries")
    # group.add_argument('--share_vocab', action='store_true',
    #                     help="Share source and target vocabulary")

def parse_train_args(parser):
    # Model loading/saving options

    group = parser.add_argument_group('train')

    group.add_argument('--save_model', default='output/model',
                       help="""Model filename validation perplexity""")

    group.add_argument('--checkpoint', default='', type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
   
    group.add_argument('--epochs', type=int, default=13,
                       help='Number of training epochs')

    group.add_argument('--batch_size', type=int, default=64,
                       help='Maximum batch size for training')

    group.add_argument('--valid_batch_size', type=int, default=32,
                       help='Maximum batch size for training')

    group.add_argument('--param_init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    
    group.add_argument('--param_init_glorot', action='store_true',
                       help="""Init parameters with xavier_uniform.
                       Required for transformer.""")

    
    


def parse_network_args(parser):
    group = parser.add_argument_group('Network')


    group.add_argument('--enc_num_layers', type=int, default=2,
                       help='Number of layers in the encoder')

    group.add_argument('--dec_num_layers', type=int, default=2,
                       help='Number of layers in the decoder')
    group.add_argument('--bidirectional', action='store_true',
                       help="""bidirectional encoding for encoder.""")

    group.add_argument('--hidden_size', type=int, default=500,
                       help='Size of rnn hidden states')

    group.add_argument('--latent_size', type=int, default=300,
                       help='Size of latent states')
    
    group.add_argument('--rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       help="""The gate type to use in the RNNs""")
   
    group.add_argument('--attn_type', type=str, default='general',
                       choices=['dot', 'general', 'mlp'],
                       help="""The attention type to use:
                       dotprod or general (Luong) or MLP (Bahdanau)""")
    group.add_argument('--meanpool', action="store_true",
                       help='use the mean pooling of encoder outputs for VAE')
    # group.add_argument('--use_target', action='store_true',
    #                    help="""use target for variational re-parameterization.""")
    group.add_argument('--dropout', type=float, default=0.3,
                       help="Dropout probability; applied in RNN stacks.")

def parse_loss_args(parser):
    # loss options.
    group = parser.add_argument_group('Loss Functions')
    group.add_argument("--kld_weight", default=0.05, type=float,
                        help="weight for the Kullback-Leibler divergence Loss in total loss score.")
    group.add_argument("--start_increase_kld_at", default=8, type=int,
                        help="start to increase KLD loss weight at .")


def parse_optim_args(parser):
    group = parser.add_argument_group('Optimization- Rate')
    # Optimization options
    group = parser.add_argument_group('Optimizer')
    
    
                       
    group.add_argument('--optim', default='sgd',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam',
                                'sparseadam'],
                       help="""Optimization method.""")
    
    group.add_argument('--adagrad_accum_init', type=float, default=0,
                       help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
    
    group.add_argument('--max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    
    
    group.add_argument('--lr', type=float, default=1.0,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('--lr_decay_rate', type=float, default=0.5,
                       help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past
                       start_decay_at""")
    group.add_argument('--start_decay_at', type=int, default=8,
                       help="""Start decaying every epoch after and including this
                       epoch""")

    group.add_argument('--decay_method', type=str, default="",
                       choices=['noam'], help="Use a custom decay rate.")

    group.add_argument('--warmup_steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")
    group.add_argument('--alpha', type=float, default=0.9,
                       help="""The alpha parameter used by RMSprop.""")
    group.add_argument('--eps', type=float, default=1e-8,
                       help="""The eps parameter used by RMSprop/Adam.""")

    group.add_argument('--weight_decay', type=float, default=0,
                       help="""The weight_decay parameter used by RMSprop.""")
    group.add_argument('--momentum', type=float, default=0,
                       help="""The momentum parameter used by RMSprop[0]/SGD[0.9].""")
    group.add_argument('--adam_beta1', type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add_argument('--adam_beta2', type=float, default=0.999,
                       help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Keras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
                       
def parse_translate_args(parser):

    group = parser.add_argument_group('Translator')
    group.add_argument('--output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")

    group.add_argument('--save_model', default='model',
                       help="""Pre-trained models""")

    group.add_argument('--beam_size',  type=int, default=5,
                       help='Beam size')

    group.add_argument('--max_length', type=int, default=100,
                       help='Maximum prediction length.')

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('--length_penalty', default='none',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('--coverage_penalty', default='none',
                       choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    group.add_argument('--alpha', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('--beta', type=float, default=-0.,
                       help="""Coverage penalty parameter""")

    group.add_argument('--block_ngram_repeat', type=int, default=0,
                       help='Block repetition of ngrams during decoding.')
    
    group.add_argument('--ignore_when_blocking', nargs='+', type=str,
                       default=[],
                       help="""Ignore these strings when blocking repeats.
                       You want to block sentence delimiters.""")

    group.add_argument('--replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group.add_argument('-k_best', type=int, default=1,
                       help="""If verbose is set, will output the k_best
                       decoded sentences""")

    

def parse_logging_args(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('--verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('--plot_attn', action="store_true",
                       help='Plot attention matrix for each pair')
    


    




  

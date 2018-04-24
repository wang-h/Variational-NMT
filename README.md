Variational Neural Machine Translation System 
==========

Implemented by Pytorch 0.4, some modules references to OpenNMT-py.




## References

1. Su, Jinsong, et al. "Variational Recurrent Neural Machine Translation." arXiv preprint arXiv:1801.05119 (2018).

2. Zhang, Biao, et al. "Variational neural machine translation." arXiv preprint arXiv:1605.07869 (2016)

## Differences

For Variational NMT, 
I did not use the mean-pooling for both sides (source and target).
I tested only using the last source hidden state is sufficient to achieve good performance.

For Variational Recurrent NMT, 
I tested only using the current RNN state is sufficient to achieve good performance.

The paper

`Yang, Zichao, et al. "Improved variational autoencoders for text modeling using dilated convolutions." arXiv preprint arXiv:1702.08139 (2017). `

explains the reason why use GRU instead of LSTM for building RNN cell, in general, VAE-LSTM-decoder performs worse than vanilla-LSTM-decoder.


## Usage
Training
    python train.py --config config/nmt.ini

Test
    python translate.py --config config/nmt.ini    
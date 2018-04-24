import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from matplotlib.font_manager import FontProperties
from Utils.DataLoader import BOS_WORD, EOS_WORD
JaFont = FontProperties(fname = '/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf')
ZhFont = FontProperties(fname = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')

def plot_attn(source, target, attn):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    
    ax.set_xticklabels(['']+[BOS_WORD]+source+[EOS_WORD], rotation=90 , fontproperties = ZhFont)
    ax.set_yticklabels(['']+target+[EOS_WORD], fontproperties = JaFont)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
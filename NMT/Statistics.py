import sys
import time
import math

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0, loss_dict={"CELoss": 0.0, "KLDLoss": 0.0}):
        self.loss = loss
        self.loss_dict = loss_dict
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        for key, val in stat.loss_dict.items():
            self.loss_dict[key] += val
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        if self.n_words == 0:
            return 0
        return 100 * (float(self.n_correct) / self.n_words)

    def loss_detail(self):
        return (self.loss_dict["CELoss"], self.loss_dict["KLDLoss"])
    
    def xent(self):
        if self.n_words == 0:
            return 0
        return self.loss / self.n_words

    def ppl(self):
        if self.n_words == 0:
            return 0
        return math.exp(min(float(self.loss)/self.n_words, 100))
        #return math.exp(min(self.loss, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_trgper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


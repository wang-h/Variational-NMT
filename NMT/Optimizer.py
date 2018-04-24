import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from Utils.utils import trace


class MultipleOptimizer(object):
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class Optimizer(object):
    def __init__(self, method, config):
            
        self.last_ppl = None
        self.lr = config.lr
        self.original_lr = config.lr
        self.max_grad_norm = config.max_grad_norm
        self.method = method
        self.lr_decay_rate = config.lr_decay_rate
        self.start_decay_at = config.start_decay_at
        self.start_decay = False
        self.alpha = config.alpha
        self._step = 0
        self.momentum = config.momentum
        self.betas = [config.adam_beta1, config.adam_beta2]
        self.adagrad_accum=config.adagrad_accum_init,
        self.decay_method=config.decay_method,
        self.warmup_steps=config.warmup_steps,
        self.model_size=config.hidden_size
        self.eps = config.eps

    def set_parameters(self, params):
        self.params = []
        #self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                # else:
                #     self.sparse_params.append(p)
        if self.method == 'SGD':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'Adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.optimizer.state[p]['sum'] = self.optimizer\
                        .state[p]['sum'].fill_(self.adagrad_accum)
        elif self.method == 'Adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'Adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9)
        elif self.method == 'RMSprop':
            # does not work properly.
            self.optimizer = optim.RMSprop(self.params, lr=self.lr, 
                alpha=self.alpha, eps=self.eps, weight_decay=self.lr_decay_rate, 
                momentum=self.momentum, centered=False)
    def _set_rate(self, lr):
        self.lr = lr
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.lr
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.lr

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_lr(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        # if self.last_ppl is not None and ppl > self.last_ppl:
        #     self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay_rate
            trace("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.lr

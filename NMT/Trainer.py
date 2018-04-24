import sys
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from NMT.Statistics import Statistics
from NMT.Models import NMTModel
from NMT.Models import VNMTModel
from NMT.Models import VRNMTModel
from Utils.utils import report_stats
class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model (NMT.Model.NMTModel): NMT base model

            loss_func (NMT.Loss.LossComputeBase): set loss func
            optimizer (NMT.Optimizer.Optimizer): optimizer
            config (Config): global configurations
    """

    def __init__(self, model, loss_func, optimizer, config):
        # Basic attributes.
        self.model = model
        self.loss_func = loss_func
        self.optim = optimizer
        self.config = config
        self.progress_step = 0

    def train(self, train_iter, epoch, num_batches):
        """ Train next epoch.
        Args:
            train_iter (BatchDataIterator): training data iterator
            epoch (int): the epoch number
            num_batches (int): the batch number
        Returns:
            stats (Statistics): epoch loss statistics
        """
        self.model.train()
       
        total_stats = Statistics()
        self.loss_func.kld_weight_step(epoch, self.config.start_increase_kld_at)
        for idx, batch in enumerate(train_iter):
            self.model.zero_grad()
            src, src_lengths = batch.src, batch.src_Ls
            trg, trg_lengths = batch.trg, batch.trg_Ls
            ref = batch.trg[1:]
            kld_loss = 0.
            normalization = batch.batch_size
            if isinstance(self.model, VRNMTModel):
                outputs, _, _, kld_loss = self.model(
                            src, src_lengths, trg)
            elif isinstance(self.model, VNMTModel):
                outputs, _, _, kld_loss = self.model(
                            src, src_lengths, trg)
            elif isinstance(self.model, NMTModel):
                outputs, _, _ = self.model(
                        src, src_lengths, trg)
            
            probs = self.model.generator(
                outputs.view(-1, outputs.size(2)))
                
            loss, batch_stats = self.loss_func.compute_batch_loss(
                        probs, ref, normalization, kld_loss=kld_loss)
            
            loss.backward()
            # 4. Update the parameters and statistics.
            self.optim.step()

            del loss, outputs, probs

            
            report_stats(
                batch_stats, epoch, idx, num_batches, 
                self.progress_step, self.optim.lr)
            
            total_stats.update(batch_stats)
            self.progress_step += 1
        
        
        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        
        self.model.eval()
        total_stats = Statistics()
        self.model.zero_grad()
        for batch in valid_iter:
            kld_loss = 0.
            normalization = batch.batch_size
            
            src, src_lengths = batch.src, batch.src_Ls
            trg, trg_lengths = batch.trg, batch.trg_Ls
            ref = batch.trg[1:]
            # F-prop through the model.
            if isinstance(self.model, VRNMTModel):
                outputs, _, _, kld_loss = self.model(
                            src, src_lengths, trg)
            elif isinstance(self.model, VNMTModel):
                outputs, _, _, kld_loss = self.model(
                            src, src_lengths, trg)
            elif isinstance(self.model, NMTModel):
                outputs, _, _ = self.model(
                        src, src_lengths, trg)

            probs = self.model.generator(outputs.view(-1, outputs.size(2)))
            loss, batch_stats = self.loss_func.compute_batch_loss(
                         probs, ref, normalization, kld_loss=kld_loss)
            
            # # Update statistics.
            total_stats.update(batch_stats)
            del outputs, probs, batch_stats, loss
            # # Set model back to training mode.
        return total_stats

    def lr_step(self, ppl, epoch):
        return self.optim.update_lr(ppl, epoch)

    def dump_checkpoint(self, epoch, config, valid_stats):
        """ 
        Save a checkpoint.

        Args:
            epoch (int): epoch number
            config (Config): global configurations
            valid_stats : statistics of last validation run
        """

        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()

        model_state_dict = {k: v for k, v in model_state_dict.items()}


        checkpoint = {
            'model': model_state_dict,
            'config': config,
            'epoch': epoch,
            'optim': self.optim,
        }
        # torch.save(checkpoint,
        #            '%s_acc_%.2f_loss_%.2f_e%d.pt'
        #            % (config.save_model, valid_stats.accuracy(),
        #               valid_stats.loss, epoch))
        
        # torch.save(checkpoint,
        #            '%s_acc_%.2f_loss_%.2f_e%d.pt'
        #            % (config.save_model, valid_stats.accuracy(),
        #               valid_stats.loss, epoch))
        if epoch == config.epochs:
            torch.save(checkpoint, '%s.pt'% config.save_model)

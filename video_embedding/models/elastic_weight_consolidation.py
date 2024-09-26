import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader


class ElasticWeightConsolidation:
    def __init__(self):
        super(ElasticWeightConsolidation, self).__init__()
        self.lambda_ = 0.4 # bigger means stiffer model, see https://petrvancjr.github.io/notes/

    def compute_fisher_information(self):
        fisher_information = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        self.model.eval()

        for data in self.dataloader:
            inputs = data[0]

            self.model.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, inputs)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad.detach() ** 2

        for name in fisher_information:
            fisher_information[name] /= len(self.dataloader)
        return fisher_information
    

    def register_ewc_params(self):
        """Is called after each training of model
        """        
        for name, param in self.model.named_parameters():
            self.model.register_buffer(f"old_params_{name.replace('.', '_')}", param.data.detach().clone())

        fisher_information = self.compute_fisher_information()
        for name in fisher_information:
            self.model.register_buffer(f"fisher_{name.replace('.', '_')}", fisher_information[name].detach().clone())

    def ewc_loss(self):
        """Is called during training - then `final_loss = ewc_loss + loss`
        """        
        try:
            loss = 0
            for name, param in self.model.named_parameters():
                old_param = getattr(self.model, f"old_params_{name.replace('.', '_')}")
                fisher = getattr(self.model, f"fisher_{name.replace('.', '_')}")

                loss += (fisher * (param - old_param) ** 2).sum()
            return (self.lambda_ / 2) * loss
        except AttributeError: # ewc parameter is not registered, so it is initial training
            # print("EWC LOSS NOT ENABLED")
            return 0 # no loss at initial training



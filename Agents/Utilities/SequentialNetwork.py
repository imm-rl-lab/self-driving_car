import numpy as np
import torch
from torch import nn


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class SequentialNetwork(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation=None, init_w=3e-3):
        super().__init__()
        hidden_layers = layers[:-1]
        network = [nn.Sequential(nn.Linear(i, o), hidden_activation) 
                   for i, o in zip(hidden_layers, hidden_layers[1:])]
        
        if output_activation:
            network.append(nn.Sequential(nn.Linear(layers[-2], layers[-1]), output_activation))
        else:
            network.append(nn.Sequential(nn.Linear(layers[-2], layers[-1])))

        self.network = nn.Sequential(*network)
        self.init_weights(init_w)
        #self.apply(self._init_weights_)

    def forward(self, tensor):
        return self.network(tensor)

    def init_weights(self, init_w):
        layers_n = len(self.network)
        for i in range(layers_n - 2):
            self.network[i][0].weight.data = fanin_init(self.network[i][0].weight.data.size())
        self.network[layers_n - 1][0].weight.data.uniform_(-init_w, init_w)
    
    #def _init_weights_(self, m):
    #    if type(m) == nn.Linear:
    #        torch.nn.init.xavier_normal_(m.weight)
    #        m.bias.data.fill_(0.01)

    
class Seq_Network(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation=None):
        super().__init__()
        hidden_layers = layers[:-1]
        network = [nn.Sequential(nn.Linear(i, o), hidden_activation) for i, o in
                   zip(hidden_layers, hidden_layers[1:])]
        network.append(nn.Linear(layers[-2], layers[-1]))
        if output_activation:
            network.append(output_activation)
        self.network = nn.Sequential(*network)
        self.apply(self._init_weights_)

    def forward(self, tensor):
        return self.network(tensor)

    @staticmethod
    def _init_weights_(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
import torch.nn as nn
import torch.nn.functional as F
import torch
from Params import Params

class TokNet(nn.Module):
    
    def __init__(self, alphabet_size, sentences_max_length, output_syms ):
        super(TokNet, self).__init__()
        
        self.vocabulary_size = alphabet_size
        self.sentences_max_length = sentences_max_length
        self.alphabet_size = alphabet_size
        self.output_syms = output_syms
        
        self.w1 = 8
        self.w2 = 8
        
     
        self.fc1 = nn.Linear(self.alphabet_size * Params.monograms, self.w1)
        self.fc2 = nn.Linear(self.w1, self.w2)
        self.fc3 = nn.Linear(self.w2, self.output_syms)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        #in the dev: 577530 chars
        #B: 0.186437
        #I: 0.66035
        #S: 0.153528
        #cross_entropy_weights = torch.Tensor([0.60, 0.10, 0.30])
        #self.loss_function = nn.CrossEntropyLoss(weight=cross_entropy_weights)
        self.loss_function = nn.CrossEntropyLoss()
        #self.loss_function = nn.MSELoss()


    def forward(self, x):
        inp = self.relu(self.fc1(x))
        hid1 = self.relu(self.fc2(inp))
        o = (self.fc3(hid1))

        return o
import torch.nn as nn
import torch.nn.functional as F
import torch


class TokNet(nn.Module):
    
    def __init__(self, alphabet_size, sentences_max_length, output_syms ):
        super(TokNet, self).__init__()
        
        self.vocabulary_size = alphabet_size
        self.sentences_max_length = sentences_max_length
        self.alphabet_size = alphabet_size
        self.output_syms = output_syms
        
        self.w1 = 64
        self.w2 = 64
        
     
        self.fc1 = nn.Linear(self.alphabet_size, self.w1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.w1, self.w2)
        self.fc3 = nn.Linear(self.w2, self.output_syms)
        #self.ls = nn.Softmax(dim=1)
        #in the dev: 577530 chars
        #B: 0.186437
        #I: 0.66035
        #S: 0.153528
        cross_entropy_weights = torch.Tensor([0.60, 0.45, 0.10])
        #self.loss_function = nn.CrossEntropyLoss(weight=cross_entropy_weights)
        self.loss_function = nn.MSELoss()


    def forward(self, x):
        inp = self.relu(self.fc1(x))
        hid = self.relu(self.fc2(inp))
        o = self.fc3(hid)
        
        return o
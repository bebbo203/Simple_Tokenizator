import torch.nn as nn
import torch.nn.functional as F


class TokNet(nn.Module):
    
    def __init__(self, alphabet_size, sentences_max_length, output_syms ):
        super(TokNet, self).__init__()
        
        self.vocabulary_size = alphabet_size
        self.sentences_max_length = sentences_max_length
        self.alphabet_size = alphabet_size
        self.output_syms = output_syms
        
        
     
        self.fc1 = nn.Linear(self.alphabet_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.output_syms)
        #self.ls = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        


    def forward(self, x):
        inp = self.relu(self.fc1(x))
        hid = self.relu(self.fc2(inp))
        o = self.fc3(hid)
        
        return o
        

        '''
        l1 = self.linear1(input_idx)  # compute the embeddings for the input words
        l2 = F.relu(self.linear2(l1))
        #l3 = F.relu(self.linear3(l2))
        o = self.output(l2).view(self.sentences_max_length, self.output_syms).clone().detach().requires_grad_(True)
        #o = F.softmax(self.output(l2).view(self.sentences_max_length, self.output_syms), dim=1)
        o = F.softmax(o, dim=1)
        '''
        
        
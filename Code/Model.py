import torch.nn as nn
import torch.nn.functional as F


class TokNet(nn.Module):
    
    def __init__(self, alphabet_size, sentences_max_length, output_syms ):
        super(TokNet, self).__init__()
        
        self.vocabulary_size = alphabet_size
        self.sentences_max_length = sentences_max_length
        self.alphabet_size = alphabet_size
        self.output_syms = output_syms
        self.loss_function = nn.MSELoss()
        
     

        
        self.linear1 = nn.Linear(self.alphabet_size * self.sentences_max_length, 512)
        self.linear2 = nn.Linear(512, 512)
        #self.linear3 = nn.Linear(150, 150)
        self.output = nn.Linear(512, self.sentences_max_length * self.output_syms)


    def forward(self, input_idx):
        # This method defines the outputs of a forward pass on the model
        l1 = self.linear1(input_idx)  # compute the embeddings for the input words
        l2 = self.linear2(l1)
        l2 = F.relu(l2)
        #l3 = self.linear3(l2)
        #l3 = F.relu(l3)
        o = self.output(l2)
        
        
        # CrossEntropyLoss applies log_softmax internally
        return o
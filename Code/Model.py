import torch.nn as nn
import torch.nn.functional as F


class TokNet(nn.Module):
    
    def __init__(self, alphabet_size, sentences_max_length, output_syms ):
        super(TokNet, self).__init__()
        
        self.vocabulary_size = alphabet_size
        self.sentences_max_length = sentences_max_length
        self.alphabet_size = alphabet_size
        self.output_syms = output_syms
        self.loss_function = nn.CrossEntropyLoss()
        
     

        
        self.linear1 = nn.Linear(self.alphabet_size * self.sentences_max_length, 128)
        self.linear2 = nn.Linear(128, 128)
        #self.linear3 = nn.Linear(10, 10)
        self.output = nn.Linear(128, self.output_syms * self.sentences_max_length)


    def forward(self, input_idx):
        # This method defines the outputs of a forward pass on the model
        l1 = self.linear1(input_idx)  # compute the embeddings for the input words
        l2 = F.relu(self.linear2(l1))
        #l3 = F.relu(self.linear3(l2))
        #Controlla che la softmax sia sui 4 elementi e non su tutta la frase
        o = F.softmax(self.output(l2).view(self.sentences_max_length, self.output_syms), dim=1)

        return o
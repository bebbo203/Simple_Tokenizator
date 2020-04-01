from DataGenerator import DataGenerator
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os

torch.manual_seed(42)

class Trainer():

    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train(self, train_generator, labels_generator, epochs = 1, sentences_max_length = 0):
        train_loss = 0.0

        
        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0
            self.model = self.model.float()
            for step, data in tqdm(enumerate(zip(train_generator, labels_generator)), desc="Batch", leave=False):
                x = data[0].float()
                y = data[1].long()
                output = self.model(x)
                
                #OCCHIO CHE QUA NON QUATTRO MA TRE
                y = torch.tensor([elem.argmax() for elem in y.view(-1, labels_generator.get_dictionary_size())])
                y = y.view(y.size(0), 1).float()

                output = torch.tensor([elem.argmax().float() for elem in output.view(-1, labels_generator.get_dictionary_size())], requires_grad=True).view(train_generator.get_sentences_max_length(), 1)

                loss = self.model.loss_function(output, y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                len_train += 1
                epoch_loss += loss.item()
                
            
            avg_epoch_loss = epoch_loss / len_train
            print('Epoch: {} avg loss = {:0.4f}'.format(epoch, avg_epoch_loss))
            train_loss += avg_epoch_loss
            torch.save(self.model.state_dict(),
                       os.path.join("./models", 'state_{}.pt'.format(epoch)))
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    
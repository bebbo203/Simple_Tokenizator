from DataGenerator import DataGenerator
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import time

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
            start_time = time.time()
            self.model = self.model.float()
            for step, data in tqdm(enumerate(zip(train_generator, labels_generator)), desc="Batch", leave=False):
                x = data[0].float()
                y = data[1].long()
                
                self.optimizer.zero_grad() 
                output = self.model(x)

                #print(output)
                #print("END")
                
                y = torch.tensor([elem.argmax() for elem in y.view(-1, labels_generator.get_dictionary_size())])
                y = y.view(-1).long()
               
                
                
                loss = self.model.loss_function(output.view(sentences_max_length, labels_generator.get_dictionary_size()), y)
                loss.backward()

                self.optimizer.step()
                       

                len_train += 1
                epoch_loss += loss.item()
                
            
            avg_epoch_loss = epoch_loss / len_train
            print('Epoch:{} avg_loss = {:0.4f} time_elapsed = {:.2f}s'.format(epoch, avg_epoch_loss, time.time() - start_time))
            train_loss += avg_epoch_loss
            torch.save(self.model.state_dict(),
                       os.path.join("./models", 'state_{}.pt'.format(epoch)))

            train_generator.reset()
            labels_generator.reset()
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    
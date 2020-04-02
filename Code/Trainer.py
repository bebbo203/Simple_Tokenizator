from DataGenerator import DataGenerator
from BatchWrapper import BatchWrapper
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import time
import Evaluator


torch.manual_seed(42)

class Trainer():

    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train(self, dataloader, epochs = 1, sentences_max_length = 0):
        train_loss = 0.0
        self.model = self.model.float()
        
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0
            start_time = time.time()
            
            for step, data in tqdm(enumerate(dataloader), desc="Batch", leave=False):
                x = data['inputs'].float()
                y = data['targets'].long()
                
                
                output = self.model(x)

                #print(output)
                #print("END")
                
                y = torch.tensor([elem.argmax() for elem in y.view(-1, dataloader.dataset.labels_gen.get_dictionary_size())])
                y = y.view(-1).long()
              
                
                loss = self.model.loss_function(output, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 
                       

                len_train += 1
                epoch_loss += loss.item()
                
            
            avg_epoch_loss = epoch_loss / len_train

            
            
            
            train_loss += avg_epoch_loss
            torch.save(self.model.state_dict(),
                       os.path.join("./models", 'state_{}.pt'.format(epoch)))
            dataloader.dataset.train_gen.reset()
            dataloader.dataset.labels_gen.reset()
            
            print('Epoch:{} avg_loss = {:0.4f} time_elapsed = {:.2f}s'.format(epoch, avg_epoch_loss, time.time() - start_time))

            
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    
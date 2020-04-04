from DataGenerator import DataGenerator
import torch
from Params import Params
from BatchWrapper import BatchWrapper

class BalancedDataset(torch.utils.data.IterableDataset):

    def __init__(self, train_data_path, labels_data_path, monograms=Params.monograms, batch_size = 1):
        self.train_data_path = train_data_path
        self.labels_data_path = labels_data_path
        
        self.dataset = BatchWrapper(Params.train_dataset_path, Params.labels_dataset_path, monograms=monograms) 

        self.batch_size = batch_size

    def __iter__(self):

        
            eq = [0,0,0]
            limit = self.batch_size / 3
            out_text = []
            out_labels = []
            ret = False
            
            for elem in self.dataset:
                if(ret):
                    out_text = []
                    out_labels = []
                    ret = False

                if ((eq[0]+eq[1]+eq[2]) < self.batch_size):
                    text = elem["inputs"]
                    labels = elem["targets"]
                    
                    class_ = labels.argmax()
                    if(eq[class_] < limit):
                        out_text.append(text)
                        out_labels.append(labels)
                        eq[class_] += 1
                    if((eq[0]+eq[1]+eq[2]) == self.batch_size):
                        out_text = torch.stack(out_text)
                        out_labels = torch.stack(out_labels)
                        eq = [0,0,0]
                        ret = True
                        yield {"inputs": out_text, "targets": out_labels}
                        
                    
     
        

            

           
    
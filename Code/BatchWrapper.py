from DataGenerator import DataGenerator
import torch

class BatchWrapper(torch.utils.data.IterableDataset):

    def __init__(self, train_data_path, labels_data_path):
        self.train_data_path = train_data_path
        self.labels_data_path = labels_data_path

        self.train_gen = DataGenerator(self.train_data_path, 1, tensor=True, monograms=True)
        self.labels_gen = DataGenerator(self.labels_data_path, 1, tensor=True, monograms=True)

     

        

    def __iter__(self):
        for text, labels in zip(self.train_gen, self.labels_gen):
            output_dict = {'inputs': text, 'targets':labels}
            yield output_dict
    
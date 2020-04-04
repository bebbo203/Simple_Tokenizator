from DataGenerator import DataGenerator
import torch
from Params import Params

class BatchWrapper(torch.utils.data.IterableDataset):

    def __init__(self, train_data_path, labels_data_path, monograms=1):
        self.train_data_path = train_data_path
        self.labels_data_path = labels_data_path

        self.train_gen = DataGenerator(self.train_data_path, Params.sentences_max_length, tensor=True, monograms=monograms)
        self.labels_gen = DataGenerator(self.labels_data_path, Params.sentences_max_length, tensor=True, monograms=1)

    def __iter__(self):
        for text, labels in zip(self.train_gen, self.labels_gen):
            if(float(torch.rand(1)) < Params.prob_shuffle):
                continue
            output_dict = {'inputs': text, 'targets':labels}
            yield output_dict
    
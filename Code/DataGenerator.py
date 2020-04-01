import torch.nn as nn
import torch
import codecs


class DataGenerator(torch.utils.data.IterableDataset):

    def __init__(self, path, sentences_max_length, tensor=True):
        self.file = self.open_file(path)
        self.sentences_max_length_computed = self.longest_sentence_size()
        self.tensor = tensor
        
        if(sentences_max_length == 0):
            self.sentences_max_length = self.sentences_max_length_computed
        else:
            self.sentences_max_length = sentences_max_length
        
        self.chars_dict = self.create_chars_dict()
        
    
    def __iter__(self):
        #n_words_in_sentence * one_hot_encoding_dim 
        
        if(self.tensor == False):         
            for _line in self.file.readlines():
                yield _line
        else:   
            for _line in self.file.readlines():
                line = _line
                for start in range(0, len(line), self.sentences_max_length):
                    reduced = line[start:start+self.sentences_max_length]
                    out = torch.zeros(self.sentences_max_length, len(self.chars_dict))
                
                    for i,char in enumerate(reduced):
                        if(char != "\n"):
                            out[i][self.chars_dict[char]] = 1
                    yield out.flatten().view(1, self.sentences_max_length * len(self.chars_dict))

             

    def open_file(self, path):
        file = open(path, "r", encoding="utf-8")
        return file

    def longest_sentence_size(self):
        M = 0
        for line in self.file.readlines():
            M += len(line)
        self.file.seek(0)
        return M

    def avg_sentence_size(self):
        M = 0
        i=0
        for line in self.file.readlines():
            M += len(line)
            i += 1
        self.file.seek(0)
        return int(M/i)
    
    def get_dictionary_size(self):
        return len(self.chars_dict)

    def get_sentences_max_length(self):
        return self.sentences_max_length
    
    def create_chars_dict(self):
        ret_dict = {}
        for sentence in self.file.readlines():
            for word in sentence:
                for char in word:
                    if(char not in ret_dict and char != '\n'):
                        ret_dict.update({char: len(ret_dict)})
        self.file.seek(0)
        return ret_dict

    
    
            

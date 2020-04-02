from DataGenerator import DataGenerator
import sys
from Trainer import Trainer
from Model import TokNet
import torch
import torch.nn as nn
import os

train_dataset_path = "../Dataset/en.wiki.sentences.dev"
labels_dataset_path = "../Dataset/en.wiki.gold.dev"
validation_dataset_path = "../Dataset/en.wiki.gold.dev"



sentences_max_length = 1

train_generator = DataGenerator(train_dataset_path, sentences_max_length, tensor=True, monograms=True)
eval_generator = DataGenerator(labels_dataset_path, sentences_max_length, tensor=True, monograms=True)
alphabet_size = train_generator.get_dictionary_size()
output_syms = eval_generator.get_dictionary_size()

chars_dict = eval_generator.create_chars_dict()
model = TokNet(alphabet_size, sentences_max_length, output_syms)

file_name = "state_0.pt"
if(len(sys.argv) > 1):
    file_name = "state_"+str(sys.argv[1])+".pt"

model.load_state_dict(torch.load(os.path.join("./models",file_name)))
model.eval()



n_int = 0
OK = 0
out_dict = {'0': 0, '1': 0, '2': 0, "3": 0}

for i in zip(train_generator, eval_generator):
    x = i[0]
    o = model(x)
    y = i[1].view(train_generator.sentences_max_length, eval_generator.get_dictionary_size())

    o = o.view(train_generator.sentences_max_length, eval_generator.get_dictionary_size())
    

    for elem in zip(o, y):
        n_int += 1
        val = elem[0].argmax()
        out_dict.update({str(int(val)): out_dict[str(int(val))]+1})
        if val == elem[1].argmax():
            OK+=1
        
    
    if(n_int >= 100000):
        break
  
print(n_int)
print(out_dict)
print("Accuracy: " + str(OK/n_int))

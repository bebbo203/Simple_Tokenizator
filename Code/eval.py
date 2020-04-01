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

alphabet_size = 179
sentences_max_length = 300
output_syms = 3

train_generator = DataGenerator(train_dataset_path, sentences_max_length)
eval_generator = DataGenerator(labels_dataset_path, sentences_max_length, tensor=False)
chars_dict = eval_generator.create_chars_dict()
model = TokNet(alphabet_size, sentences_max_length, output_syms)
model.load_state_dict(torch.load(os.path.join("./models", 'state_0.pt')))
model.eval()

t_it = train_generator.__iter__()
e_it = eval_generator.__iter__()


n_int = 0
OK = 0
for i in range(1):
    sentence = next(t_it)
    out = model(sentence)
    y = torch.tensor([elem.argmax() for elem in out.view(-1, 3)])
    gold = str(next(e_it))
    



    s = ""
    for n in y:
        s += list(chars_dict.keys())[list(chars_dict.values()).index(int(n))]


    
    for i in range(min(len(s), len(gold))):
        n_int += 1
        if( s[i] == gold[i] ):
            OK += 1
    
    print(s)
    print(gold)


print("Accuracy: " + str(OK/n_int))

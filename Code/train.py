from DataGenerator import DataGenerator
import sys
from Trainer import Trainer
from Model import TokNet
import torch
import torch.nn as nn

train_dataset_path = "../Dataset/en.wiki.sentences.dev"
labels_dataset_path = "../Dataset/en.wiki.gold.dev"
validation_dataset_path = "../Dataset/en.wiki.gold.test"

sentences_max_length = 300
train_generator = DataGenerator(train_dataset_path, sentences_max_length, tensor=True)
labels_generator = DataGenerator(labels_dataset_path, sentences_max_length, tensor=True)

#Parametrizzalo!
alphabet_size = len(train_generator.chars_dict)
output_syms = len(labels_generator.chars_dict)



print("Alphabet size: %d \nOutput_syms: %d\nSentences_max_length: %d" % (alphabet_size, output_syms, sentences_max_length))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TokNet(alphabet_size, sentences_max_length, output_syms)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)




t = Trainer(model, optimizer, device)
t.train(train_generator, labels_generator, epochs=1, sentences_max_length=sentences_max_length)



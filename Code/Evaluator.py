from DataGenerator import DataGenerator
import sys
from Model import TokNet
import torch
import torch.nn as nn
import os



def evaluate(epoch_number, train_dataset_path, labels_dataset_path, sentences_max_length):
    train_generator = DataGenerator(train_dataset_path, sentences_max_length, tensor=True)
    eval_generator = DataGenerator(labels_dataset_path, sentences_max_length, tensor=False, labels=True)
    alphabet_size = train_generator.get_dictionary_size()
    output_syms = eval_generator.get_dictionary_size()

    chars_dict = eval_generator.create_chars_dict()
    model = TokNet(alphabet_size, sentences_max_length, output_syms)

    file_name = "state_{}.pt".format(epoch_number)

    model.load_state_dict(torch.load(os.path.join("./models", file_name)))
    model.eval()

    t_it = train_generator.__iter__()
    e_it = eval_generator.__iter__()


    n_int = 0
    OK = 0
    for i in range(1000):
        sentence = next(t_it)
        out = model(sentence)
        y = torch.tensor([elem.argmax() for elem in out.view(-1, eval_generator.get_dictionary_size())])
        gold = str(next(e_it))

        s = ""
        for n in y:
            s += list(chars_dict.keys())[list(chars_dict.values()).index(int(n))]
        for i in range(min(len(s), len(gold))):
            n_int += 1
            if( s[i] == gold[i] ):
                OK += 1
    
    return OK/n_int
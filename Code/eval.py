from DataGenerator import DataGenerator
import sys
from Trainer import Trainer
from Model import TokNet
import torch
import torch.nn as nn
import os
from Params import Params

train_dataset_path = "../Dataset/en.wiki.sentences.dev"
labels_dataset_path = "../Dataset/en.wiki.gold.dev"
validation_dataset_path = "../Dataset/en.wiki.gold.dev"

def get_key(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    
    return  listOfKeys




train_generator = DataGenerator(train_dataset_path, Params.sentences_max_length, tensor=True, monograms=Params.monograms)
eval_generator = DataGenerator(labels_dataset_path, Params.sentences_max_length, tensor=True, monograms=1)

alphabet_size = train_generator.get_dictionary_size()
output_syms = eval_generator.get_dictionary_size()

chars_dict = eval_generator.create_chars_dict()
model = TokNet(alphabet_size, Params.sentences_max_length, output_syms)

file_name = "state_0.pt"
write = 0
if(len(sys.argv) == 2):
    file_name = "state_"+str(sys.argv[1])+".pt"
if(len(sys.argv) == 3):
    file_name = "state_"+str(sys.argv[1])+".pt"
    write = int(sys.argv[2])
    

model.load_state_dict(torch.load(os.path.join("./models",file_name)))
model.eval()



n_int = 0
OK = 0
out_dict_pred = {'0': 0, '1': 0, '2':0}
out_dict_gold = {'0': 0, '1': 0, '2':0}

s_pred = ""
s_gold = ""

conf = [[0,0,0],[0,0,0],[0,0,0]]
for i in zip(train_generator, eval_generator):
    

    x = i[0].view(1, alphabet_size * Params.monograms)
    o = model(x)
    y = i[1].view(train_generator.sentences_max_length, eval_generator.get_dictionary_size())
    
    

    o = o.view(train_generator.sentences_max_length, eval_generator.get_dictionary_size())
    
    if(write):
        pred = get_key(chars_dict, int(o.argmax()))[0]
        c = get_key(chars_dict, int(y.argmax()))[0]


        s_pred += str(pred)
        s_gold += str(c)

        
        if(len(s_pred) > 64):
            print("+++++++++++++++++++")
            print("PRED: " + str(s_pred))
            print("GOLD: " + str(s_gold))
            s_pred = ""
            s_gold = ""
        

    for elem in zip(o, y):
        n_int += 1
        val = elem[0].argmax()
        out_dict_pred.update({str(int(val)): out_dict_pred[str(int(val))]+1})
        out_dict_gold.update({str(int(elem[1].argmax())): out_dict_gold[str(int(elem[1].argmax()))]+1})
        conf[int(val)][int(elem[1].argmax())] += 1
        if val == elem[1].argmax():
            OK+=1
        
    
    if(n_int >= 100000):
        break
  

print("PRED: " + str(out_dict_pred))
print("GOLD: " + str(out_dict_gold))
print("##################################")
for elem in conf:
    print("%5d %5d %5d" % (elem[0],elem[1], elem[2]))
print("##################################")
print("Accuracy: " + str(OK/n_int))

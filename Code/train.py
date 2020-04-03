from DataGenerator import DataGenerator
from BatchWrapper import BatchWrapper
import sys
from Trainer import Trainer
from Model import TokNet
import torch
from Params import Params

def get_key(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    
    if(len(listOfKeys) == 0):
        print("CASINOOOOOOOOOOOOOOO")
    return  listOfKeys






bw = BatchWrapper(Params.train_dataset_path, Params.labels_dataset_path)

alphabet_size = len(bw.train_gen.chars_dict)
output_syms = len(bw.labels_gen.chars_dict)





print("Alphabet size: %d \nOutput_syms: %d\nSentences_max_length: %d" % (alphabet_size, output_syms, sentences_max_length))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TokNet(alphabet_size, Params.sentences_max_length, output_syms)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)



dataloader = torch.utils.data.DataLoader(bw, batch_size=Params.batch_size)



t = Trainer(model, optimizer, device)
t.train(dataloader, epochs=10, sentences_max_length=sentences_max_length)



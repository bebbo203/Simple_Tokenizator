from DataGenerator import DataGenerator
from BatchWrapper import BatchWrapper
import sys
from Trainer import Trainer
from Model import TokNet
import torch

def get_key(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    
    return  listOfKeys

torch.set_default_tensor_type('torch.cuda.FloatTensor')
train_dataset_path = "../Dataset/en.wiki.sentences.dev"
#train_dataset_path = "../Dataset/test_bigram.txt"
labels_dataset_path = "../Dataset/en.wiki.gold.dev"
validation_dataset_path = "../Dataset/en.wiki.gold.test"

sentences_max_length = 32
#train_generator = DataGenerator(train_dataset_path, sentences_max_length, tensor=True)
#labels_generator = DataGenerator(labels_dataset_path, sentences_max_length, tensor=True, labels=True)

#print(train_generator.frequencies_dict())

'''

for i in zip(train_generator, labels_generator):
    t = i[0]
    l = i[1]

    t = t.view(sentences_max_length, train_generator.get_dictionary_size())
    l = l.view(sentences_max_length, labels_generator.get_dictionary_size())
    for i in zip(t, l):
        #print(i[0].argmax())
        print(get_key(train_generator.chars_dict, i[0].argmax()))
        print(get_key(labels_generator.chars_dict, i[1].argmax()))

'''


bw = BatchWrapper(train_dataset_path, labels_dataset_path, monograms=True)

alphabet_size = len(bw.train_gen.chars_dict)
output_syms = len(bw.labels_gen.chars_dict)





print("Alphabet size: %d \nOutput_syms: %d\nSentences_max_length: %d" % (alphabet_size, output_syms, sentences_max_length))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TokNet(alphabet_size, sentences_max_length, output_syms)
model.cuda()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


dataloader = torch.utils.data.DataLoader(bw, batch_size=256) 


'''
for a in dataloader:
    print(get_key(bw.train_gen.chars_dict, a['inputs'].argmax()))
    print(get_key(bw.labels_gen.chars_dict, a['targets'].argmax()))
'''

t = Trainer(model, optimizer, device)
t.train(dataloader, epochs=100, sentences_max_length=sentences_max_length)



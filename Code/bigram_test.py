from DataGenerator import DataGenerator
import sys
from Trainer import Trainer
from Model import TokNet
import torch
import torch.nn as nn

train_dataset_path = "../Dataset/en.wiki.gold.dev"

dg = DataGenerator(train_dataset_path, 64, tensor=True, labels=False)

print(dg.chars_dict)

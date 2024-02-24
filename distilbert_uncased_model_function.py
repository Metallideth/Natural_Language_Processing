# Inspired by code at https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=zHxRRzqpBf76

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import pickle
from netskope_dataloader import NetSkopeDataset

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier_function = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier_function = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        # Take only the hidden state corresponding to the first token, CLS, as a representation for the entire sequence
        pooler = hidden_state[:, 0]
        function = self.pre_classifier_function(pooler)
        function = torch.nn.ReLU()(function)
        function = self.dropout(function)
        function = self.classifier_function(function)
        return function
    
if __name__ == '__main__':
    model = DistilBERTClass()
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    ROLE_DIMENSION = 6
    FUNCTION_DIMENSION = 5
    LEVEL_DIMENSION = 6
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

    with open('Data/index_label_mapping.pkl','rb') as file:
        index_label_mapping = pickle.load(file)

    train = NetSkopeDataset('Data/train.pkl',tokenizer,MAX_LEN)
    val = NetSkopeDataset('Data/val.pkl',tokenizer,MAX_LEN)

    train_params = {
        'batch_size':TRAIN_BATCH_SIZE,
        'shuffle':True,
        'num_workers':0
    }
    val_params = {
        'batch_size':VALID_BATCH_SIZE,
        'shuffle':True,
        'num_workers':0
    }

    train_loader = DataLoader(train,**train_params)
    val_loader = DataLoader(val,**val_params)

    s = next(iter(train_loader))
    o = model(s['ids'],s['mask'])
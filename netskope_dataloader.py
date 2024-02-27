# Inspired by code at https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=zHxRRzqpBf76

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
import pickle
import pandas as pd

class NetSkopeDataset(Dataset):
    def __init__(self, dataframe_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        if '.pkl' in dataframe_path:
            with open(dataframe_path,'rb') as file:
                dataframe = pickle.load(file)
        if '.csv' in dataframe_path:
            dataframe = pd.read_csv(dataframe_path, encoding = 'utf-8')
        self.data = dataframe
        self.title = dataframe.Title
        self.targets = self.data.drop(columns='Title')
        self.max_len = max_len
    
    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        # title = str(self.title[index])
        # title = " ".join(title.split())

        # inputs = self.tokenizer.encode(
        #     title,
        #     None,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     padding='max_length',
        #     return_token_type_ids=True
        # )
        # if isinstance(self.title[index], str):
        inputs = self.tokenizer.encode_plus(self.title[index],
                                    add_special_tokens = True,
                                    max_length = self.max_len,
                                    padding='max_length',
                                    truncation = True)
        # else:
        #     inputs = [token for sample in self.title[index].apply(lambda x: tokenizer.encode(x,
        #                                                                                      add_special_tokens=True,
        #                                                                                      max_length = self.max_len,
        #                                                                                      padding='max_length')).tolist() for token in sample 
        #           if token != tokenizer.cls_token_id]
        #     inputs.insert(0,tokenizer.cls_token_id)

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'Role':torch.tensor(self.targets['Job Role'][index].tolist()),
            'Function':torch.tensor(self.targets['Job Function'][index].tolist()),
            'Level':torch.tensor(self.targets['Job Level'][index].tolist())
        }

if __name__ == '__main__':
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
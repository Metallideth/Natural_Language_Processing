# Inspired by code at https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb#scrollTo=zHxRRzqpBf76

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import pickle
import pandas as pd

class NetSkopeDataset(Dataset):
    """Dataset class to feed into pytorch DataLoader so that small pieces of dataset can be fed into the model one batch at a time
    
    :param dataframe_path: Path to the dataframe to turn into a Dataset
    :type dataframe_path: String object, required
    :param tokenizer: Tokenizer to use to convert input data Titles into a sequence of tokens for the model
    :type tokenizer: transformers.DistilBertTokenizer object, required
    :param max_len: The maximum length of an input sequence, at which point the tokenizer will truncate the sequence
    :type max_len: Integer
    """
    def __init__(self, dataframe_path, tokenizer, max_len):
        """Constructor method
        """
        self.tokenizer = tokenizer
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY BELOW
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY BELOW
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY BELOW
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY BELOW
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY BELOW
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        # If a change is desired for data import architecture, changes should be made to the following lines

        if '.pkl' in dataframe_path:
            with open(dataframe_path,'rb') as file:
                dataframe = pickle.load(file)
        if '.csv' in dataframe_path:
            dataframe = pd.read_csv(dataframe_path, encoding = 'utf-8')
        
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY ABOVE
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY ABOVE
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY ABOVE
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY ABOVE
        # INITIAL DATA IMPORT HANDLED IN THE CODE BLOCK DIRECTLY ABOVE
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        self.data = dataframe
        self.title = dataframe.Title
        self.targets = self.data.drop(columns='Title')
        self.max_len = max_len
    
    def __len__(self):
        """Length method

        :return: Number of rows in the dataset
        :rtype: Integer
        """
        return len(self.title)

    def __getitem__(self, index):
        """Indexing method
        
        :param index: The index used to access the desired item
        :type index: Integer

        :return: A dictionary of input token ids and mask ids, both of which are tensors of integers. If not in production mode, also return a tensor of individual integers for each sequence, one for each of Role, Function, and Level.
        :rtype: A dictionary of torch integer tensors
        """
        inputs = self.tokenizer.encode_plus(self.title[index],
                                    add_special_tokens = True,
                                    max_length = self.max_len,
                                    padding='max_length',
                                    truncation = True)
        
        try:
            return_dict = {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'Role':torch.tensor(self.targets['Job Role'][index].tolist()),
                'Function':torch.tensor(self.targets['Job Function'][index].tolist()),
                'Level':torch.tensor(self.targets['Job Level'][index].tolist())
            }
        except KeyError:
            return_dict = {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
            }
        return return_dict

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
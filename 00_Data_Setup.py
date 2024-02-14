# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import pickle

parser = argparse.ArgumentParser(description='Setup data into training, validation, and test sets')
parser.add_argument('-r','--randomseed', help = 'Random seed for data splitting, default = 2024', default = 2024)
parser.add_argument('-v','--valsize', help = 'Size of the validation data, default = 0.1', default = 0.1)
parser.add_argument('-t','--testsize', help = 'Size of the test data, default = 0.1', default = 0.1)
parser.add_argument('-dp','--datapath', help = 'Path to overall data file, default = Data/Historical Lead Records.csv',
                    default = 'Data/Historical Lead Records.csv')
parser.add_argument('-de','--dataencoding', help = 'Encoding to use for reading the data file, default = ISO-8859-1',
                    default = 'ISO-8859-1')
parser.add_argument('-df','--datafolder', help = 'Path to datafolder to save data splits, default = Data/',
                    default = 'Data/')
args = parser.parse_args()
RANDOMSEED = args.randomseed
VALSIZE = args.valsize
TESTSIZE = args.testsize
PATH = args.datapath
ENCODING = args.dataencoding
DATAFOLDER = args.datafolder

def consolidate_records(path):
    '''path: relative path to the csv file to be fed in for data'''
    data = pd.read_csv(path,
                        encoding=ENCODING)
    # No need for rows with na input, we'll drop those
    data = data.loc[~data['Title'].isna()].reset_index().drop(columns='index')
    # We'll also drop rows with na in any one of the outputs, since we saw in EDA that accounts for a small portion of the data
    data = data.loc[~(data['Job Role'].isna()|data['Job Function'].isna()|data['Job Level'].isna())].reset_index().drop(columns='index')
    # Need encoding change for weird characters to come through
    data = data.copy()

    data = data.replace({'Job Role':['INformation Security', 'information security']}, 'Information Security')
    data = data.replace({'Job Role':['Netoworking']}, 'Networking')
    data = data.replace({'Job Role':['IT Facilities', 'IT', 'Senior Manager, Information Technology']}, 'IT General')
    data = data.replace({'Job Role':['Business Systems']}, 'Systems')
    data = data.replace({'Job Role':['Senior Manager, Security, Risk, and Compliance', 'IT/IS Compliance/Risk/Control Staff']}, 'Governance Risk Compliance')
    data.loc[~data['Job Role'].isin(['Information Security','Networking','IT General','Systems','Governance Risk Compliance']) &
                ~data['Job Role'].isna(),
                data.columns == 'Job Role'] = 'Non-ICP'
    
    data = data.replace({'Job Function':['Information Technology','IT - Security','IT - Network','Information Security, Information Technology','IT Operations','IT-Sec Admin','Director Global IT','Information Security, Information Technology, Enterprise Architecture','It','Information Technology, Information Technology Executive']},
                              'IT')
    data = data.replace({'Job Function':['Engineering & Technical','Engineer SASE']},'Engineering')
    data = data.replace({'Job Function':['Purchasing','Sourcing / Procurement']},'Procurement')
    data = data.replace({'Job Function':['Legal','Risk, Legal Operations','Lawyer / Attorney','Governmental Affairs & Regulatory Law']},
                                'Risk/Legal/Compliance')
    data.loc[~data['Job Function'].isin(['IT','Engineering','Procurement','Risk/Legal/Compliance']) &
                ~data['Job Function'].isna(),
                data.columns == 'Job Function'] = 'Non-ICP'
    
    data = data.replace({'Job Level':['Individual Contributor','contributor','contribtuor']},'Contributor')
    data = data.replace({'Job Level':['Management','Manager Level','manager','Threat Hunting Manager','IT Security Manager']},'Manager')
    data = data.replace({'Job Level':['Senior Executive','Exec.']},'Executive')
    data = data.replace({'Job Level':['Director Level','IT Infrastructure Director','Director of Enterprise Cloud Business','IT Security Director']},'Director')
    data = data.replace({'Job Level':['C-level','CxO','C level','C-suite','Director (It & Project) & Chief Information Security Officer','C Level']},'C-Level')
    data.loc[~data['Job Level'].isin(['Contributor','Manager','Executive','Director','C-Level']) &
                ~data['Job Level'].isna(),
                data.columns == 'Job Level'] = 'Unknown'
    return data

lead_data = consolidate_records(PATH)
# Train, validation, and test split
train_data, val_and_test_data = train_test_split(lead_data, test_size = VALSIZE+TESTSIZE, 
                                                 random_state = RANDOMSEED) # 80% of data for training
val_data, test_data = train_test_split(val_and_test_data, test_size = TESTSIZE/(VALSIZE+TESTSIZE))
    # 10% of data for validation, and 10% of data for test

# Integer encoding of training dataset, to then follow it with integer encoding of validation and test based on the same encoding

target_columns = ['Job Role','Job Function','Job Level']

index_to_labels = {}
label_encoders = {}
for column in target_columns:
    this_encoder = {}
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(train_data[column].values)
    index_to_label = {i:label for i,label in enumerate(label_encoder.classes_)}
    train_data[column] = integer_encoded
    label_encoders[column] = label_encoder
    index_to_labels[column] = index_to_label

# Now apply to validation and test

for column in target_columns:
    label_encoder = label_encoders[column]
    val_data[column] = label_encoder.transform(val_data[column].values)
    test_data[column] = label_encoder.transform(test_data[column].values)

# Reset indexes
train_data = train_data.reset_index().drop(columns='index')
val_data = val_data.reset_index().drop(columns='index')
test_data = test_data.reset_index().drop(columns='index')

# Save final files

train_data.to_csv(f'{DATAFOLDER}train.csv')
val_data.to_csv(f'{DATAFOLDER}val.csv')
test_data.to_csv(f'{DATAFOLDER}test.csv')

with open(f'{DATAFOLDER}train.pkl','wb') as file:
    pickle.dump(train_data,file)

with open(f'{DATAFOLDER}val.pkl','wb') as file:
    pickle.dump(val_data,file)
    
with open(f'{DATAFOLDER}test.pkl','wb') as file:
    pickle.dump(test_data,file)

with open(f'{DATAFOLDER}index_label_mapping.pkl','wb') as file:
    pickle.dump(index_to_labels,file)
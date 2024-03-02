# Importing stock ml libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import pickle

parser = argparse.ArgumentParser(description='Setup data into training, validation, and test sets')
parser.add_argument('-r','--randomseed', help = 'Random seed for data splitting, default = 2024', default = 2024)
parser.add_argument('-v','--valsize', help = 'Size of the validation data, default = 0.1', default = 0.1)
parser.add_argument('-t','--testsize', help = 'Size of the test data, default = 0.1', default = 0.1)
parser.add_argument('-dp','--datapath', help = 'Path to overall data file, default = Data/Historical Lead Records - Condensed REMAPPED.csv',
                    default = 'Data/Historical Lead Records - Condensed REMAPPED.csv')
parser.add_argument('-de','--dataencoding', help = 'Encoding to use for reading the data file, default = utf-8',
                    default = 'utf-8')
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
    data = data.loc[(data['Title'] != "#NAME?")].reset_index().drop(columns='index')
    data = data.loc[(data['Title'] != '** NO LONGER WITH COMPANY **')].reset_index().drop(columns='index')
    data = data.loc[data['Title'].str.contains('[A-Za-z]')].reset_index().drop(columns='index')
    # Just some more odd data records that should be dropped
    data = data.loc[~(data['Job Role'].isna()|data['Job Function'].isna()|data['Job Level'].isna())].reset_index().drop(columns='index')
    # Need encoding change for weird characters to come through
    data = data.copy()

    data = data.replace({'Job Role':['NETOWORKING']}, 'NETWORKING')
    data = data.replace({'Job Role':['IT FACILITIES', 'IT', 'SENIOR MANAGER, INFORMATION TECHNOLOGY']}, 'IT GENERAL')
    data = data.replace({'Job Role':['BUSINESS SYSTEMS']}, 'SYSTEMS')
    data = data.replace({'Job Role':['SENIOR MANAGER, SECURITY, RISK, AND COMPLIANCE', 'IT/IS COMPLIANCE/RISK/CONTROL STAFF']}, 'GOVERNANCE RISK COMPLIANCE')
    data.loc[~data['Job Role'].isin(['INFORMATION SECURITY','NETWORKING','IT GENERAL','SYSTEMS','GOVERNANCE RISK COMPLIANCE','DEVELOPMENT']) &
                ~data['Job Role'].isna(),
                data.columns == 'Job Role'] = 'NON-ICP'
    
    data = data.replace({'Job Function':['INFORMATION TECHNOLOGY','IT - SECURITY','IT - NETWORK','INFORMATION SECURITY, INFORMATION TECHNOLOGY','IT OPERATIONS','IT-SEC ADMIN','DIRECTOR GLOBAL IT','INFORMATION SECURITY, INFORMATION TECHNOLOGY, ENTERPRISE ARCHITECTURE','INFORMATION TECHNOLOGY, INFORMATION TECHNOLOGY EXECUTIVE']},
                              'IT')
    data = data.replace({'Job Function':['ENGINEERING & TECHNICAL','ENGINEER SASE']},'ENGINEERING')
    data = data.replace({'Job Function':['PURCHASING','SOURCING / PROCUREMENT']},'PROCUREMENT')
    data = data.replace({'Job Function':['LEGAL','RISK, LEGAL OPERATIONS','LAWYER / ATTORNEY','GOVERNMENTALK AFFAIRS & REGULATORY LAW']},
                                'RISK/LEGAL/COMPLIANCE')
    data.loc[~data['Job Function'].isin(['IT','ENGINEERING','PROCUREMENT','RISK/LEGAL/COMPLIANCE']) &
                ~data['Job Function'].isna(),
                data.columns == 'Job Function'] = 'NON-ICP'
    
    data = data.replace({'Job Level':['INDIVIDUAL CONTRIBUTOR','CONTRIBTUOR']},'CONTRIBUTOR')
    data = data.replace({'Job Level':['MANAGEMENT','MANAGER LEVEL','MANAGER','THREAT HUNTING MANAGER','IT SECURITY MANAGER']},'MANAGER')
    data = data.replace({'Job Level':['SENIOR EXECUTIVE','EXEC.']},'EXECUTIVE')
    data = data.replace({'Job Level':['DIRECTOR LEVEL','IT INFRASTRUCTURE DIRECTOR','DIRECTOR OF ENTERPRISE CLOUD BUSINESS','IT SECURITY DIRECTOR']},'DIRECTOR')
    data = data.replace({'Job Level':['CXO','C-SUITE','DIRECTOR (IT & PROJECT) & CHIEF INFORMATION SECURITY OFFICER','C LEVEL']},'C-LEVEL')
    data.loc[~data['Job Level'].isin(['CONTRIBUTOR','MANAGER','EXECUTIVE','DIRECTOR','C-LEVEL']) &
                ~data['Job Level'].isna(),
                data.columns == 'Job Level'] = 'UNKNOWN'
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

lead_data_encoded = lead_data.copy()

for column in target_columns:
    label_encoder = label_encoders[column]
    val_data[column] = label_encoder.transform(val_data[column].values)
    test_data[column] = label_encoder.transform(test_data[column].values)
    lead_data_encoded[column] = label_encoder.transform(lead_data[column].values)

# Reset indexes
lead_data = lead_data.reset_index().drop(columns='index')
lead_data_encoded = lead_data_encoded.reset_index().drop(columns='index')
train_data = train_data.reset_index().drop(columns='index')
val_data = val_data.reset_index().drop(columns='index')
test_data = test_data.reset_index().drop(columns='index')
train_small = train_data.iloc[:1280]
val_small = val_data.iloc[:1280]
test_small = test_data.iloc[:1280]
# Save final files

lead_data.to_csv(f'{DATAFOLDER}lead_data.csv')
lead_data_encoded.to_csv(f'{DATAFOLDER}lead_data_encoded.csv')
train_data.to_csv(f'{DATAFOLDER}train.csv')
train_small.to_csv(f'{DATAFOLDER}train_small.csv')
val_data.to_csv(f'{DATAFOLDER}val.csv')
val_small.to_csv(f'{DATAFOLDER}val_small.csv')
test_data.to_csv(f'{DATAFOLDER}test.csv')
test_small.to_csv(f'{DATAFOLDER}test_small.csv')

with open(f'{DATAFOLDER}lead_data.pkl','wb') as file:
    pickle.dump(lead_data,file)

with open(f'{DATAFOLDER}lead_data_encoded.pkl','wb') as file:
    pickle.dump(lead_data_encoded,file)

with open(f'{DATAFOLDER}train.pkl','wb') as file:
    pickle.dump(train_data,file)

with open(f'{DATAFOLDER}train_small.pkl','wb') as file:
    pickle.dump(train_small,file)

with open(f'{DATAFOLDER}val.pkl','wb') as file:
    pickle.dump(val_data,file)

with open(f'{DATAFOLDER}val_small.pkl','wb') as file:
    pickle.dump(val_small,file)
    
with open(f'{DATAFOLDER}test.pkl','wb') as file:
    pickle.dump(test_data,file)

with open(f'{DATAFOLDER}test_small.pkl','wb') as file:
    pickle.dump(test_small,file)

with open(f'{DATAFOLDER}index_label_mapping.pkl','wb') as file:
    pickle.dump(index_to_labels,file)
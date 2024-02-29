settings_dict = {
    'MAX_LEN':64, # Will truncate some sequences, but they'll have to be super long to surpass this, and most of the relevant information will be contained in the beginning
    'TRAIN_BATCH_SIZE':128,
    'VALID_BATCH_SIZE':128,
    'INF_BATCH_SIZE':128,
    'IMPACT_EVAL_BATCH_SIZE':1,
    'EPOCHS':20,
    'LEARNING_RATE':1e-05,
    'WEIGHTS':{'Role':1,'Function':1,'Level':1},
    'INF_WEIGHTS':{'Role': 1.0116539409992857,'Function': 0.990414851774975,'Level': 0.9979312072257394},
    'DIMENSIONS':{'Role':7,'Function':5,'Level':6},
    'ACCSTOP':{'Role':0.999,'Function':0.999,'Level':0.999},
    'RANDOMSEED':2024,
    'LOGGINGFOLDER':'logging/',
    'INFERENCEFOLDER':'inference/',
    'TESTFOLDER':'test/',
    'CHECKPOINTLOC':'./checkpoints/26-02-2024_1922/epoch08',
    'ENCODER':'./Data/index_label_mapping.pkl',
    'OVERRIDE_TABLE':'./overrides/override_table.csv'
}
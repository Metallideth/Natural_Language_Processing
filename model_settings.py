settings_dict = {
    'MAX_LEN':64, # Will truncate some sequences, but they'll have to be super long to surpass this, and most of the relevant information will be contained in the beginning
    'TRAIN_BATCH_SIZE':128,
    'VALID_BATCH_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':1e-05,
    'WEIGHTS':{'Role':1,'Function':1,'Level':1},
    'DIMENSIONS':{'Role':6,'Function':5,'Level':6},
    'ACCSTOP':{'Role':0.9,'Function':0.9,'Level':0.9},
}
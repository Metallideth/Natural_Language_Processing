# Netskope Job Role/Function/Level Tagging

This package contains all the code you'll need to run a production-level model to tag a series of input Job Titles with their associated Roles, Functions, and Levels, according to the desired future label hierarchy. There is built-in override functionality, as well as the ability to retrain models in the future. A user input mode allows the user to test output tags for title labels that are directly entered into the terminal.

1. [Prerequisites](#prerequisites)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Directory Setup - Subfolders](#directory-setup---subfolders)
4. [Directory Setup - Main Files](#directory-setup---main-files)
    * [00_Data_Setup.py](#00_data_setuppy)
    * [distilbert_uncased_model.py](#distilbert_uncased_modelpy)
    * [model_settings.py](#model_settingspy)
    * [netskope_dataloader.py](#netskope_dataloaderpy)
    * [requirements.txt](#requirementstxt)
    * [utils.py](#utilspy)
    * [main.py](#mainpy)

## Prerequisites

Before beginning, you'll need to install some software (all free except for Office Suite):

1. Latest Anaconda distribution/python: https://www.anaconda.com/download
2. SQLNotebook: https://github.com/electroly/sqlnotebook/releases
3. Microsoft Office Suite (for Microsoft Excel)

## Setting Up the Environment

With Anaconda installed, you should be able to navigate to the start menu and search for "Anaconda Powershell Prompt", which allows access to the cmd terminal for running python scripts. Use the cd command to change the directory to the root directory of this repository (i.e. the location of the main.py python file), which you'll need to wrap in quotes:

```
cd "path-to-root-folder-here"
```

To install all required packages for this code (from the requirements.txt file), run the following commands (substitute for \<my-env\> whatever your desired environment name would be). If asked to proceed at any point, type "y" and hit enter.

```
conda create --name <my-env> python=3.12.1
conda activate <my-env>
pip install -r requirements.txt
```

In addition to the above, you'll need to install pytorch on your machine, which was not included in the requirements.txt file because the installation for pytorch is different depending on whether or not your machine has a graphics card to help with parallel processing during the neural network computations. It is highly recommended that you install a cuda-enabled version of pytorch if you can, since training and inference are run considerably faster on a cuda-enabled graphics card. Instructions for this installation can be found here: https://pytorch.org/get-started/locally/. This will take a few minutes due to the size of the pytorch package. As of this writing, full model functionality testing was achieved using the cuda 11.8 install of pytorch. Everything should work under the cpu installation as well, but model runs will be slower, especially in training mode.

If you have a graphics card and want to check to make sure pytorch was set up correctly with cuda enabled, open up a new Jupyter Notebook file with the kernel set to your newly created conda environment that you just completed the pytorch install. Then, run the following lines of code:

```
import torch
torch.cuda.is_enabled()
```

If the output is True, then pytorch is correctly installed with cuda functionality.

## Directory Setup - Subfolders

From the root directory, there are several folders:

1. checkpoints - when in training mode, upon the completion of a full epoch (pass through the training data), just prior to validation, a snapshot of the model checkpoint is saved under a subfolder within this folder. The subfolder is named according to the date and time of the training run: "dd-mm-yyyy_HHMM", and each checkpoint is named "epochxx", starting at epoch00 with potential for checkpoints up to epoch99.
2. Data - within this folder is all the data used for training and inference, as well as other important files needed for model functionality. Train, validation, and test sets are saved here in csv as well as pkl form, along with small subsets of each for functionality testing. In addition to these are the following files:
    * "Job Function_Role_Level Keywords.xlsx", "Historical Lead Records.xlsx", "Historical Lead Records - Condensed.xlsx" - files originally sent over by netskope, including a list of keywords for outputs for certain functions/roles/levels, a full list of original data, and a subset of original data filtered on only US sources
    * "00_Tagging_Mismatches.sqlnb" and "Historical Lead Records - Condensed REMAPPED.xlsx" - A SQL Notebook file for remapping the condensed data mentioned above such that each title has one unique mapping of function/role/level associated with it, the mapping that occurs most often. Titles are also converted to all uppercase so as to avoid needless differentiation in case; the output is saved to an excel file. Some other exploring is done in the SQL Notebook file as well, such as variance of distinct mapping counts.
    * index_label_mapping.pkl - a python dictionary object mapping model output integers to their associated labels
3. final_model - the model checkpoint to use for production is saved here
4. inference - this folder is the output destination of inference/predictions, when a csv of titles is fed into the final model to produce a function/role/level for each input title. Outputs are saved as csv and pkl versions. Already in this folder are:
    * Inference after initial training for analysis of observations with large loss
    * Inference for keyword (impact) and antikeyword analysis, along with associated excel files. The excel files are labeled as ALTERNATE, but they have replaced the original files in the Not_Used subfolder within this folder, as their priority index methodology was more statistically robust.
    * Further along in the production process, a change was made to reclassify results with level = UNKNOWN to the next most likely level, in accordance with desired final hierarchy; inference files were made before and after this change.
    * model_historic_divergence.xlsx - a full list of unique titles for which the model and the remapped source data differed, along with their associated loss and the frequency of each title's appearance in the data, ordered by loss from largest to smallest.
5. logging - similar to the checkpoints folder, this folder saves intermediate logging results every 1000 batches during iterations on the training data, as well as a summary of statistics on the training and validation datasets at the end of each training epoch. Logs are saved in subfolders according to the date and time of the training run: "dd-mm-yyyy_HHMM".
6. Notebooks - Jupyter Notebooks for various ad-hoc analyses. Included in this folder are the initial data exploration and post-training analysis after initial training for models. Files 00 and 01 corresponded to the original data before non-US sources were filtered out, with 02 and 03 corresponding to condensed US-only data that had been remapped so that each title had only one unique label. 04 shows the training visualizations underlying the model used to create the inference data in 03. Keyword and antikeyword data was processed and turned into the excel files shown in the inference folder mentioned above from notebooks 05 and 06. **NOTE:** 05 and 06 were superseded by outputs from notebooks 10 and 11. 07, 08, and 09 were used to generate supplementary exhibits for the final report writeup.
7. overrides - this folder contains a single table used for overriding predictions the model will make. The first column is for the Title, the 2nd through 4th columns for the associated Role, Function, and Level to override the model's prediction.
8. test - this file includes the test summary files in subfolders based on the time of test run, labeled similarly to logging and checkpoint subfolders.

## Directory Setup - Main Files

### 00_Data_Setup.py

This file was used to create the train, val, test, and their small versions from the original data after it was remapped and condensed. This file can be run from the command line provided the cd command is first used to navigate to the root directory for this package, and the right conda environment with all the necessary packages is activated:

```
python 00_Data_Setup.py <args>
```

For more information about the arguments to specify and what the defaults are if you enter nothing, look into the file code or run the following command:

```
python 00_Data_Setup.py -h
```

If there is desire to train further models, the prior data will likely need to be remapped with the SQL Notebooks file in the Data folder, a csv will need to be generated, and this script will need to process and clean the data in that csv.

### distilbert_uncased_model.py

This script sets up the structure of the model to use for prediction by appending a few classification layers to a pretrained distilbert uncased model.

### model_settings.py

This file is a dictionary that contains various fixed settings that affect training, validation, and inference. If changes are desired, they should be made directly to this file: Settings include:

1. MAX_LEN - the maximum token length of input sequences. This should be set at the time of training and not changed during inference, as the model is implicitly trained by only paying attention to the first MAX_LEN tokens of every sequence. Higher values will guarantee that even longer sequences are fully represented at the expense of taking longer to compute. At the current value of 64, only extremely long titles will be truncated.
2. TRAIN_BATCH_SIZE - the size of each batch to feed in for training. Current value is 128, but this might need to be decreased if the graphics card/system ram is not capable of dealing with batches that large. General rule of thumb is to have the batch size as large as the system can manage.
3. VALID_BATCH_SIZE - see TRAIN_BATCH_SIZE, batch size used for validation and testing.
4. INF_BATCH_SIZE - see TRAIN_BATCH_SIZE, batch size used for general inference.
5. IMPACT_EVAL_BATCH_SIZE - batch size used for impact (keyword and antikeyword) evaluation. This is fixed at 1 and should not be changed.
6. EPOCHS - number of iterations through entire training set to run for a training run. 20 is a good size, as the best model loss was achieved after 9 epochs in the best model version.
7. LEARNING_RATE - learning rate of the model during training
8. WEIGHTS - loss function is a weighted sum of cross-entropy losses for each category. The initial weights for training are determined by this setting, equal weights are assumed; weights are updated during training to weight more towards which of Role/Function/Level is less accurate, so as to prioritize updates for more inaccurate classifications.
9. INF_WEIGHTS - weights calculated in 03 notebook file, cell 4, to use for inference that calculates loss along with predicted values. Update after new training run, otherwise leave as is.
10. DIMENSIONS - output dimensions for each category. Leave as is.
11. ACCSTOP - training setting to serve as a stopping point if accuracy for all output categories reaches at least as high as the chosen level. I set a prohibitively high standard of 0.999 for each category here, opting instead to select the best checkpoint based on weighted cross-entropy loss, but this can be tweaked if the user desires.
12. RANDOMSEED - set to implement reproducible results, but ultimately this is not implemented anywhere.
13. LOGGINGFOLDER - folder to save logging results in
14. INFERENCEFOLDER - folder to save inference runs in
15. TESTFOLDER - folder to save test results in
16. CHECKPOINTLOC - checkpoint to load for model mode runs. If training is run, the loaded model is the starting point for training; can be set to None if desired to train from scratch.
17. ENCODER - dictionary with keys that determine how to map integer values to string outputs. Processed data is saved as integers for saving space/computational resources during model training/inference.
18. OVERRIDE_TABLE - path to csv of override table

### netskope_dataloader.py

This file sets up dataloaders so that the input data can be fed onto the graphics card/into the model in small pieces. More documentation about this can be found here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html.

### requirements.txt

Contains required packages to run scripts and notebooks in this directory.

### utils.py

Various functions and methods needed to run scripts on main.py

### main.py

File that contains core functionality for running training, validation, test, and inference modes. Similar to the data setup file, this file can be run from the command line:

```
python main.py
```

To see a description of the script arguments to be passed in from the command, enter into the command line:

```
python main.py -h
```

Note that the --modelmode argument can have many values:

1. training - runs training on the chosen training set, with inter-epoch validation done on the chosen validation set. Logging is turned on by default and checkpoints are saved after each epoch.
2. inference_production - **probably should be the model mode most often used**. Using the model from the checkpoint in model_settings.py, predictions are made based on job titles passed in, and are then mapped to the desired go-forward hierarchy for netskope.
3. inference - similar to inference_production, except without the mapping to new desired go-forward hierarchy
4. inference_loss - similar to inference, except with the addition of the loss for each observation. This requires a dataset with labeled targets already, processed with 00_Data_Setup.py. I used this to produce the table shown in notebook 03 with loss, as well as the model_historic_divergence.xlsx file in the inference folder.
5. user_input - loads up the model from the checkpoint in model_settings.py to allow user testing for one input at a time, entered into the command terminal.
6. test - computes model scores on test set and saves them in the specified test folder according to model_settings.py
7. impact_eval and antikey_eval - both modes used to calculate basis for keyword and antikeyword impact files in inference folder. These take an extremely long time to run, even with a graphics card.
#NEED TO LOAD TESTING TOOLS
import logging
from utils.testing_tools import create_test, dump_test_case, load_test_cases
import glob, os

#stract stats from the training folder
n_valid_images = len(glob.glob(os.path.join('..', 'data', 'validation','images','*')))
n_train_images = len(glob.glob(os.path.join('..', 'data', 'train','images','*')))

#define Hyperparameters to load
encoder_type= 'DFCN_5layers'
learning_rate = 0.001
batch_size = 50
num_epochs = 200
steps_per_epoch = int(n_train_images/num_epochs)
validation_steps = int(n_valid_images/num_epochs)
workers = 2

learning_rates = [0.001, 0.0001, 0.00001]

for lear in learning_rates:
    test = create_test(encoder_type,lear,batch_size,num_epochs,steps_per_epoch,validation_steps,workers)
    tests_cases = dump_test_case(test,'DFCN_tests.p')
#Create a pickle file and load models in it
for it in tests_cases:
    print(it)

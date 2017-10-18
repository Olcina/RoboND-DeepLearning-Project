#NEED TO LOAD TESTING TOOLS
import logging
from utils.testing_tools import create_test, dump_test_case, load_test_cases
import glob, os

#stract stats from the training folder
n_valid_images = len(glob.glob(os.path.join('..', 'data', 'validation','images','*')))
n_train_images = len(glob.glob(os.path.join('..', 'data', 'train','images','*')))

print(n_train_images)

#define Hyperparameters to load
encoder_type= 'FCN4_SIMPLE'
learning_rate = 0.0001
batch_size = 200
num_epochs = 40
steps_per_epoch = int(n_train_images/num_epochs)
validation_steps = int(n_valid_images/num_epochs)
workers = 4


# load the test with the function create_test
for i in range(0,5):
    test = create_test(encoder_type,learning_rate,batch_size,num_epochs,steps_per_epoch,validation_steps,workers)
    #load the test if doesnt exist another with the same name before
    tests_cases = dump_test_case(test,'best_model_test_more_data2.p')

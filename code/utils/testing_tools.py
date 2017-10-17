import pickle
import logging
import os

def create_test(encoder_type,learning_rate,batch_size,num_epochs,steps_per_epoch,validation_steps,workers):
    test = {}
    #Define a code name for the model on each launch
    lr = str(learning_rate).split('.') # avoid the . or , in the number for file name
    code_name = encoder_type + '_'
    try:
        code_name += 'lr' + str(lr[1]) + '_'
    except:
        code_name += 'lr' + str(learning_rate) + '_'
    code_name += 'bs' + str(batch_size) + '_'
    code_name += 'ne' + str(num_epochs) + '_'
    code_name += 'se' + str(steps_per_epoch) + '_'
    code_name += 'vs' + str(validation_steps) + '_'
    code_name += 'wr' + str(workers)

    #add variables to the test dict
    test['name'] = code_name
    test['encoder_type'] = encoder_type
    test['learning_rate'] = learning_rate
    test['batch_size'] = batch_size
    test['num_epochs'] = num_epochs
    test['steps_per_epoch'] = steps_per_epoch
    test['validation_steps'] = validation_steps
    test['workers'] = workers

    return test

def dump_test_case(test, pickle_file ='test_cases.p'):
    #load test in the pickle file and creates the file if doesn't exist
        pickle_path = os.path.join('test_cases', pickle_file)
        print(pickle_path)
        tests_cases = load_test_cases(pickle_path)
        #add the new models to the model lists in pickle
        if not test['name'] in [x['name'] for x in tests_cases]:
            tests_cases.append(test)
            print('test: ', test['name'] , ' loaded')
        else:
            print('this settings are already loaded')
        #load the model to the text file
        with open(pickle_path, "wb" ) as file:
            pickle.dump( tests_cases, file )

        return tests_cases

def load_test_cases(pickle_file):
    pickle_path = os.path.join(pickle_file)
    #load test cases for
    try:
        with open(pickle_path, "rb" ) as file:
            tests_cases = pickle.load(file)
    except:
            tests_cases = []
            open(pickle_path, 'wb+')
            logging.info('created file: "', pickle_file, '" with empty string test = []')
    return tests_cases

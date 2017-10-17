from utils.testing_tools import create_test, dump_test_case, load_test_cases
import glob, os

def load_available_models(print_in_terminal= False):
    available_models = glob.glob(os.path.join('test_cases','*.p'))

    print(' ')
    print('SELECT MODEL YOU WANT TO VISUALIZE')
    print(' ')
    for i in range(0,len(available_models)):
        print(i+1, ' : ', available_models[i])

    case = input('Enter the number you want to view:  ')
    print('')

    tests_cases = load_test_cases(available_models[int(case)-1])
    if print_in_terminal:
        for it in tests_cases:
            print(it)
    return tests_cases
#Create a pickle file and load models in it
if __name__ == '__main__':

    tests_cases = load_available_models(True)

import argparse
from data_io.test_data import TestData
import logging


def create_loggers():
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='run.log', filemode='w', level=logging.DEBUG)

    # create logger
    logger = logging.getLogger('io')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

create_loggers()

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--sourcepath', help='Root path for the source files.', required=False, default='C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/apache-ant-1.7.0-src/apache-ant-1.7.0/src/main')
parser.add_argument('-b', '--bugdatapath', help='Path to the csv bug data sheet.', required=False, default='C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/ant-1.7.csv')
parser.add_argument('-s', '--save', help='Path to the location to save the model data in.', required=False, default='C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/save/')
parser.add_argument('-lt', '--loadtestdata', help='Path to pickeled feature vector.', required=False)
parser.add_argument('-im', '--buginfomapping', help='Row index of the class info inside the bug info csv.', required=False, default=2)
parser.add_argument('-bn', '--bugnumbermapping', help='Row index of the number_of_bugs inside the bug info csv.', required=False, default=23)
parser.add_argument('-st', '--savetestdata', help='Save test data or not.', action='store_true')
args = parser.parse_args()
test_data_path = args.sourcepath
bug_data_path = args.bugdatapath

#test_data_path = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/test/test'
#bug_data_path = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/test/ant-1.7.csv'
loadtestdata = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/'


test = TestData(test_data_path, bug_data_path)

if loadtestdata is None:
    test.initialize(args.buginfomapping, args.bugnumbermapping)
else:
    test.load_features(loadtestdata)
if args.savetestdata:
    test.save_features('C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/')

a = test.get_test_train_split()


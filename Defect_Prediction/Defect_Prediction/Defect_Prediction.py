import argparse
import logging
from data_io.test_data import DefectDataSetLoader, DataSet
from prediction.tf_model import TensorFlowNet


def create_loggers():
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='run.log', filemode='w', level=logging.DEBUG)

    # create loggers
    logger_io = logging.getLogger('io')
    logger_io.setLevel(logging.DEBUG)

    logger_prediction = logging.getLogger('prediction')
    logger_prediction.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger_io.addHandler(ch)
    logger_prediction.addHandler(ch)

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
load_test_data = args.loadtestdata
save_data_set = args.savetestdata
save_data_set = True
#test_data_path = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/test/test'
#bug_data_path = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/test/ant-1.7.csv'
load_test_data = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/'


data_set_loader = DefectDataSetLoader(test_data_path, bug_data_path, source_files_extension='.java', one_hot=False)

if load_test_data is None:
    data_set_loader.initialize(args.buginfomapping, args.bugnumbermapping)
else:
    data_set_loader.load_features(load_test_data)

if save_data_set:
    data_set_loader.save_features('C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/')

X_train, X_test, y_train, y_test = data_set_loader.get_test_train_split()

# create test sets
train = DataSet(X_train, y_train, 'Train', one_hot=False, num_classes=data_set_loader.num_classes)
test = DataSet(X_test, y_test, 'Test', one_hot=False, num_classes=data_set_loader.num_classes)

net = TensorFlowNet(train, test, data_set_loader.num_classes, 50, 0.01)
net.run_training()




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
    ch_pred = logging.StreamHandler()
    ch_pred.setLevel(logging.INFO)

    ch_io = logging.StreamHandler()
    ch_io.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch_io.setFormatter(formatter)
    ch_pred.setFormatter(formatter)

    # add ch to logger
    logger_io.addHandler(ch_io)
    logger_prediction.addHandler(ch_pred)

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


test_data_path = ['C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/apache-ant-1.6.0-src/src/main', 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/apache-ant-1.7.0-src/apache-ant-1.7.0/src/main']
bug_data_path = ['C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/ant-1.6.csv', 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/ant-1.7.csv']
#load_test_data = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/'


data_set_loader = DefectDataSetLoader(test_data_path, bug_data_path, source_files_extension='.java', one_hot=False, binary_class_labels=True)

if load_test_data is None:
    data_set_loader.initialize(args.buginfomapping, args.bugnumbermapping) 
else:
    data_set_loader.load_features(load_test_data, name='feature_vector_binary.pickle')

if save_data_set:
    data_set_loader.save_features('C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/')

#X_train, X_test, y_train, y_test = data_set_loader.get_test_train_split()
# create test sets
#train = DataSet(X_train, y_train, 'Train', one_hot=False)
#test = DataSet(X_test, y_test, 'Test', one_hot=False)


X, y = data_set_loader.get_project_split()
train = DataSet(X[0], y[0], 'Train', one_hot=False)
test = DataSet(X[1], y[1], 'Test', one_hot=False)

net = TensorFlowNet(
    train,
    test, 
    data_set_loader.num_classes, 
    50, 
    initial_learning_rate=0.01, 
    architecture_shape=(64,16), 
    log_dir='C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/tf',
    max_steps=20000
    )
net.run_training()
num_prediction_tests = 10
X, y = test.get_random_elements(num_prediction_tests)

for i in range(num_prediction_tests):
    y_hat, prob = net.predict(X[i])
    print('Y: {0} - Y predicted: {1} ({2:.2f})'.format(y[i], y_hat, prob))




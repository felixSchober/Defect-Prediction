import argparse
from io.test_data import TestData
import logging


def create_loggers():
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='run.log', filemode='w', level=logging.DEBUG)

    # create logger
    logger = logging.getLogger('io')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

create_loggers()
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path for test images', required=False, default='C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/apache-ant-1.7.0-src/apache-ant-1.7.0/src/main')
args = parser.parse_args()
test_data_path = args.path

test = TestData(test_data_path)
test.initialize()





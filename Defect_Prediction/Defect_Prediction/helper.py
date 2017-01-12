from enum import Enum
import uuid
import os
import errno



TF_LAYER = Enum('Layer_type', 'Dense Dropout Convolution2D MaxPooling Normalization')


def get_uuid():
    """ Generates a unique string id."""

    x = uuid.uuid1()
    return str(x)

def colored_shell_seq(color):
    if color == 'RED':
        return '\033[31m'
    elif color == 'GREEN':
        return '\033[32m'
    elif color == 'WHITE':
        return '\033[37m'

# helper methods
def create_dir_if_necessary(path):
    """ Save way for creating a directory (if it does not exist yet). 
    From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary

    Keyword arguments:
    path -- path of the dir to check
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def check_if_dir_exists(path):
    """ Checks if a directory exists."""

    # From http://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python
    return os.path.isdir(path)

def get_unique_layer_name(layer_type):
    return str(layer_type) + '_' + get_uuid()


def create_loggers():
    import logging
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='run.log', filemode='w', level=logging.DEBUG)

    # create loggers
    logger_io = logging.getLogger('main')
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

def tensor_shape_to_list(tensor_shape):
    output = []
    for dim in tensor_shape:
        output.append(dim.value)
    return output
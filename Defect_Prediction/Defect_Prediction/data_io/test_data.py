from os import walk, remove
import os.path as osPath
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import javalang
from data_io.csv_data import get_csv_row_generator
from misc import utils


logger = logging.getLogger('io')


def to_one_hot(y):
    """Transform multi-class labels to binary labels
        The output of to_one_hot is sometimes referred to by some authors as the
        1-of-K coding scheme.
        Parameters
        ----------
        y : numpy array or sparse matrix of shape (n_samples,) or
            (n_samples, n_classes) Target values. The 2-d matrix should only
            contain 0 and 1, represents multilabel classification. Sparse
            matrix can be CSR, CSC, COO, DOK, or LIL.
        Returns
        -------
        Y : numpy array or CSR matrix of shape [n_samples, n_classes]
            Shape will be [n_samples, 1] for binary problems.
        classes_ : class vector extraceted from y.
        """
    lb = LabelBinarizer()
    lb.fit(y)
    Y = lb.transform(y)
    return (Y, lb.classes_)
        

class DefectDataSetLoader(object):
    """description of class"""

    def __init__(self, source_root_path, bug_data_path, source_files_extension=('.java'), one_hot=True):
        # root path of the java project and location of the bug data sheet
        self.__root_path = source_root_path
        self.__bug_data_path = bug_data_path

        # dict which contains the path (value) of a single file /class
        # and the package info (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory)
        # as the index.
        self.__source_files = {}

        # dict which contains the number of bugs of a single file / class
        # and the package info (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory)
        # as the index. 
        self.__bug_data = {}

        # list: [(class_info, path_to_class_file, number_of_bugs)]
        # class_info: (package info) (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory) 
        self.test_data = []

        self.test_data_X = []
        self.test_data_Y = []
        self.one_hot = one_hot
        self.num_classes = -1
        self.class_vector = []

        self.token_mapping = {
            'MethodDeclaration': 1,
            'ClassDeclaration': 2,
            'FieldDeclaration': 3,
            'EnumDeclaration': 4,
            'WhileStatement': 5,
            'ForStatement': 6,
            'IfStatement': 7,
            'ThrowStatement': 8,
            'TryStatement': 9,
            'CatchClause': 10,
            'ReturnStatement': 11
            }

        self.token_mapping_names = {
            'main': 12
            }
        
        self.current_mapping_index = 13

        self.source_files_extension = source_files_extension

        
    def initialize(self, class_info_mapping, number_of_bugs_mapping):

        logger.debug('Initializing source data set with path {0}.'.format(self.__root_path))

        # append '/' at the end if it does not exist.
        if not self.__root_path.endswith('/'):
            self.__root_path += '/'

        # iterate over root path and add all files to the source_files dict
        folder_list = []
        try:
            folder_list = next(walk(self.__root_path))[1]
        except:
            logger.exception('\tCould not iterate through root folder of dataset. - Path: {0}.'.format(self.__root_path))

        if len(folder_list) == 0:
            logger.error('Could not locate project root on path {0}. There were no folders inside the directory.'.format(self.__root_path))
            return None
        elif len(folder_list) > 1:
            logger.error('Could not locate project root on path {0}. There were more than one folder in the directory.\n\t\tExpected something like this [org]\n\t\tGot {2}.'.format(self.__root_path, folder_list))
            return None
        
        logger.debug('\tFound project root {0}.'.format(folder_list[0]))

        # go over source dir and index source files
        self.__find_files_recursively(self.__root_path + folder_list[0] + '/', folder_list[0])
        logger.debug('Finished indexing of source folder. Found {0} files.'.format(len(self.__source_files)))

        # iterate over bug data
        self.__load_bug_data(class_info_mapping, number_of_bugs_mapping)
        logger.debug('Finished indexing of bug data. Found data for {0} classes.'.format(len(self.__bug_data)))

        # map bug data 
        self.__map_bug_data()
        logger.debug('Finished mapping.')

        # set number of samples
        self.__num_examples = len(self.test_data)

        # create abstract syntax trees for every file
        self.__create_ast_vectors()

        # filter rare tokens and prepare data for use
        self.__prepare_data()

        self.num_classes = self.__get_num_classes()

        logger.debug('Finished data initialization.')


    def __get_num_classes(self):
        if self.one_hot:
            return self.test_data_Y.shape[1]
        
        return np.max(self.test_data_Y) + 1
        #return len(np.unique(self.test_data_Y))


    def save_features(self, path, name='feature_vector.pickle'):
        # append '/' at the end if it does not exist.
        if not path.endswith('/'):
            path += '/'
        file_name = path + name 
        logger.debug('Saving test data to file {0}.'.format(file_name))

        pickle_this = (self.test_data_X, self.test_data_Y, self.token_mapping_names, self.class_vector, self.one_hot)

        with open(file_name, 'wb') as f: 
            pickle.dump(pickle_this, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_features(self, path, name='feature_vector.pickle'):
        if not path.endswith('/'):
            path += '/'
        file_name = path + name
        logger.debug('Loading test data from file {0}.'.format(file_name))

        with open(file_name, 'rb') as f:
            unpickle_this = pickle.load(f)
        self.test_data_X = unpickle_this[0]
        self.test_data_Y = unpickle_this[1]
        self.token_mapping_names = unpickle_this[2]
        self.class_vector = unpickle_this[3]
        self.one_hot = unpickle_this[4]
                
        self.num_classes = self.__get_num_classes()

        logger.debug('Loaded test data. test_X shape: {0} - test_Y shape: {1} - Number of custom tokens: {2}'.format(self.test_data_X.shape, self.test_data_Y.shape, len(self.token_mapping_names)))


    def get_test_train_split(self, test_ratio=0.2, random_seed=42):
        logger.debug('Generating train/test split with test_size {0} and random_state {1}'.format(test_ratio, random_seed))
        X_train, X_test, y_train, y_test = train_test_split(self.test_data_X, self.test_data_Y, test_size=test_ratio, random_state=random_seed)
        logger.debug('Testset Split:')
        logger.debug('\tTrain Shape: {0} - {1}'.format(X_train.shape, y_train.shape))
        logger.debug('\tTest Shape: {0} - {1}'.format(X_test.shape, y_test.shape))
        return X_train, X_test, y_train, y_test
        
    def __find_files_recursively(self, current_path, current_index):
        # search for source files in current path
        source_files = []
        try:
            _, _, source_files = walk(current_path).__next__()
        except:
            logger.exception('\tCould not iterate through folder {0}.'.format(current_path))

        # filter by extension.
        source_files = [ file for file in source_files if file.endswith(self.source_files_extension)]

        # found some files (add them to the source_files dict)
        if len(source_files) > 0:
            print ('\n')
            logger.debug('\tFound {0} source files in {1}:'.format(len(source_files), current_index))
            for f in source_files:                
                # remove extension for index
                fi = f[:-len(self.source_files_extension)]
                self.__source_files[current_index + '.' + fi] = current_path + '/' + f
                logger.debug('\t\t- {0}.{1}'.format(current_index, fi))

        # search for new folders
        folder_list = []
        try:
            folder_list = next(walk(current_path))[1]
        except:
            logger.exception('\tCould not iterate through root folder of dataset. - Path: {0}.'.format(self.__root_path))

        for folder in folder_list:
            self.__find_files_recursively(current_path + folder + '/', current_index + '.' + folder)


    def __load_bug_data(self, class_info_mapping, number_of_bugs_mapping):
        logger.debug('Initializing bug data set with parameters {0} - {1}'.format(class_info_mapping, number_of_bugs_mapping))
        if not self.__bug_data_path.endswith('.csv'):
            raise AttributeError('Only .csv files are supported.')
        
        # iterate over entries and add them to the __bug_data list
        # skip the first row (contains only title)
        for entry in get_csv_row_generator(self.__bug_data_path, delimiter=';', skip_first_row=True):
            logger.debug('\tBugs: {1} \t-> Class: {0}'.format(entry[class_info_mapping], entry[number_of_bugs_mapping]))
            self.__bug_data[entry[class_info_mapping]] = entry[number_of_bugs_mapping]


    def __map_bug_data(self):
        logger.debug('Mapping bug data and source files.')
        for class_info, number_of_bugs in self.__bug_data.items():
            if not class_info in self.__source_files:
                logger.error('Could not find match for bug data file {0}.'.format(class_info))
                continue
            self.test_data.append((class_info, self.__source_files[class_info], number_of_bugs))


    def __create_ast_vectors(self):
        number_of_classes = len(self.test_data)
        i = 0
        logger.debug('Creating abstract syntax trees for {0} classes.'.format(number_of_classes))
        for (class_info, path_to_class_file, number_of_bugs) in self.test_data:
            # open file and read it
            source_code = ''
            with open(path_to_class_file, 'rb') as f:
                source_code = f.read()

            try:
                tree = javalang.parse.parse(source_code)
            except:
                logger.exception('Could not parse sourcefile {0} (Path: {1}). (Syntax errors)'.format(class_info, path_to_class_file))
                continue           

            # try to generate feature vector
            tree_feature_vector = self.__convert_tree_to_feature_vector_unstructured(tree)
            self.test_data_X.append(tree_feature_vector)
            self.test_data_Y.append(number_of_bugs)
            self.test_data[i] = (class_info, path_to_class_file, number_of_bugs, tree, tree_feature_vector)           

            utils.show_progress(True, i+1, number_of_classes, 'AST creation for {0} classes.\tToken mappings: {1}:', number_of_classes, len(self.token_mapping_names))
            i += 1


    def __convert_tree_to_feature_vector_unstructured(self, tree):
        feature_vector = []

        # iterate over tree vector
        for _, node in tree:
            # if node is either a method invocation or a class instance creation, check if name already exists in token mapping
            if str(node) == 'MethodInvocation' or str(node) == 'ClassCreator':
                name = ''
                if str(node) == 'MethodInvocation':
                    name = node.member
                else:
                    # search for 'ReferenceType' in class creators children
                    for childNode in node.children:
                        if str(childNode) == 'ReferenceType':
                            name = childNode.name
                            break

                if name in self.token_mapping_names:
                    feature_vector.append(self.token_mapping_names[name])
                else:
                    # add new method invocation mapping
                    self.token_mapping_names[name] = self.current_mapping_index
                    feature_vector.append(self.current_mapping_index)
                    self.current_mapping_index += 1 
                continue

            # if it is not either type search for standard mappings
            if str(node) in self.token_mapping:
                feature_vector.append(self.token_mapping[str(node)])
        return feature_vector


    def __prepare_data(self, rare_token_number=3):
        """
        1. Convert to numpy array
        2. Filter every token if occurence is less than 3
        3. TODO: Apply CLNI (if needed)
        4. Normalize X and y
        """
        
        logger.debug('Prepare test data for classification.')
        # convert to numpy array
        self.test_data_X = np.array(self.test_data_X)
        self.test_data_Y = np.array(self.test_data_Y, dtype=np.int32)

        # filter rare tokens
        flattened_feature_vector = [item for sublist in self.test_data_X for item in sublist]
        tokens, token_counter = np.unique(flattened_feature_vector, return_counts=True)
        token_dict = dict(zip(tokens, token_counter))
        filter = []
        # iterate over the test data and filter out. 
        # Note: this could also be done in a list comprehension but this is much easier to read.
        for i in range(len(tokens)):
            if not i in token_dict or token_dict[i] < rare_token_number:
                filter.append(i)

        logger.debug('Tokens before filtering: {0}'.format(len(tokens)))
        logger.debug('Filtered tokens: {0}'.format(len(filter)))

        # remove tokens from test_data feature vectors
        filtered_test_data_X = []
        max_feature_length = 0
        for feature_vector in self.test_data_X:
            filtered_vector = [token for token in feature_vector if not token in filter]
            max_feature_length = max(max_feature_length, len(filtered_vector))
            filtered_test_data_X.append(filtered_vector)

        logger.debug('Max feature vector length: {0}'.format(max_feature_length))
        logger.debug('Size of test_data_X before data prep: {0}'.format(self.test_data_X.nbytes))

        # append zeros so that each feature has the same length (max_feature_length)
        filtered_padded_test_data_X = []
        for feature_vector in filtered_test_data_X:
            padding = max_feature_length - len(feature_vector)
            filtered_padded_test_data_X.append(feature_vector + [0] * padding)

        self.test_data_X = np.array(filtered_padded_test_data_X, dtype=np.float32)


        # min-max normalization to range [0, 1]
        self.test_data_X /= (np.max(self.test_data_X) - np.min(self.test_data_X))
        logger.debug('Size of test_data_X after data prep: {0}'.format(self.test_data_X.nbytes))

        if self.one_hot:
            self.test_data_Y, self.class_vector = to_one_hot(self.test_data_Y)
            logger.debug('Converted Y data to one hot matrix. Classes: {0}'.format(self.class_vector))


from sklearn.utils import shuffle

class DataSet(object):

    def __init__(self, 
                 features,
                 targets,
                 name,
                 one_hot,
                 num_classes):

        self.__X = features
        self.__y = targets

        self.__epochs_completed = 0
        self.__index_in_epoch = 0

        # will be set in initialize 
        self.__num_examples = targets.shape[0]

        self.one_hot = one_hot
        self.__num_classes = num_classes
        

    @property
    def features(self):
        return self.__X

    @property
    def feature_shape(self):
        return self.__X.shape

    @property
    def targets(self):
        return self.__y

    @property
    def num_examples(self):
        return self.__num_examples

    @property
    def num_classes(self):
        return self.__num_classes

    @property
    def epochs_completed(self):
        return self.__epochs_completed

    def next_batch(self, batch_size):
        start = self.__index_in_epoch
        self.__index_in_epoch += batch_size

        # Current epoch is finished (used all examples)
        if self.__index_in_epoch > self.__num_examples:
            self.__epochs_completed += 1

            # reshuffle data for next epoch
            self.__X, self.__y = shuffle(self.__X, self.__y)
            start = 0
            self.__index_in_epoch = batch_size

            # make sure batch size is smaller than the actual number of examples
            assert batch_size <= self.__num_examples
        end = self.__index_in_epoch
        return self.__X[start:end], self.__y[start:end]

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

def find_files_recursively(current_path, current_index, source_file_dict={}, file_extension='.java'):
        # search for source files in current path
        source_files = []
        try:
            _, _, source_files = walk(current_path).__next__()
        except:
            logger.exception('\tCould not iterate through folder {0}.'.format(current_path))

        # filter by extension.
        source_files = [ file for file in source_files if file.endswith(file_extension)]

        # found some files (add them to the source_files dict)
        if len(source_files) > 0:
            print ('\n')
            logger.debug('\tFound {0} source files in {1}:'.format(len(source_files), current_index))
            for f in source_files:                
                # remove extension for index
                fi = f[:-len(file_extension)]
                source_file_dict[current_index + '.' + fi] = current_path + '/' + f
                logger.debug('\t\t- {0}.{1}'.format(current_index, fi))

        # search for new folders
        folder_list = []
        try:
            folder_list = next(walk(current_path))[1]
        except:
            logger.exception('\tCould not iterate through root folder of dataset. - Path: {0}.'.format(current_path))

        for folder in folder_list:
            source_file_dict = find_files_recursively(current_path + folder + '/', current_index + '.' + folder, source_file_dict, file_extension)
        return source_file_dict

def load_bug_data(bug_data_path, class_info_mapping, number_of_bugs_mapping, binary_class_labels):
    logger.debug('Initializing bug data set with parameters {0} - {1}'.format(class_info_mapping, number_of_bugs_mapping))
    if not bug_data_path.endswith('.csv'):
        raise AttributeError('Only .csv files are supported.')
        
    # iterate over entries and add them to the __bug_data list
    # skip the first row (contains only title)
    bug_data_dict = {}
    for entry in get_csv_row_generator(bug_data_path, delimiter=',', skip_first_row=True):
        logger.debug('\tBugs: {1} \t-> Class: {0}'.format(entry[class_info_mapping], entry[number_of_bugs_mapping]))
        
        # precise mapping
        if binary_class_labels:
            bug_data_dict[entry[class_info_mapping]] = 1 if int(entry[number_of_bugs_mapping]) > 0 else 0 # Binary mapping. No Bug: 0 - Bug: 1
        else:
            bug_data_dict[entry[class_info_mapping]] = entry[number_of_bugs_mapping] # Class is number of bugs

    return bug_data_dict
        
def map_bug_data(source_files, bug_data):
        logger.debug('Mapping bug data and source files together.')
        test_data = []
        for class_info, number_of_bugs in bug_data.items():
            if not class_info in source_files:
                logger.error('Could not find match for bug data file {0}.'.format(class_info))
                continue
            test_data.append((class_info, source_files[class_info], number_of_bugs))
        return test_data

class DefectDataSetLoader(object):
    """description of class"""

    def __init__(self, source_root_path_list=[], bug_data_path_list=[], source_files_extension=('.java'), one_hot=True, binary_class_labels=True):
        
        if len(source_root_path_list) == 0 or len(bug_data_path_list) == 0 or len(source_root_path_list) != len(bug_data_path_list):
            raise AttributeError('Parameter source_root_path_list or bug_data_path_list are either empty or do not contain the same number of dirs.')

        self.num_projects = len(source_root_path_list)

        # root path of the java project and location of the bug data sheet
        self.__root_path_list = source_root_path_list
        self.__bug_data_path_list = bug_data_path_list

        # dict which contains the path (value) of a single file /class
        # and the package info (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory)
        # as the index.
        self.__source_files = [{} for _ in range(self.num_projects)]

        # dict which contains the number of bugs of a single file / class
        # and the package info (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory)
        # as the index. 
        self.__bug_data = [{} for _ in range(self.num_projects)]

        # list: [project_test_data]
        # project_test_data: [(class_info, path_to_class_file, number_of_bugs)]
        # class_info: (package info) (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory) 
        self.test_data = [[] for _ in range(self.num_projects)]

        # list that stores the start and end indices for each project for the data set feature vectors.
        # for example: [(0, 500), ...] would mean that features from index 0 to 500 belong to project 0 and features from 501 to X belong to another project.
        # list of tuples. [(start_index, end_index)]
        self.test_data_project_indices = []

        # convert bug number to binary labels (0: No Bug - 1: Bug)
        self.binary_class_labels = binary_class_labels

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

        logger.debug('Initializing {0} source data set(s) with path(s) {1}.'.format(self.num_projects, self.__root_path_list))

        for i in range(self.num_projects):
            print('-- Project {0} --'.format(i))
            logger.debug('Initializing project {0}.'.format(i))
            project_source_path = self.__root_path_list[i]
            project_bug_path = self.__bug_data_path_list[i]

            # append '/' at the end if it does not exist.
            if not project_source_path.endswith('/'):
                project_source_path += '/'

            # iterate over root path and add all files to the source_files dict
            folder_list = []
            try:
                folder_list = next(walk(project_source_path))[1]
            except:
                logger.exception('\tCould not iterate through root folder of dataset. - Path: {0}.'.format(project_source_path))

            if len(folder_list) == 0:
                logger.error('Could not locate project root on path {0}. There were no folders inside the directory.'.format(project_source_path))
                return None
            elif len(folder_list) > 1:
                logger.error('Could not locate project root on path {0}. There were more than one folder in the directory.\n\t\tExpected something like this [org]\n\t\tGot {2}.'.format(project_source_path, folder_list))
                return None
        
            logger.debug('\tFound project root {0}.'.format(folder_list[0]))

            # go over source dir and index source files
            self.__source_files[i] = find_files_recursively(project_source_path + folder_list[0] + '/', folder_list[0], source_file_dict={}, file_extension=self.source_files_extension)
            logger.debug('Finished indexing of source folder for project {0} ({1}). Found {2} files.'.format(i, project_source_path, len(self.__source_files[i])))

            # iterate over bug data
            self.__bug_data[i] = load_bug_data(project_bug_path, class_info_mapping, number_of_bugs_mapping)
            logger.debug('Finished indexing of bug data for project {0} ({1}). Found data for {2} classes.'.format(i, project_bug_path, len(self.__bug_data[i])))

            # map bug data 
            self.test_data[i] = map_bug_data(self.__source_files[i], self.__bug_data[i])
            logger.debug('Finished mapping project {0}.'.format(i))

        # create abstract syntax trees for every file
        self.__create_ast_vectors()

        # set number of samples
        self.__num_examples = len(self.test_data_X)

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

        pickle_this = (self.test_data_X, self.test_data_Y, self.token_mapping_names, self.class_vector, self.one_hot, self.test_data_project_indices)

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
        self.test_data_project_indices = unpickle_this[5]
                
        self.num_classes = self.__get_num_classes()

        logger.debug('Loaded test data. test_X shape: {0} - test_Y shape: {1} - Number of custom tokens: {2}'.format(self.test_data_X.shape, self.test_data_Y.shape, len(self.token_mapping_names)))


    def get_test_train_split(self, test_ratio=0.2, random_seed=42):
        logger.debug('Generating train/test split with test_size {0} and random_state {1}'.format(test_ratio, random_seed))
        X_train, X_test, y_train, y_test = train_test_split(self.test_data_X, self.test_data_Y, test_size=test_ratio, random_state=random_seed)
        logger.debug('Testset Split:')
        logger.debug('\tTrain Shape: {0} - {1}'.format(X_train.shape, y_train.shape))
        logger.debug('\tTest Shape: {0} - {1}'.format(X_test.shape, y_test.shape))
        return X_train, X_test, y_train, y_test

    def get_project_split(self):
        logger.debug('Splitting data set vector into projects.')
        X = []
        y = []

        for project_index in range(len(self.test_data_project_indices)):
            start = self.test_data_project_indices[project_index][0]
            end = self.test_data_project_indices[project_index][1]
            X.append(self.test_data_X[start:end])
            y.append(self.test_data_Y[start:end])
            logger.debug('\tProject {0} Shape: X: {1} - y: {2}'.format(project_index, X[project_index].shape, y[project_index].shape))

        return X, y

                     
    def __create_ast_vectors(self):
        """
        Creates abstract syntax trees for each class in each project.
        Tokens will be reused throughout all projects.
        """
        # get the total number of classes for all projects
        number_of_classes = sum([len(self.test_data[i]) for i in range(self.num_projects)])
        current_data_set_index = 0
        for project_index in range(self.num_projects):
            project_test_data = self.test_data[project_index]

            # start index for the test_data_X / Y which contains all features for every project.
            start_index = current_data_set_index

            # counter for self.test_data index. This index starts at zero for each project because the projects are sep. in test_data.
            project_test_data_index = 0
            
            logger.debug('Creating abstract syntax trees for project {0}. {1} classes.'.format(project_index, len(project_test_data)))
            for (class_info, path_to_class_file, number_of_bugs) in project_test_data:
                # open file and read it
                source_code = ''
                with open(path_to_class_file, 'rb') as f:
                    source_code = f.read()

                try:
                    tree = javalang.parse.parse(source_code)
                except:
                    logger.exception('Could not parse sourcefile {0} (Path: {1}) (Project {2}). (Syntax errors)'.format(class_info, path_to_class_file, project_index))
                    continue           

                # try to generate feature vector
                tree_feature_vector = self.__convert_tree_to_feature_vector_unstructured(tree)
                self.test_data_X.append(tree_feature_vector)
                self.test_data_Y.append(number_of_bugs)

                # replace existing test_data entry tuples with additional info
                self.test_data[project_index][project_test_data_index] = (class_info, path_to_class_file, number_of_bugs, tree, tree_feature_vector)           

                project_test_data_index += 1
                current_data_set_index += 1
                utils.show_progress(True, current_data_set_index, number_of_classes, 'AST creation for {0} classes.\tToken mappings: {1}:', number_of_classes, len(self.token_mapping_names))
            print('\n')
            logger.debug('AST creation for project {0} done. Progress: {1:.2f}%'.format(project_index, ((current_data_set_index / number_of_classes) * 100)))

            end_index = current_data_set_index - 1
            logger.debug('Data set index interval for project {0}: {1} to {2}'.format(project_index, start_index, end_index))
            self.test_data_project_indices.append((start_index, end_index))
            print('')
        print('\n**')


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


    def __prepare_data(self, rare_token_number=5):
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
        logger.debug('Filtered tokens: {0} -> Number of tokens after filtering: {1}'.format(len(filter), len(tokens) - len(filter)))

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
                 one_hot):

        self.__X = features
        self.__y = targets

        self.__epochs_completed = 0
        self.__index_in_epoch = 0

        # will be set in initialize 
        self.__num_examples = targets.shape[0]

        self.one_hot = one_hot
        

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

    @property
    def most_frequent_class(self):
        counts = np.bincount(self.__y)
        return np.argmax(counts), np.max(counts)

    @property
    def zero_error(self):
        # how many occurences has the most frequent class?
        _, fc = self.most_frequent_class
        return fc / self.__num_examples

    def get_random_elements(self, num_elements=1, seed=42):
        if not seed is None:
            np.random.seed(seed=seed)
        
        # get indices of elements to sample
        indices = np.random.choice(np.arange(self.__num_examples), size=num_elements, replace=False)

        X_sample = self.__X[indices]
        y_sample = self.__y[indices]

        return X_sample, y_sample


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

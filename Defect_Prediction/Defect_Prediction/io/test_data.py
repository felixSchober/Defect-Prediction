from os import walk, remove
import os.path as osPath
import logging
from io.csv_data import get_csv_row_generator
import javalang

logger = logging.getLogger('io')

class TestData(object):
    """description of class"""

    def __init__(self, source_root_path, bug_data_path, source_files_extension=('.java')):
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

        self.source_files_extension = source_files_extension

    def initialize(self, class_info_mapping, number_of_bugs_mapping):

        logger.info('Initializing source data set with path {0}.'.format(self.__root_path))

        # Append '/' at the end if it does not exist.
        if not self.__root_path.endswith("/"):
            self.__root_path += "/"

        # iterate over root path and add all files to the source_files dict
        folder_list = []
        try:
            folder_list = next(walk(self.__root_path))[1]
        except:
            logger.exception("\tCould not iterate through root folder of dataset. - Path: {0}.".format(self.__root_path))

        if len(folder_list) == 0:
            logger.error('Could not locate project root on path {0}. There were no folders inside the directory.'.format(self.__root_path))
            return None
        elif len(folder_list) > 1:
            logger.error('Could not locate project root on path {0}. There were more than one folder in the directory.\n\t\tExpected something like this [org]\n\t\tGot {2}.'.format(self.__root_path, folder_list))
            return None
        
        logger.info('\tFound project root {0}.'.format(folder_list[0]))

        # Go over source dir and index source files
        self.find_files_recursively(self.__root_path + folder_list[0] + '/', folder_list[0])
        logger.info('Finished indexing of source folder. Found {0} files.'.format(len(self.__source_files)))

        # Iterate over bug data
        self.load_bug_data(class_info_mapping, number_of_bugs_mapping)
        logger.info('Finished indexing of bug data. Found data for {0} classes.'.format(len(self.__bug_data)))

        # map bug data 
        self.map_bug_data()
        logger.info('Finished mapping.')

        
    def find_files_recursively(self, current_path, current_index):
        # search for source files in current path
        source_files = []
        try:
            _, _, source_files = walk(current_path).next()
        except:
            logger.exception("\tCould not iterate through folder {0}.".format(self.get_root_path() + className))

        # filter by extension.
        source_files = [ file for file in source_files if file.endswith(self.source_files_extension)]

        # found some files (add them to the source_files dict)
        if len(source_files) > 0:
            logger.info('\n\tFound {0} source files in {1}:'.format(len(source_files), current_index))
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
            logger.exception("\tCould not iterate through root folder of dataset. - Path: {0}.".format(self.__root_path))

        for folder in folder_list:
            self.find_files_recursively(current_path + folder + '/', current_index + '.' + folder)


    def load_bug_data(self, class_info_mapping, number_of_bugs_mapping):
        logger.info('Initializing bug data set with parameters {0} - {1}'.format(class_info_mapping, number_of_bugs_mapping))
        if not self.__bug_data_path.endswith('.csv'):
            raise AttributeError('Only .csv files are supported.')
        
        # iterate over entries and add them to the __bug_data list
        # skip the first row (contains only title)
        for entry in get_csv_row_generator(self.__bug_data_path, delimiter=';', skip_first_row=True):
            logger.debug('\tBugs: {1} \t-> Class: {0}'.format(entry[class_info_mapping], entry[number_of_bugs_mapping]))
            self.__bug_data[entry[class_info_mapping]] = entry[number_of_bugs_mapping]


    def map_bug_data(self):
        logger.info('Mapping bug data and source files.')
        for class_info, number_of_bugs in self.__bug_data.items():
            if not class_info in self.__source_files:
                logger.error('Could not find match for bug data file {0}.'.format(class_info))
                continue
            self.test_data.append((class_info, self.__source_files[class_info], number_of_bugs))

             
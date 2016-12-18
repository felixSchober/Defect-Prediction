from os import walk, remove
import os.path as osPath
import logging

logger = logging.getLogger('io')

class TestData(object):
    """description of class"""

    def __init__(self, root_path, source_files_extension=('.java')):
        # root path of the java project
        self.__root_path = root_path

        # dict which contains the path (value) of a single file
        # and the package info (e.g. org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory)
        # as the index
        self.source_files = {}

        self.source_files_extension = source_files_extension

    def initialize(self):

        logger.info('Initializing data set with path {0}.'.format(self.__root_path))

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
        self.find_files_recursively(self.__root_path + folder_list[0] + '/', folder_list[0])
        logger.info('Indexing of source folder is finished. Found {0} files.'.format(len(self.source_files)))
        
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
                self.source_files[current_index + '.' + fi] = current_path + '/' + f
                logger.debug('\t\t- {0}.{1}'.format(current_index, fi))

        # search for new folders
        folder_list = []
        try:
            folder_list = next(walk(current_path))[1]
        except:
            logger.exception("\tCould not iterate through root folder of dataset. - Path: {0}.".format(self.__root_path))

        for folder in folder_list:
            self.find_files_recursively(current_path + folder + '/', current_index + '.' + folder)

    def load_bug_data(self, path):









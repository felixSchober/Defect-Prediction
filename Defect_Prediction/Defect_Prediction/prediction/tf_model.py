# python imports
from __future__ import division
from math import sqrt
import time
import os
import os.path
import sys
import logging
from enum import Enum
import errno

# lib imports
import tensorflow as tf
import numpy as np
import colorama

#project imports
from helper import colored_shell_seq, create_dir_if_necessary, TF_LAYER, get_unique_layer_name, tensor_shape_to_list, check_if_dir_exists


logger = logging.getLogger('prediction')

# TF Constants

TF_RANDOM_SEED = 42

TF_LOG_CREATE_SUB_DIR = True
TF_LOG_DIR = os.path.join(os.getcwd(), 'log', 'tensorflow')
SUMMARY_EVERY_X_EPOCHS = 1

TF_CONV2D_PADDING_DEFAULT = 'SAME'
TF_MAXPOOLING_PADDING_DEFAULT = 'SAME'
TF_NORM_DR_DEFAULT = 5
TF_NORM_BIAS_DEFAULT = 1.0
TF_NORM_ALPHA = 1.0
TF_NORM_BETA = 0.5

def get_weights_variable(dim_x, dim_y, std_dev=1.0, name='weights'):
    weights = tf.Variable(
        tf.truncated_normal([dim_x, dim_y], stddev=std_dev / sqrt(float(dim_x))),
        name=name)
    return weights


def get_bias_variable(num_neurons, initial_value=0.1, name='biases'):
    biases = tf.Variable(tf.constant(initial_value, shape=[num_neurons]), name=name)
    return biases


def get_placeholders(X_feature_vector_length, batch_size, num_classes):
    input_features_placeholders = tf.placeholder(tf.float32, shape=(None, *X_feature_vector_length), name='x-input')
    targets_placeholder = tf.placeholder(tf.int32, shape=(None), name='y-input')
    keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout-placeholder')
    return input_features_placeholders, targets_placeholder, keep_prob_placeholder


def fill_feed_dict(data_set, features_placeholders, targets_placeholders, keep_prob_placeholder, keep_prob, batch_size, shuffle=True, reshape_into=None):
    try:
        features_feed, targets_feed = data_set.next_batch(batch_size, shuffle=shuffle)
    except:
        # in case this is a tf dataset
        features_feed, targets_feed = data_set.next_batch(batch_size)

    # reshape if necessary
    if reshape_into is not None:
        features_feed = features_feed.reshape(reshape_into)

    feed_dict = {
        features_placeholders: features_feed,
        targets_placeholders: targets_feed,
        keep_prob_placeholder: keep_prob
    }
    return feed_dict


def get_dense_layer(input_tensor, input_dimension, output_dimension, name=None, create_summary=True):
    if name is None:
        name = get_unique_layer_name(TF_LAYER.Dense)

    # Layer:
    with tf.variable_scope(name) as scope:
        weights = get_weights_variable(input_dimension[-1], output_dimension)            
        biases = get_bias_variable(output_dimension)

        # pre activate layer with bias
        with tf.name_scope('Wx_plus_bias'):
            preactivation = tf.add(tf.matmul(input_tensor, weights), biases, name='preactivation_add_op')

        layer_tensor = tf.nn.relu(preactivation, name=name + '_relu')
        if create_summary:
            activation_summary(layer_tensor)
    return layer_tensor

def get_dropout_layer(input_tensor, keep_prob_pl, name=None):
    if name is None:
        name = get_unique_layer_name(TF_LAYER.Dropout)

    tf.summary.scalar(name, keep_prob_pl)
    # Dropout:
    with tf.variable_scope(name) as scope:
        dropout_tensor = tf.nn.dropout(input_tensor, keep_prob_pl)
    return dropout_tensor

def get_convolutional_layer(input_tensor, kernel_shape, strides, padding, use_gpu=True, name=None, create_summary=True):
    """Computes a 2-D convolution given 4-D input and filter tensors.

        Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel shape of [filter_height, filter_width, in_channels, out_channels], this op performs the following:
        1.Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
        2.Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
        3.For each patch, right-multiplies the filter matrix and the image patch vector.


        Args:
            input_tensor: A Tensor. Must be one of the following types: half, float32, float64.
            input_shape: Shape of the input tensor
            kernel_shape: Shape of the kernel. List of ints. 1-D of length 4. Example: [5, 5, 3, 64]
            strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input. Must be in the same order as the dimension specified with format.
            padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
            use_cudnn_on_gpu: An optional bool. Defaults to True.
            name: A name for the operation (optional).
            create_summary: create activation summary for layer

        Returns:
            Layer Tensor.
        """

    if name is None:
        name = get_unique_layer_name(TF_LAYER.Convolution2D)
    with tf.variable_scope(name) as scope:
        # TODO: add weight decay

        kernel_tensor = tf.Variable(
            tf.truncated_normal(kernel_shape, mean=0.0, stddev=0.1),
            name=name + '_kernel')
        conv = tf.nn.conv2d(input_tensor,
                            kernel_tensor,
                            strides,
                            padding,
                            use_gpu,
                            name=name + '_conv')
        biases = get_bias_variable(kernel_shape[-1]) # this would be out_channels
        convolutional_tensor = tf.nn.relu(conv, name=scope.name)
        
        if create_summary:
            # convert the kernels to a grid form to save space in Tensorboard
            grid = put_kernels_on_grid(kernel_tensor)
            image_summary(grid, name, 1)
    return convolutional_tensor


def get_max_pooling_layer(input_tensor, kernel_shape, strides, padding, name=None):
    """Performs max pooling on the input.

    Args:
        input_tensor: A Tensor. Must be one of the following types: half, float32, float64.
        kernel_shape: Shape of the kernel. List of ints. 1-D of length 4. Example: [1, 3, 3, 1]
        strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input. Must be in the same order as the dimension specified with format. Example: [1, 2, 2, 1]
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.

    Returns:
        Layer Tensor.
    """

    if name is None:
        name = get_unique_layer_name(TF_LAYER.MaxPooling)

    pooling = tf.nn.max_pool(input_tensor, kernel_shape, strides, padding, name=name)
    return pooling


def get_normalization_layer(input_tensor, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=None):
    """Performs Local Response Normalization on the input.
    The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently. Within a given vector, each component is divided by the weighted, squared sum of inputs within depth_radius.

    Args:
        input_tensor: A Tensor. Must be one of the following types: half, float32, float64.
        depth_radius: An optional int. Defaults to 5. 0-D. Half-width of the 1-D normalization window.
        bias: An optional float. Defaults to 1. An offset (usually positive to avoid dividing by 0).
        alpha: An optional float. Defaults to 1. A scale factor, usually positive.
        beta: An optional float. Defaults to 0.5. An exponent.

    Returns:
        Layer Tensor.
    """
    if name is None:
        name = get_unique_layer_name(TF_LAYER.Normalization)

    norm = tf.nn.local_response_normalization(input_tensor, depth_radius, bias, alpha, beta, name)
    return norm


def inference(input_shape, output_shape, architecture, features_pl, keep_prob_pl):
        """Build the defect prediction model.

        Args:
            input_shape: Length of the feature vector
            output_shape: num_classes = num_neurons in the last layer
            architecture: List of layers. Format: [(TF_LAYER.Dense|Dropout, SIZE_OF_LAYER, NAME)]
            features_pl: feature placeholder
            keep_prob_pl: dropout keep placeholder

        Returns:
            Logits Tensor.
        """
        
        # Hidden Layers
        predecessor_tensor = features_pl
        predecessor_shape = tensor_shape_to_list(features_pl.get_shape())
        for type, name, parameters in architecture:
            if type == TF_LAYER.Dense:
                logger.info('\t\tDense-Layer {0}:'.format(name))
                logger.info('\t\t\t--> Input: {0}'.format(predecessor_shape))
                # Do we need to change the input dimension?
                if len(predecessor_shape) > 2:
                    predecessor_shape = predecessor_shape[1] * predecessor_shape[2] * predecessor_shape[3]
                    predecessor_tensor = tf.reshape(predecessor_tensor, [-1, predecessor_shape])
                    logger.info('\t\t\t--> Input (reshaped): {0}'.format(tensor_shape_to_list(predecessor_tensor.get_shape())))

                layer_tensor = get_dense_layer(predecessor_tensor, predecessor_shape, parameters, name)
                predecessor_shape = tensor_shape_to_list(layer_tensor.get_shape())
                logger.info('\t\t\t<-- Output: {0}'.format(predecessor_shape))

            elif type == TF_LAYER.Dropout:
                logger.info('\t\tDropout {0}:'.format(name))

                # TODO: Variable dropout
                logger.warning('Dropout Keep probability is not implemented yet. All dropout layers default to 0.5.')
                if keep_prob_pl is None:
                    raise AttributeError('Model contains dropout layer but keep_prob placeholder is not specified.')
                layer_tensor = get_dropout_layer(predecessor_tensor, keep_prob_pl, name)

            elif type == TF_LAYER.Convolution2D:
                if parameters[2] is None:
                    padding = TF_CONV2D_PADDING_DEFAULT
                else:
                    padding = parameters[2]
                logger.info('\t\tConv2D-Layer {0}:'.format(name))
                logger.info('\t\t\tKernel: {0} - Stride: {1} - Padding: {2}'.format(parameters[0], parameters[1], padding))
                logger.info('\t\t\t--> Input: {0}'.format(predecessor_shape))
                
                layer_tensor = get_convolutional_layer(predecessor_tensor, parameters[0], parameters[1], padding, name=name, create_summary=False)
                predecessor_shape = tensor_shape_to_list(layer_tensor.get_shape())

                logger.info('\t\t\t<-- Output: {0}'.format(predecessor_shape))


            elif type == TF_LAYER.MaxPooling:

                if parameters[2] is None:
                    padding = TF_CONV2D_PADDING_DEFAULT
                else:
                    padding = parameters[2]

                logger.info('\t\tMax Pooling-Layer {0}:'.format(name))
                logger.info('\t\t\tKernel: {0} - Stride: {1} - Padding: {2}'.format(parameters[0], parameters[1], padding))

                layer_tensor = get_max_pooling_layer(predecessor_tensor, parameters[0], parameters[1], padding, name=name)
                logger.info('\t\t\t--> Input: {0}'.format(predecessor_shape))
                predecessor_shape = tensor_shape_to_list(layer_tensor.get_shape())
                logger.info('\t\t\t<-- Output: {0}'.format(predecessor_shape))

            elif type == TF_LAYER.Normalization:
                if parameters[0] is None:
                    p1 = TF_NORM_DR_DEFAULT
                else:
                    p1 = parameters[0]
                if parameters[1] is None:
                    p2 = TF_NORM_BIAS_DEFAULT
                else:
                    p2 = parameters[1]
                if parameters[2] is None:
                    p3 = TF_NORM_ALPHA
                else:
                    p3 = parameters[2]
                if parameters[3] is None:
                    p4 = TF_NORM_BETA
                else:
                    p4 = parameters[2]
                logger.info('\t\tNormalization-Layer {0}:'.format(name))
                layer_tensor = get_normalization_layer(predecessor_tensor, p1, p2, p3, p4, name)
            predecessor_tensor = layer_tensor    
                    
        # Output:
        with tf.variable_scope('softmax_linear') as scope:
            logger.info('\t\tOutput-Layer softmax_linear:')
            logger.info('\t\t\t--> Input: {0}'.format(predecessor_shape))
            
            weights = get_weights_variable(predecessor_shape[1], output_shape)
            biases = get_bias_variable(output_shape)
            logit_tensor = tf.add(tf.matmul(predecessor_tensor, weights), biases, name=scope.name)
            activation_summary(logit_tensor)
            logger.info('\t\t\t<-- Output: {0}'.format(tensor_shape_to_list(logit_tensor.get_shape())))
        return logit_tensor


def loss(logit_tensor, targets_pl):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Targets Placeholder. 1-D tensor of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    targets = tf.to_int64(targets_pl)

    # calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_tensor, targets, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    tf.summary.scalar('loss', cross_entropy_mean)
    return cross_entropy_mean


def evaluate(logit_tensor, targets_pl):
    correct = tf.nn.in_top_k(logit_tensor, targets_pl, 1)

    # get the number of correct entries
    correct_sum = tf.reduce_sum(tf.cast(correct, tf.int32))
    return correct_sum

def accuracy(logits, targets_pl, one_hot=False):
    targets = tf.to_int64(targets_pl)
    
    if one_hot:
        # compare the indices of the outputs. For a correct prediction they should be the same
        correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(targets, 1), name='accuracy_equals_oh')
    else:
        # compare the indices of the outputs with the correct label which is a number here.
        correct_prediction = tf.equal(tf.arg_max(logits, 1), targets, name='accuracy_equals')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'), name='accuracy_mean')
    tf.summary.scalar('accuracy_mean', accuracy)
    return accuracy

def f1_score(logits, targets_pl, one_hot=False):
    targets = tf.to_int64(targets_pl)
    
    y_predicted = tf.arg_max(logits, 1)
    if one_hot:
        y_true = tf.arg_max(targets, 1)
    else:
        y_true = logits

    # get true positives (by multiplying the predicted and actual labels we will only get a 1 if both labels are 1)
    tp = tf.count_nonzero(y_predicted * y_true)

    # get true negatives (basically the same as tp only the inverse)
    tn = tf.count_nonzero((y_predicted - 1) * (y_true - 1)) 

    fp = tf.count_nonzero(y_predicted * (y_true - 1))
    fn = tf.count_nonzero((y_predicted - 1) * y_true)

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)
    tf.summary.scalar('f1-score', f1_score)

    f1_score = tf.reduce_mean(tf.cast(f1_score, 'float32'), name='f1_score_reduce_mean')
    return f1_score




def logistic_regression_inference(input_size, output_size, features_pl):
    W = tf.Variable(tf.zeros([input_size, output_size]))
    b = tf.Variable(tf.zeros([output_size]))

    logit = tf.nn.softmax(tf.matmul(features_pl, W) + b)
    activation_summary(logit)
    return logit

def loss_logistic_regression(logit_tensor, targets_pl):
    targets = tf.to_int64(targets_pl)

    cost = tf.reduce_mean(-tf.reduce_sum(targets_pl*tf.log(logit_tensor), reduction_indices=1))
    tf.summary.scalar('cost', cost)
    return cost

def validate_architecture(architecture):
    # For documentation see https://github.com/Jorba123/tf_net/blob/master/README.md

    if architecture is None:
        raise AttributeError('Architecture is None.')
    if len(architecture) == 0:
        raise AttributeError('Architecture does not contain a layer.')

    # validate every layer
    names = []
    for layer in architecture:

        # Layer needs at least 3 arguments
        if len(layer) < 3:
            raise AttributeError('Architecture contains invalid entries.')

        type, name, parameters = layer

        # check if name is unique
        if name == '':
            raise AttributeError('Architecture contains a layer without a name.')

        if name in names:
            raise AttributeError('Architecture contains the same name for two different layers.')
        names.append(name)

        # check type and parameters
        if type == TF_LAYER.Dense:
            try:
                parameters += 1
            except:
                raise AttributeError('DENSE: Architecture contains invalid layer parameters. -> {0}'.format(parameters))

        elif type == TF_LAYER.Dropout:
            if float(parameters) != parameters:
                raise AttributeError('DROPOUT: Architecture contains invalid layer parameters. -> {0}'.format(parameters))

        elif type == TF_LAYER.Convolution2D:
            if len(parameters) != 3:
                raise AttributeError('CONV2D: Architecture contains invalid layer parameters. -> {0}'.format(parameters))
            # check kernel size
            if len(parameters[0]) != 4:
                raise AttributeError('Conv2D layer {0} contains a kernel size with length {1}'.format(name, len(parameters[0])))
            # check strides size
            if len(parameters[1]) != 4:
                raise AttributeError('Conv2D layer {0} contains a stride size with length {1}'.format(name, len(parameters[0])))

        elif type == TF_LAYER.MaxPooling:
            if len(parameters) != 3:
                raise AttributeError('MAX_POOLING: Architecture contains invalid layer parameters. -> {0}'.format(parameters))

            # check kernel size
            if len(parameters[0]) != 4:
                raise AttributeError('MaxPool layer {0} contains a kernel size with length {1}'.format(name, len(parameters[0])))
            # check strides size
            if len(parameters[1]) != 4:
                raise AttributeError('MaxPool layer {0} contains a stride size with length {1}'.format(name, len(parameters[0])))

        elif type == TF_LAYER.Normalization:
            if len(parameters) != 4:
                raise AttributeError('NORM: Architecture contains invalid layer parameters. -> {0}'.format(parameters))

        else:
            raise AttributeError('Architecture contains an invalid layer type.')
                   

        
         

class TensorFlowNet(object):
    """description of class"""

    def __init__(self,
                train_data_set, 
                test_data_set, 
                num_classes, 
                input_shape,  
                targets_shape,              
                input_is_image,
                batch_size, 
                reshape_input_to=None,
                initial_learning_rate=0.1, 
                architecture_shape=[(TF_LAYER.Dense, 1024, 'hidden1'), (TF_LAYER.Dense, 64, 'hidden2'), (TF_LAYER.Dropout, 0.6, 'dropout1')], 
                log_dir=TF_LOG_DIR, 
                max_epochs=500,
                num_epochs_per_decay=150,
                learning_rate_decay_factor=0.1,
                model_name=str(int(time.time())),
                early_stopping_epochs = 100,
                calculate_f1_score=False):
        self.sess = None
        self.saver = None
        self.input_shape = input_shape
        self.train = train_data_set
        self.test = test_data_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.reshape_input_to = reshape_input_to

        self.targets_shape = targets_shape
        # [batch_size, num_classes] or [-1, num_classes]
        if len(targets_shape) == 2:
            self.one_hot = True
        # [batch_size] or [-1]
        elif len(targets_shape) == 1:
            self.one_hot = False
        
        if num_classes > 2 and calculate_f1_score:
            raise Exception('F1 score can only be calculated with binary classification.')

        self.calculate_f1_score = calculate_f1_score

        validate_architecture(architecture_shape)
        self.model_architecture = architecture_shape

        self.max_epochs = max_epochs
        self.steps_per_epoch = train_data_set.num_examples // self.batch_size
        self.global_step = 0

        self.input_is_image = input_is_image

        self.initial_learning_rate = initial_learning_rate

        # epochs after which learning rate decays.
        self.num_epochs_per_decay = num_epochs_per_decay
        

        # how much does the learning rate decay after num_epochs_per_decay
        self.learning_rate_decay_factor = learning_rate_decay_factor

        # the last layer of the net used after training for prediction
        self.model = None 

        # Inputs placeholder
        self.features_pl = None

        self.best_train_loss = np.inf
        self.best_train_precission = 0
        self.best_test_loss = np.inf
        self.best_test_precission = 0
        self.best_test_f1 = 0
        self.best_train_f1 = 0

        self.last_test_loss_improvement = 0
        self.early_stopping_epochs = early_stopping_epochs

        # check if model_name dir already exists

        self.model_name = model_name
        self.log_dir = os.path.join(log_dir, model_name)
        if check_if_dir_exists(self.log_dir):
            self.model_name += str(int(time.time()))
            logger.warning('Tensorflow model dir with name {0} already exits. Renaming model to {1}'.format(model_name, self.model_name))
            self.log_dir = os.path.join(log_dir, self.model_name)

        if TF_LOG_CREATE_SUB_DIR:            
            create_dir_if_necessary(self.log_dir)
        else:
            self.log_dir = log_dir 
        colorama.init()
        

    
    def get_train_op(self, loss_tensor, global_step):

        # decay learning rate based on the number of steps (global_step)
        num_batches_per_epoch = self.train.num_examples / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='learning_rate_decay')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        # apply the gradients to minimize loss.
        # for each minimization increment global_step counter
        train_op = optimizer.minimize(loss_tensor, global_step=global_step)

        # add histogram for each trainable variable
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return train_op

    def get_train_op_logistic_regression(self, loss_tensor, global_step):
        num_batches_per_epoch = self.train.num_examples / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='learning_rate_decay')
        tf.summary.scalar('learning_rate', learning_rate)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_tensor)
        return train_op

    

    def do_eval(self, eval_correct_tensor, features_pl, targets_pl, keep_prob_pl, data_set):

        # number of correct predictions
        true_count = 0
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in range(steps_per_epoch):
            feed_dict = fill_feed_dict(data_set, features_pl, targets_pl, keep_prob_pl, 1.0, self.batch_size)
            true_count += self.sess.run(eval_correct_tensor, feed_dict=feed_dict)
        precision = true_count / num_examples
        return num_examples, true_count, precision
       

    def run_training(self):
        logger.info('Building NN model. Attributes:')
        logger.info('\tTrain Samples: {0}'.format(self.train.num_examples))
        logger.info('\tTest Samples: {0}'.format(self.test.num_examples))
        logger.info('\tReshape input to {0}'.format(self.reshape_input_to))
        logger.info('\tTargets Shape {0}'.format(self.targets_shape))
        logger.info('\tOne-Hot Targets {0}'.format(self.one_hot))
        logger.info('\tF1-Score {0}'.format(self.calculate_f1_score))

        try:
            logger.info('\tTrain Zero Error: {0}'.format(self.train.zero_error))
            logger.info('\tTest Zero Error: {0}'.format(self.test.zero_error))
        except:
            logger.info('\tTrain Zero Error: ?')
            logger.info('\tTest Zero Error: ?')
        logger.info('\tBatch Size: {0}'.format(self.batch_size))
        logger.info('\tMax Epochs: {0}'.format(self.max_epochs))
        logger.info('\tNum Classes: {0}'.format(self.num_classes))
        logger.info('\tInitial Learning rate: {0}'.format(self.initial_learning_rate))
        
        # print model architecture
        logger.debug('Network architecture')
        #for type, name, parameters in self.model_architecture:
        #    logger.info('\t\t{0} {1}\t: {2}'.format(type, name, parameters))


        with tf.Graph().as_default():

            tf.set_random_seed(TF_RANDOM_SEED)


            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

            # Generate the input placeholders
            features_pl, targets_pl, keep_prob_pl = get_placeholders(self.input_shape, self.batch_size, self.num_classes)

            #if self.reshape_input_to is not None:
                #features_pl = tf.reshape(features_pl, self.reshape_input_to, name='Input-reshaping')

            # if targets come as a one-hot vector convert the targets tensor to one-hot tensor with shape (None, num_classes)
            # on_value and off_value describe the on and off state (when is a feature 'hot')
            if self.one_hot:
                targets_pl = tf.one_hot(tf.cast(targets_pl, tf.int32, name='targets_pl_cast'), self.num_classes, on_value=1, off_value=0, name='targets_pl_one_hot_conv')

            self.features_pl = features_pl
            self.keep_prob_pl = keep_prob_pl

            # build the model
            try:
                logit_tensor = inference(self.input_shape, self.num_classes, self.model_architecture, features_pl, keep_prob_pl)
            except Exception as e:
                logger.exception('Could not create model.')
                raise e
            logger.info('Model was successfully built. Initializing tensorboard logging and training operations.')
            self.model = logit_tensor

            # add loss tensor to graph
            loss_tensor = loss(logit_tensor, targets_pl)


            # create gradient training op
            train_op = self.get_train_op(loss_tensor, global_step_tensor)

            # add evaluation op for the training and test set.
            eval_correct = evaluate(logit_tensor, targets_pl)

            if self.calculate_f1_score:
                f1_score_tensor = f1_score(logit_tensor, targets_pl, one_hot=self.one_hot) 
            accuracy_tensor = accuracy(logit_tensor, targets_pl, self.one_hot)

            # Build the summary Tensor based on the TF collection of Summaries.
            summary_tensor = tf.summary.merge_all()

            # add variables initializer
            init = tf.global_variables_initializer()

            # initialize model saver
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            self.sess = tf.Session()

            # initialize a SummaryWriter which writes a log file
            summary_writer_train = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.sess.graph)
            summary_writer_test = tf.summary.FileWriter(os.path.join(self.log_dir, 'test'))

            # initialize variables
            self.sess.run(init)

            logger.info('Neural Net is initialized and ready to train.')
            print('\n')
            logger.debug('Step (/100)\tLoss\tDuration') 

            if self.calculate_f1_score:
                print('Epoch\tTrain Loss\tTrain accuracy\tTrain F1\t\tTest Loss\tTest accuracy\tTest F1\tDuration')   
            else:
                print('Epoch\tTrain Loss\tTrain accuracy\t\tTest Loss\tTest accuracy\tDuration')   
            print('=====================================================================================================')      

            # start training    
            start_time = time.time()
            average_train_loss = 0  
            early_stopping = False      
            test_feed_dict = fill_feed_dict(
                self.test, 
                features_pl, 
                targets_pl, 
                keep_prob_pl, 
                keep_prob=1.0, 
                batch_size=self.test.num_examples, 
                shuffle=False, 
                reshape_into=self.reshape_input_to)

            for epoch in range(self.max_epochs):                

                for step in range(self.steps_per_epoch):
                    # fill feed dict with batch
                    train_feed_dict = fill_feed_dict(
                        self.train, 
                        features_pl, 
                        targets_pl, 
                        keep_prob_pl, 
                        keep_prob=0.6, 
                        batch_size=self.batch_size, 
                        reshape_into=self.reshape_input_to
                        )

                    # run the model
                    # _: result of train_op (is None)
                    # loss_value: result of loss operation (the actual loss)
                    train_loss_value = -1
                    try:                    
                        _, train_loss_value = self.sess.run(
                            [train_op, loss_tensor],
                            feed_dict=train_feed_dict)
                    except:
                        logger.exception('Could not run train epoch {0} step {1}. Loss Value: {2}'.format(epoch, self.global_step, train_loss_value))

                    assert not np.isnan(train_loss_value), 'Model diverged with loss = NaN'
                    average_train_loss += train_loss_value
                    self.global_step += 1

                    summary_str_train = self.sess.run(summary_tensor, feed_dict=train_feed_dict)
                    summary_str_test = self.sess.run(summary_tensor, feed_dict=test_feed_dict)

                    summary_writer_train.add_summary(summary_str_train, self.global_step)
                    summary_writer_train.flush()
                    summary_writer_test.add_summary(summary_str_test, self.global_step)
                    summary_writer_test.flush()

                # Write summaries SUMMARY_EVERY_X_EPOCHS.
                if epoch % SUMMARY_EVERY_X_EPOCHS == 0:
                    duration = time.time() - start_time
                    start_time = time.time()    


                    # compute detailed stats                    
                    train_feed_dict = fill_feed_dict(
                        self.train, 
                        features_pl, 
                        targets_pl, 
                        keep_prob_pl, 
                        keep_prob=1.0, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        reshape_into=self.reshape_input_to)

                    # don't take the average in the first step
                    if epoch > 0:
                        average_train_loss /= (self.steps_per_epoch * SUMMARY_EVERY_X_EPOCHS)

                    try:
                        train_f1_score = test_f1_score = -1
                        if self.calculate_f1_score:
                            train_accuracy_value, train_f1_score = self.sess.run([accuracy_tensor, f1_score_tensor], feed_dict=train_feed_dict)
                            test_accuracy_value, test_loss_value, test_f1_score = self.sess.run([accuracy_tensor, loss_tensor, f1_score_tensor], feed_dict=test_feed_dict)
                        else:
                            train_accuracy_value = self.sess.run([accuracy_tensor], feed_dict=train_feed_dict)[0]
                            test_accuracy_value, test_loss_value = self.sess.run([accuracy_tensor, loss_tensor], feed_dict=test_feed_dict)
                    except:
                        logger.exception('Could not compute train- and test accuracy values in epoch {0}, step {1}.'.format(epoch, self.global_step))
                        train_accuracy_value = test_accuracy_value = -1
                        train_num_examples, train_true_count, train_precision = self.do_eval(eval_correct, features_pl, targets_pl, keep_prob_pl, self.train)
                        test_num_examples, test_true_count, test_precision = self.do_eval(eval_correct, features_pl, targets_pl, keep_prob_pl, self.test)

                        logger.debug('Train: Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(train_num_examples, train_true_count, train_precision))
                        logger.debug('Test: Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(test_num_examples, test_true_count, test_precision))
                    logger.debug('{0}\t\t{1:.4f}\t{2:.5f}'.format(epoch, train_loss_value, duration))                  
                                        
                    # if early stopping is True abort training and write a last summary
                    early_stopping = self.early_stopping(epoch, test_loss_value)

                    
                    # only save checkpoint if the test loss improved
                    if test_accuracy_value > self.best_test_precission:
                        checkpoint_file = os.path.join(self.log_dir, 'model')                    
                        try:
                            self.save_model(checkpoint_file, step)
                        except:
                            logger.exception('Could not save model.')

                    self.print_step_summary_and_update_best_values(epoch, average_train_loss, train_accuracy_value, test_loss_value, test_accuracy_value, duration, train_f1_score, test_f1_score, colored=True)
                    
                    average_train_loss = 0

                    if early_stopping:
                        print('-----\n\n')
                        logger.info('Early stopping after {0} steps.'.format(epoch))                    
                        break

            logger.info('Training complete.')
            logger.info('Restoring best model.') 
            

            # Restore best model
            ckpt = tf.train.get_checkpoint_state(os.path.join(self.log_dir, 'model'))
            self.saver.restore(self.sess, os.path.join(self.log_dir, 'model'))
            #if ckpt and ckpt.model_checkpoint_path:
            #    print('..')
            #else:
            #    logger.error('Could not restore model. No model found.')

            logger.info('Best Losses: Train {0:.5f} - Test: {1:.5f}'.format(self.best_train_loss, self.best_test_loss)) 
            logger.info('Best Precisions: Train {0:.5f} - Test: {1:.5f}'.format(self.best_train_precission, self.best_test_precission)) 


    def model_improved(self, test_loss_value):
        return test_loss_value < self.best_test_loss

    def early_stopping(self, epoch, test_loss):
        if not self.model_improved(test_loss):
            # when did the model last improve?
            improvement = epoch - self.last_test_loss_improvement
            if improvement > self.early_stopping_epochs:
                return True
        return False


    def predict(self, X):
        X = X.reshape((1,X.shape[0]))
        feed_dict = {self.features_pl: X, self.keep_prob_pl: 1.0}
        softmax = tf.nn.softmax(self.model)
        activations = softmax.eval(session=self.sess, feed_dict=feed_dict)
        y = activations / sum(activations[0])

        # get predicted label
        label = np.argmax(y)
        return label, y[0][label]

    
    def calculate_manual_accuracy(self):
        num_prediction_tests = self.test.num_examples
        X, y = self.test.get_random_elements(-1)
        correct = 0
        for i in range(num_prediction_tests):
            y_hat, prob = self.predict(X[i])
            if y_hat == y[i]:
                correct += 1
                #print('[!] Y: {0} - Y predicted: {1} ({2:.2f})'.format(y[i], y_hat, prob))

        return float(correct) / float(num_prediction_tests)


    def print_step_summary_and_update_best_values(self, epoch, train_loss, train_precission, test_loss, test_precission, duration, train_f1, test_f1, colored=True):
        
        # print table header again after every 3000th step
        if epoch % 100 == 0 and epoch > 0:
            if self.calculate_f1_score:
                print('Epoch\tTrain Loss\tTrain accuracy\tTrain F1\t\tTest Loss\tTest accuracy\tTest F1\tDuration')   
            else:
                print('Epoch\tTrain Loss\tTrain accuracy\t\tTest Loss\tTest accuracy\tDuration')   
           
        tr_l_color = colored_shell_seq('WHITE')
        te_l_color = colored_shell_seq('WHITE')
        tr_p_color = colored_shell_seq('WHITE')
        te_p_color = colored_shell_seq('WHITE')
        tr_f_color = colored_shell_seq('WHITE')
        te_f_color = colored_shell_seq('WHITE')

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            tr_l_color = colored_shell_seq('GREEN')        

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.last_test_loss_improvement = epoch
            te_l_color = colored_shell_seq('GREEN')

        if train_precission > self.best_train_precission:
            self.best_train_precission = train_precission
            tr_p_color = colored_shell_seq('GREEN')

        if test_precission > self.best_test_precission:
            self.best_test_precission = test_precission
            te_p_color = colored_shell_seq('GREEN')

        if self.calculate_f1_score and train_f1 > self.best_train_f1:
            self.best_train_f1 = train_f1
            tr_f_color = colored_shell_seq('GREEN')

        if self.calculate_f1_score and test_f1 > self.best_test_f1:
            self.best_test_f1 = test_f1
            te_f_color = colored_shell_seq('GREEN')

        if self.calculate_f1_score:
            train_string = tr_l_color + '{0:.5f}\t\t'.format(train_loss) + tr_p_color + '{0:.2f}%\t\t'.format(train_precission*100) + tr_f_color + '{0:.3f}\t|\t'.format(train_f1)
            test_string = te_l_color + '{0:.5f}\t\t'.format(test_loss) + te_p_color + '{0:.2f}%\t\t'.format(test_precission*100) + te_f_color + '{0:.3f}\t|\t'.format(test_f1)
        else:
            train_string = tr_l_color + '{0:.5f}\t\t'.format(train_loss) + tr_p_color + '{0:.2f}%\t\t\t'.format(train_precission*100)
            test_string = te_l_color + '{0:.5f}\t\t'.format(test_loss) + te_p_color + '{0:.2f}%\t\t'.format(test_precission*100)

        print('{0}\t'.format(epoch) + train_string + test_string + colored_shell_seq('WHITE') + '{0:.3f}'.format(duration))
              


    def add_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)

    def save_model(self, checkpoint_file, step):
        if self.saver is None:
            raise Exception('Could not save model because saver is not initialized. Models can only be saved during training. Model dump: ' + str(self))
        save_path = self.saver.save(self.sess, checkpoint_file)       

        
    def load_model(self, file_name):
        if self.saver is None:
            raise Exception('Could not load model because saver is not initialized. Model dump: ' + str(self))
        self.saver.restore(self.sess, file_name)
        logger.info('Model was restored.')

def activation_summary(x):

      """Helper to create summaries for activations.

      Creates a summary that provides a histogram of activations.
      Creates a summary that measures the sparsity of activations.

      Args:
        x: Tensor
      Returns:
        nothing
      """

      tensor_name = x.op.name
      tf.summary.histogram(tensor_name + '/activations', x)
      tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def image_summary(x, tensor_name=None, max_images=3):
    if tensor_name is None:
        tensor_name = x.op.name
    tf.summary.image(tensor_name, x)


def put_kernels_on_grid (kernel, pad = 1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """
    # Source: https://gist.githubusercontent.com/kukuruza/03731dc494603ceab0c5/raw/8c94358889fc83612efe7fa35a1455797da6e1b3/gist_cifar10_train.py

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7










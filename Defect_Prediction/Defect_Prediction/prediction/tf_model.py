
from __future__ import division

import tensorflow as tf
import numpy as np
import math
import time
import os.path
import sys
import logging
import colorama

from misc import utils

logger = logging.getLogger('prediction')

tf.set_random_seed(42)

TF_LOG_CREATE_SUB_DIR = True
TF_LOG_DIR = 'C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/tf/'



def get_weights_variable(dim_x, dim_y, std_dev=1.0, name='weights'):
    weights = tf.Variable(
        tf.truncated_normal([dim_x, dim_y], stddev=std_dev / math.sqrt(float(dim_x))),
        name=name)
    return weights


def get_bias_variable(num_neurons, name='biases'):
    biases = tf.Variable(tf.zeros([num_neurons]), name=name)
    return biases


def get_input_placeholders(X_feature_vector_length, batch_size, num_classes):
    input_features_placeholders = tf.placeholder(tf.float32, shape=(None, X_feature_vector_length))
    targets_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return input_features_placeholders, targets_placeholder


def fill_feed_dict(data_set, features_placeholders, targets_placeholders, batch_size):
    features_feed, targets_feed = data_set.next_batch(batch_size)

    feed_dict = {
        features_placeholders: features_feed,
        targets_placeholders: targets_feed,
    }
    return feed_dict

class TensorFlowNet(object):
    """description of class"""

    def __init__(self,
                train_data_set, 
                test_data_set, 
                num_classes, 
                batch_size, 
                initial_learning_rate=0.1, 
                architecture_shape=(1024, 64), 
                log_dir=TF_LOG_DIR, 
                max_steps=20000,
                num_epochs_per_decay=350,
                learning_rate_decay_factor=0.1,
                model_name=str(int(time.time()))):
        self.sess = None
        self.saver = None
        self.train = train_data_set
        self.test = test_data_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.hidden_1_size = architecture_shape[0]
        self.hidden_2_size = architecture_shape[1]
        self.max_steps = max_steps
        

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

        self.model_name = model_name

        if TF_LOG_CREATE_SUB_DIR:
            self.log_dir = os.path.join(log_dir, self.model_name)
            utils.create_dir_if_necessary(self.log_dir)
        else:
            self.log_dir = log_dir 
        colorama.init()
        

    def inference(self, features_pl, hidden1_size, hidden2_size):
        """Build the defect prediction model.

        Args:
            features_pl: feature placeholder
            hidden1_size: number of neurons in the first hidden layer
            hidden2_size: number of neurons in the second hidden layer

        Returns:
            Logits.
        """

        # Layer 1:
        with tf.variable_scope('hidden1') as scope:
            weights = get_weights_variable(self.train.feature_shape[1], hidden1_size)            
            biases = get_bias_variable(hidden1_size)
            hidden1_tensor = tf.nn.relu(tf.matmul(features_pl, weights) + biases, name=scope.name) # TODO: change to relu_layer
            activation_summary(hidden1_tensor)

        # Layer 2:
        with tf.variable_scope('hidden2') as scope:
            weights = get_weights_variable(hidden1_size, hidden2_size)
            biases = get_bias_variable(hidden2_size)
            hidden2_tensor = tf.nn.relu(tf.matmul(hidden1_tensor, weights) + biases, name=scope.name)
            activation_summary(hidden2_tensor)

        # Output:
        with tf.variable_scope('softmax_linear') as scope:
            weights = get_weights_variable(hidden2_size, self.num_classes)
            biases = get_bias_variable(self.num_classes)
            logit_tensor = tf.add(tf.matmul(hidden2_tensor, weights), biases, name=scope.name)
            activation_summary(logit_tensor)
        return logit_tensor


    def loss(self, logit_tensor, targets_pl):
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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_tensor, targets, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        return cross_entropy_mean

    

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
        tf.summary.scalar('loss', loss_tensor)

        #tf.summary.scalar(loss_tensor.op.name, loss_tensor)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # apply the gradients to minimize loss.
        # for each minization increment global_step counter
        train_op = optimizer.minimize(loss_tensor, global_step=global_step)

        # add histogram for each trainable variable
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return train_op


    def do_eval(self, eval_correct_tensor, features_pl, targets_pl, data_set):

        # number of correct predictions
        true_count = 0
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in range(steps_per_epoch):
            feed_dict = fill_feed_dict(data_set, features_pl, targets_pl, self.batch_size)
            true_count += self.sess.run(eval_correct_tensor, feed_dict=feed_dict)
        precision = true_count / num_examples
        return num_examples, true_count, precision


    def evaluate(self, logit_tensor, targets_pl):
        correct = tf.nn.in_top_k(logit_tensor, targets_pl, 1)

        # get the number of correct entries
        correct_sum = tf.reduce_sum(tf.cast(correct, tf.int32))
        return correct_sum

    def run_training(self):
        logger.info('Building NN model. Attributes:')
        logger.debug('\tTrain Feature Shape: {0}'.format(self.train.feature_shape))
        logger.debug('\tTest Feature Shape: {0}'.format(self.test.feature_shape))
        logger.info('\tTrain Zero Error: {0}'.format(self.train.zero_error))
        logger.info('\tTest Zero Error: {0}'.format(self.test.zero_error))
        logger.debug('\tBatch Size: {0}'.format(self.batch_size))
        logger.debug('\tMax Steps: {0}'.format(self.max_steps))
        logger.debug('\tLayer 1 Size: {0}'.format(self.hidden_1_size))
        logger.debug('\tLayer 2 Size: {0}'.format(self.hidden_2_size))
        logger.debug('\tNum Classes: {0}'.format(self.num_classes))
        logger.debug('\tLearning rate: {0}'.format(self.initial_learning_rate))

        with tf.Graph().as_default():

            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Generate the input placeholders
            features_pl, targets_pl = get_input_placeholders(self.train.feature_shape[1], self.batch_size, self.num_classes)
            self.features_pl = features_pl

            # build the model
            logit_tensor = self.inference(features_pl, self.hidden_1_size, self.hidden_2_size)
            self.model = logit_tensor

            # add loss tensor to graph
            loss_tensor = self.loss(logit_tensor, targets_pl)

            # create gradient training op
            train_op = self.get_train_op(loss_tensor, global_step)

            # add evaluation step 
            eval_correct = self.evaluate(logit_tensor, targets_pl)

            # Build the summary Tensor based on the TF collection of Summaries.
            summary_tensor = tf.summary.merge_all()

            # add variables initializer
            init = tf.global_variables_initializer()

            # initialize model saver
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            self.sess = tf.Session()

            # initialize a SummaryWriter which writes a log file
            summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

            # initialize variables
            self.sess.run(init)

            logger.info('Neural Net is initialized and ready to train.')
            print('\n')
            logger.debug('Step (/100)\tLoss\tDuration') 
            
            print('Step (in hundreds)\tTrain Loss\tTrain precission\tTest Loss\tTest precission\tDuration')    
            print('=====================================================================================================')      

            # start training            
            for step in range(self.max_steps):
                start_time = time.time()

                # fill feed dict with batch
                train_feed_dict = fill_feed_dict(self.train, features_pl, targets_pl, self.batch_size)
                test_feed_dict = fill_feed_dict(self.test, features_pl, targets_pl, self.batch_size)

                # run the model
                # _: result of train_op (is None)
                # loss_value: result of loss operation (the actual loss)
                train_loss_value = -1
                test_loss_value = -1
                try:
                    _, train_loss_value = self.sess.run(
                        [train_op, loss_tensor],
                        feed_dict=train_feed_dict)
                    test_loss_value = self.sess.run([loss_tensor], feed_dict=test_feed_dict)[0]

                except:
                    logger.exception('Could not run train step {0}. Loss Value: {1}'.format(step, train_loss_value))

                assert not np.isnan(train_loss_value), 'Model diverged with loss = NaN'

                

                # Write summaries
                if step % 200 == 0:
                    duration = time.time() - start_time
                    
                    train_num_examples, train_true_count, train_precision = self.do_eval(eval_correct, features_pl, targets_pl, self.train)
                    test_num_examples, test_true_count, test_precision = self.do_eval(eval_correct, features_pl, targets_pl, self.test)

                    logger.debug('Train: Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(train_num_examples, train_true_count, train_precision))
                    logger.debug('Test: Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(test_num_examples, test_true_count, test_precision))
                    logger.debug('{0}\t\t{1:.4f}\t{2:.5f}'.format(step/100, train_loss_value, duration))                  

                    self.print_step_summary(step, train_loss_value, train_precision, test_loss_value, test_precision, duration, colored=True)
                    
                    tf.summary.scalar('Train_Loss', tf.Variable(train_loss_value, trainable=False, name='train_loss_value'))
                    tf.summary.scalar('Test_Loss', tf.Variable(test_loss_value, trainable=False, name='test_loss_value'))
                    summary_str = self.sess.run(summary_tensor, feed_dict=train_feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save checkpoint
                if (step + 1) % 1000 == 0 or (step + 1) == self.max_steps:
                    checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')                    
                    try:
                        self.save_model(checkpoint_file, step)
                    except:
                        logger.exception('Could not save model.')

                    

    def predict(self, X):
        X = X.reshape((1,X.shape[0]))
        feed_dict = {self.features_pl: X}
        softmax = tf.nn.softmax(self.model)
        activations = softmax.eval(session=self.sess, feed_dict=feed_dict)
        y = activations / sum(activations[0])

        # get predicted label
        label = np.argmax(y)
        return label, y[0][label]


    def print_step_summary(self, step, train_loss, train_precission, test_loss, test_precission, duration, colored=True):
        
        # print table header again after every 3000th step
        if step % 5000 == 0 and step > 0:
            print('\nStep (in hundreds)\tTrain Loss\tTrain precission\tTest Loss\tTest precission\tDuration')    
           
        tr_l_color = utils.colored_shell_seq('WHITE')
        te_l_color = utils.colored_shell_seq('WHITE')
        tr_p_color = utils.colored_shell_seq('WHITE')
        te_p_color = utils.colored_shell_seq('WHITE')

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            tr_l_color = utils.colored_shell_seq('GREEN')        

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            te_l_color = utils.colored_shell_seq('GREEN')

        if train_precission > self.best_train_precission:
            self.best_train_precission = train_precission
            tr_p_color = utils.colored_shell_seq('GREEN')

        if test_precission > self.best_test_precission:
            self.best_test_precission = test_precission
            te_p_color = utils.colored_shell_seq('GREEN')

        train_string = tr_l_color + '{0:.3f}\t\t'.format(train_loss) + tr_p_color + '{0:.1f}%\t\t\t'.format(train_precission*100)
        test_string = te_l_color + '{0:.3f}\t\t'.format(test_loss) + te_p_color + '{0:.1f}%\t\t'.format(test_precission*100)

        print('{0}\t\t\t'.format(step/100) + train_string + test_string + utils.colored_shell_seq('WHITE') + '{0:.3f}'.format(duration))
              


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
        save_path = self.saver.save(self.sess, checkpoint_file, global_step=step)       

        
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
      #tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
      #tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def add_loss_summaries(loss_tensor):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss_tensor])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [loss_tensor]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op





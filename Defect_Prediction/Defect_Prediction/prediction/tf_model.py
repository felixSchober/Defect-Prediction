
from __future__ import division

import tensorflow as tf
import numpy as np
import math
import time
import os.path
import sys
import logging


logger = logging.getLogger('prediction')

tf.set_random_seed(42)

def get_weights_variable(dim_x, dim_y, std_dev=1.0, name='weights'):
    weights = tf.Variable(
        tf.truncated_normal([dim_x, dim_y], stddev=std_dev / math.sqrt(float(dim_x))),
        name=name)
    return weights


def get_bias_variable(num_neurons, name='biases'):
    biases = tf.Variable(tf.zeros([num_neurons]), name=name)
    return biases


def get_input_placeholders(X_feature_vector_length, batch_size, num_classes):
    input_features_placeholders = tf.placeholder(tf.float32, shape=(batch_size, X_feature_vector_length))
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
                learning_rate, 
                architecture_shape=(1024, 32), 
                log_dir='C:/Users/felix/OneDrive/Studium/Studium/2. Semester/Seminar/Project/Training/', 
                max_steps=5000):
        self.sess = None
        self.saver = None
        self.train = train_data_set
        self.test = test_data_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.hidden_1_size = architecture_shape[0]
        self.hidden_2_size = architecture_shape[1]
        self.max_steps = max_steps
        self.log_dir = log_dir 
        

    def get_network_tensor(self, features_pl, hidden1_size, hidden2_size):

        # Layer 1:
        with tf.name_scope('hidden1'):
            weights = get_weights_variable(self.train.feature_shape[1], hidden1_size)
            biases = get_bias_variable(hidden1_size)
            hidden1_tensor = tf.nn.relu(tf.matmul(features_pl, weights) + biases) # TODO: change to relu_layer

        # Layer 2:
        with tf.name_scope('hidden2'):
            weights = get_weights_variable(hidden1_size, hidden2_size)
            biases = get_bias_variable(hidden2_size)
            hidden2_tensor = tf.nn.relu(tf.matmul(hidden1_tensor, weights) + biases)

        # Output:
        with tf.name_scope('softmax_linear'):
            weights = get_weights_variable(hidden2_size, self.num_classes)
            biases = get_bias_variable(self.num_classes)
            logit_tensor = tf.matmul(hidden2_tensor, weights) + biases
        return logit_tensor


    def get_loss_tensor(self, logit_tensor, targets_pl):
        targets = tf.to_int64(targets_pl)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_tensor, targets, name='xentropy')
        loss_tensor = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss_tensor


    def get_train_op(self, loss_tensor):
        tf.summary.scalar(loss_tensor.op.name, loss_tensor)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        global_step = tf.Variable(0, trainable=False, name='global_step')

        # apply the gradients to minimize loss.
        # for each minization increment global_step counter
        train_op = optimizer.minimize(loss_tensor, global_step=global_step)
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
        logger.info('Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(num_examples, true_count, precision))




    def evaluate(self, logit_tensor, targets_pl):
        correct = tf.nn.in_top_k(logit_tensor, targets_pl, 1)

        # get the number of correct entries
        correct_sum = tf.reduce_sum(tf.cast(correct, tf.int32))
        return correct_sum

    def run_training(self):
        logger.info('Building NN model. Attributes:')
        logger.debug('\tTrain Feature Shape: {0}'.format(self.train.feature_shape))
        logger.debug('\tTest Feature Shape: {0}'.format(self.test.feature_shape))
        logger.debug('\tBatch Size: {0}'.format(self.batch_size))
        logger.debug('\tMax Steps: {0}'.format(self.max_steps))
        logger.debug('\tLayer 1 Size: {0}'.format(self.hidden_1_size))
        logger.debug('\tLayer 2 Size: {0}'.format(self.hidden_2_size))
        logger.debug('\tNum Classes: {0}'.format(self.num_classes))
        logger.debug('\tLearning rate: {0}'.format(self.learning_rate))
        with tf.Graph().as_default():
            # Generate the input placeholders
            features_pl, targets_pl = get_input_placeholders(self.train.feature_shape[1], self.batch_size, self.num_classes)

            # build the model
            logit_tensor = self.get_network_tensor(features_pl, self.hidden_1_size, self.hidden_2_size)

            # add loss tensor to graph
            loss_tensor = self.get_loss_tensor(logit_tensor, targets_pl)

            # create gradient training op
            train_op = self.get_train_op(loss_tensor)

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
            logger.info('Step\tLoss\tDuration')            

            # start training
            for step in range(self.max_steps):
                start_time = time.time()

                # fill feed dict with batch
                feed_dict = fill_feed_dict(self.train, features_pl, targets_pl, self.batch_size)

                # run the model
                # _: result of train_op (is None)
                # loss_value: result of loss operation (the actual loss)
                loss_value = -1
                try:
                    _, loss_value = self.sess.run(
                        [train_op, loss_tensor],
                        feed_dict=feed_dict)
                except:
                    logger.exception('Could not run train step {0}. Loss Value: {1}'.format(step, loss_value))

                duration = time.time() - start_time

                # Write summaries
                if step % 100 == 0:
                    logger.info('{0}\t\t{1:.4f}\t{2:.2f}'.format(step, loss_value, duration))

                    summary_str = self.sess.run(summary_tensor, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save checkpoint
                if (step + 1) % 1000 == 0 or (step + 1) == self.max_steps:
                    checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')
                    
                    try:
                        self.save_model(checkpoint_file, step)
                    except:
                        logger.exception('Could not save model.')

                    logger.info('Eval')    
                    logger.info('Training Data Eval:')
                    self.do_eval(eval_correct, features_pl, targets_pl, self.train)
                    logger.info('Test Data Eval:')
                    self.do_eval(eval_correct, features_pl, targets_pl, self.test)

                    





    def save_model(self, checkpoint_file, step):
        if self.saver is None:
            raise Exception('Could not save model because saver is not initialized. Models can only be saved during training. Model dump: ' + str(self))
        save_path = self.saver.save(self.sess, checkpoint_file, global_step=step)       

        
    def load_model(self, file_name):
        if self.saver is None:
            raise Exception('Could not load model because saver is not initialized. Model dump: ' + str(self))
        self.saver.restore(self.sess, file_name)
        logger('Model was restored.')






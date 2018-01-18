import tensorflow as tf
import numpy as np
import os
from data_config import cifar10
from data_config.cifar10 import img_size, num_channels, num_classes
import random
class AlexNet_Model(object):
    def __init__(self,sess,conf):
        self.sess = sess
        self.conf = conf
        self.self_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        self.configure_network()

    def weight_var(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_var(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="VALID")

    def conv2d_padding(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="VALID")


    def self_params(self):
        self.input_shape = self.conf.input_shape
        self.conv1_shape = self.conf.conv1_shape
        self.conv2_shape = self.conf.conv2_shape
        self.conv3_shape = self.conf.conv3_shape
        self.conv4_shape = self.conf.conv4_shape
        self.fc_lay1_shape = self.conf.fc_lay1_shape
        self.fc_lay2_shape = self.conf.fc_lay2_shape
        self.output_shape = self.conf.output_shape

    def configure_network(self):
        self.build_network()
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.training_operation = self.optimizer.minimize(self.cross_entropy, name = 'train_op')
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        with tf.variable_scope('logging'):
            tf.summary.scalar('current_cost', self.cross_entropy)
            self.summary = tf.summary.merge_all()
        self.training_writer = tf.summary.FileWriter("../.logs/training", self.sess.graph)
        self.testing_writer = tf.summary.FileWriter("../.logs/testing", self.sess.graph)

    def build_network(self):
        #with tf.variable_scope('input'):
        self.x_images = tf.placeholder(tf.float32,shape=self.input_shape,name = 'inputs')
        self.x_image = tf.reshape(self.x_images,[-1,32,32,3])
        #with tf.variable_scope('conv1'):
        self.W_conv1 = self.weight_var(self.conv1_shape)

        self.b_conv1 = self.bias_var([self.conv1_shape[3]])
        self.y_conv1 = tf.nn.relu(self.conv2d(self.x_image,self.W_conv1) + self.b_conv1)
        self.y_pool1 = self.max_pool_2x2(self.y_conv1)

        #with tf.variable_scope('conv2'):
        self.W_conv2 = self.weight_var(self.conv2_shape)
        self.b_conv2 = self.bias_var([self.conv2_shape[3]])
        self.y_conv2 = tf.nn.relu(self.conv2d(self.y_pool1, self.W_conv2) + self.b_conv2)
        self.y_pool2 = self.max_pool_2x2(self.y_conv2)

        #with tf.variable_scope('conv3'):
        self.W_conv3 = self.weight_var(self.conv3_shape)
        self.b_conv3 = self.bias_var([self.conv3_shape[3]])
        self.y_conv3 = tf.nn.relu(self.conv2d_padding(self.y_pool2, self.W_conv3) + self.b_conv3)

        # with tf.variable_scope('conv4'):
        self.W_conv4 = self.weight_var(self.conv4_shape)
        self.b_conv4 = self.bias_var([self.conv4_shape[3]])
        self.y_conv4 = tf.nn.relu(self.conv2d_padding(self.y_conv3, self.W_conv4) + self.b_conv4)
        self.y_pool4 = self.max_pool_2x2(self.y_conv4)

        #with tf.variable_scope('fc1'):
        self.W_fc1 = self.weight_var(self.fc_lay1_shape)
        self.b_fc1 = self.bias_var([self.fc_lay1_shape[1]])

        self.flat_pool4 = tf.reshape(self.y_pool4,[-1,self.fc_lay1_shape[0]])
        self.y_fc1 = tf.nn.relu(tf.matmul(self.flat_pool4,self.W_fc1) + self.b_fc1)

        #with tf.variable_scope('fc2'):
        self.W_fc2 = self.weight_var(self.fc_lay2_shape)
        self.b_fc2 = self.bias_var([self.fc_lay2_shape[1]])
        self.y_fc2 = tf.nn.relu(tf.matmul(self.y_fc1,self.W_fc2) + self.b_fc2)

        self.dropout_prob = tf.placeholder(tf.float32)
        self.y_fc2_dropout = tf.nn.dropout(self.y_fc1,self.dropout_prob)

        #with tf.variable_scope('output'):
        self.W_out = self.weight_var(self.output_shape)
        self.b_out = self.bias_var([self.output_shape[1]])
        self.y_final = tf.matmul(self.y_fc2_dropout,self.W_out) + self.b_out
        self.calc_loss()

    def calc_loss(self):
        with tf.variable_scope('cost'):
            self.Y = tf.placeholder(tf.float32, [self.output_shape[1]], name="Y")
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.y_final))

    def save_summary(self,summary, step):
        print('---->summarizing', step)
        self.training_writer.add_summary(summary, step)

    def save(self,step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def train(self):
        images_train, cls_train, labels_train = cifar10.load_training_data()
        images_test, cls_test, labels_test = cifar10.load_test_data()

        print("Size of:")
        print("- Training-set:\t\t{}".format(len(images_train)))
        print("- Test-set:\t\t{}".format(len(images_test)))
        for epoch_num in range(self.conf.max_step):
            for _ in range(50000):
                y_true = np.zeros([10])
                y_true[cls_train[_]] = 1.0
                feed_dict = {self.x_images: images_train[_],self.Y:y_true, self.dropout_prob:0.5}
                self.sess.run(self.training_operation,feed_dict=feed_dict)
                if _ %10000 == 0:
                    training_error= self.sess.run(self.cross_entropy, feed_dict=feed_dict)
                    y_test = np.zeros([10])
                    y_test[cls_test[random.randint(0, 9999)]] = 1.0
                    testing_feedict = {self.x_images: images_test[random.randint(0, 9999)], self.Y: y_test,
                                       self.dropout_prob: 1.0}
                    testing_error = self.sess.run(self.cross_entropy, feed_dict=testing_feedict)
                    print("Training Error: {} for Epoch {}".format(training_error, epoch_num))
                    print("Testing Error: {} for Epoch {}".format(testing_error, epoch_num))


            if epoch_num % 100 == 0:
                training_error, training_summary = self.sess.run([self.cross_entropy, self.summary],feed_dict=feed_dict)
                y_test = np.zeros([10])
                y_test[cls_test[random.randint(0,9999)]] = 1.0
                testing_feedict = {self.x_images: images_test[random.randint(0, 9999)],self.Y: y_test, self.dropout_prob:1.0}
                testing_error, testing_summary = self.sess.run([self.cross_entropy, self.summary],feed_dict=testing_feedict)
                print("Training Error: {} for Epoch {}".format(training_error,epoch_num))
                print("Testing Error: {} for Epoch {}".format(testing_error,epoch_num))
                self.save_summary(training_summary,epoch_num )
                self.save_summary(testing_summary,epoch_num)
                self.save(epoch_num)























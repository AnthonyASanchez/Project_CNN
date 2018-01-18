from data_config import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from data_config.cifar10 import img_size, num_channels, num_classes
from AlexNet_Model import AlexNet_Model
import os
import time
import argparse

def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 100, '# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 0, '# of interval to save a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
    # data
    flags.DEFINE_string('data_dir', '/tempspace/wzhang/bigneuron/PixelDCN-feature-3d/dataset/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'train_data/train.txt', 'Training data')
    flags.DEFINE_string('valid_data', 'train_data/val.txt', 'Validation data')

    flags.DEFINE_float('input_shape',[ 32, 32, 3],'input shape')
    flags.DEFINE_float('conv1_shape', [4, 4, 3, 22], 'conv1 shape')
    flags.DEFINE_float('conv2_shape', [2, 2, 22, 52], 'conv2 shape')
    flags.DEFINE_float('conv3_shape', [2, 2, 52, 75], 'conv3 shape')
    flags.DEFINE_float('conv4_shape', [2, 2, 75, 70], 'conv4 shape')
    flags.DEFINE_float('fc_lay1_shape', [630,256], 'FC Lay1 shape')
    flags.DEFINE_float('fc_lay2_shape', [256,256], 'FC Lay2 shape')
    flags.DEFINE_float('output_shape',[256,10],'output shape')
    # Debug
    flags.DEFINE_string('logdir', '../logdir', 'Log dir')
    flags.DEFINE_string('modeldir', '../models', 'Model dir')
    flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step',0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')

    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    else:
        model = AlexNet_Model(tf.Session(), configure())
        getattr(model, args.action)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.app.run()

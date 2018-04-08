import tensorflow as tf
import numpy as np

import os
import shutil

def add_neuron(inputs, split_dim, activation_function = None):
    # add one more layer and return the output of this layer
    """

    :param inputs:
    :param split_dim:
    :param activation_function:
    :return:
    """

    scope_name = "group" + str(split_dim[0]);
    with tf.name_scope(scope_name):
        x = tf.slice(inputs, [0, split_dim[0]], [tf.shape(inputs)[0], split_dim[1] - split_dim[0]])
        W = tf.Variable(tf.random_normal([split_dim[1] - split_dim[0], 1]))
        b = tf.Variable(tf.zeros([1]) + 0.1)
        Z = tf.matmul(x, W) + b
    # if activation_function is None:
    #     outputs = Z
    # else:
    #     outputs = activation_function(Z)
    return Z, activation_function(Z), W, b

def normalization(data):
    # add one more layer and return the output of this layer
    """

    :param data: a training data

    :return:
    """

    for i in range(len(data[0])):
        max = np.max(data[:, i])
        min = np.min(data[:, i])
        for j in range(len(data)):
            if max == min:
                data[j][i] = 0
            else:
                data[j][i] = (max - data[j][i]) / (max - min)
    return data

def remove(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

def save(sess, saver, checkpoint_dir, model_file):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(
        sess,
        os.path.join(
            checkpoint_dir,
            model_file))

def load(sess, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(
            sess, os.path.join(
                checkpoint_dir, ckpt_name))
        return sess
    else:
        return sess

def printAccuracy(prediction_value, x_data, y_data):
    print("data size: ", len(prediction_value))
    right_1 = 0
    right_0 = 0
    wrong_1 = 0
    wrong_0 = 0
    for i in range(len(prediction_value)):
        if prediction_value[i][0] >= 0.5:
            if y_data[i][0] == 1:
                right_1 += 1
            else:
                wrong_0 += 1
        else:
            if y_data[i][0] == 0:
                right_0 += 1
            else:
                wrong_1 += 1
    print(right_0, wrong_0, right_1, wrong_1)
    print("accuracy for 0: ", right_0/(right_0+wrong_0))
    print("accuracy for 1: ", right_1/(right_1+wrong_1))
    print("total accuracy: ", (right_1+right_0)/(right_1+wrong_1+right_0+wrong_0))

def printWeight(split_dims, sess, ws, bs, pd_Weights, pd_biases):
    for i in range(len(split_dims)):
        print("neuron1_" + str(i) + "_w:", sess.run(ws[i]))
        print("neuron1_" + str(i) + "_b:", sess.run(bs[i]))
    print("neuron2_0_w:", sess.run(pd_Weights))
    print("neuron2_0_b:", sess.run(pd_biases))

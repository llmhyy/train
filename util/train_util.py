import tensorflow as tf
import numpy as np

import os
import shutil

def train(tensors, x_data, y_data, cost_threshold,
          checkpoint_filename, model_filename, split_dims):

    training_tensor = tensors.training_tensor
    cost_tensor = tensors.cost_tensor
    prob_tensor = tensors.prob_tensor
    xs = tensors.xs
    ys = tensors.ys
    ws1 = tensors.ws1
    bs1 = tensors.bs1
    ws2 = tensors.ws2
    bs2 = tensors.bs2

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        training_accuracy = 10
        while training_accuracy > cost_threshold:
            sess.run(training_tensor, feed_dict={xs: x_data, ys: y_data})
            training_accuracy = sess.run(cost_tensor, feed_dict={xs: x_data, ys: y_data})
            print("training accuracy %f" % (training_accuracy))
        # for i in range(iteration_time):
        #     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        #     training_accuracy = sess.run(cost, feed_dict={xs: x_data, ys: y_data})
        #     print("step %d, training accuracy %f" % (i, training_accuracy))

        checkpoint_dir = checkpoint_filename
        remove(checkpoint_dir)

        t_vars = tf.trainable_variables()
        saver = tf.train.Saver(t_vars)
        save(sess, saver, checkpoint_dir, model_filename)
        tf.summary.FileWriter(checkpoint_filename+"/graph", sess.graph)

        prediction_value = sess.run(prob_tensor, feed_dict={xs: x_data})
        # printWeight(split_dims, sess, ws1, bs1, ws2, bs2)

        return prediction_value

def buildNetwork(split_dims, xs, ys, learning_rate):
    normalized_xs = tf.nn.batch_normalization(xs, 0, 1, 0, 1, 0.001, name="Norm_Input")
    ws1 = []
    bs1 = []
    for i in range(len(split_dims)):
        if i == 0:
            logit, hidden_layer1, w, b = add_neuron(normalized_xs, split_dims[i], activation_function = tf.nn.relu)
        else:
            templogit, tempprob, w, b = add_neuron(normalized_xs, split_dims[i], activation_function = tf.nn.relu)
            hidden_layer1 = tf.concat([hidden_layer1, tempprob], 1)
        ws1.append(w)
        bs1.append(b)

    with tf.name_scope("2nd"):
        ws2 = tf.Variable(tf.random_normal([len(split_dims), 1]), name="W2")
        bs2 = tf.add(tf.Variable(tf.zeros([1])), 0.1, name="b2")
        pd_Z = tf.add(tf.matmul(hidden_layer1, ws2), bs2, name="Z2")
        prob_tensor = tf.nn.sigmoid(pd_Z, name="Output")

    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pd_Z, labels=ys))
    # train_step = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss, var_list=t_vars)
    cost_tensor = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pd_Z, labels=ys))
    t_vars = tf.trainable_variables()
    training_tensor = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_tensor, var_list=t_vars)

    return training_tensor, prob_tensor, cost_tensor, ws1, bs1, ws2, bs2


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

def printAccuracy(prediction_value, x_data, y_data, debug_info):
    print("data size: ", len(prediction_value))
    right_1 = []
    right_0 = []
    wrong_1 = []
    wrong_0 = []

    for i in range(len(prediction_value)):
        if prediction_value[i][0] >= 0.5:
            if y_data[i][0] == 1:
                right_1.append(debug_info[i])
            else:
                wrong_0.append(debug_info[i])
        else:
            if y_data[i][0] == 0:
                right_0.append(debug_info[i])
            else:
                wrong_1.append(debug_info[i])

    right_0_len = len(right_0)
    wrong_0_len = len(wrong_0)
    right_1_len = len(right_1)
    wrong_1_len = len(wrong_1)
    print(right_0_len, wrong_0_len, right_1_len, wrong_1_len)
    print("accuracy for 0: ", right_0_len/(right_0_len+wrong_0_len))
    print("accuracy for 1: ", right_1_len/(right_1_len+wrong_1_len))
    print("total accuracy: ", (right_1_len+right_0_len)/(right_1_len+wrong_1_len+right_0_len+wrong_0_len))

    print("wrong_1:", wrong_1)
    print("wrong_0:", wrong_0)

def printWeight(split_dims, sess, ws, bs, pd_Weights, pd_biases):
    for i in range(len(split_dims)):
        print("neuron1_" + str(i) + "_w:", sess.run(ws[i]))
        print("neuron1_" + str(i) + "_b:", sess.run(bs[i]))
    print("neuron2_0_w:", sess.run(pd_Weights))
    print("neuron2_0_b:", sess.run(pd_biases))

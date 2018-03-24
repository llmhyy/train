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
    slice = tf.slice(inputs, [0, split_dim[0]], [tf.shape(inputs)[0], split_dim[1] - split_dim[0]])
    x = tf.cast(slice, tf.float64)

    W = tf.Variable(tf.random_normal([split_dim[1] - split_dim[0], 1]))
    Weights = tf.cast(W ,tf.float64)
    # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    b = tf.Variable(tf.zeros([1]) + 0.1)
    biases = tf.cast(b,tf.float64)

    Z = tf.matmul(x, Weights) + biases
    # if activation_function is None:
    #     outputs = Z
    # else:
    #     outputs = activation_function(Z)
    return Z, activation_function(Z)

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

def save(sess, saver, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(
        sess,
        os.path.join(
            checkpoint_dir,
            'NN.model'))

# the 2nd dimension is probably [1, 3]
split_dims = [[0,3], [3,5], [5,6], [6,109], [109,212], [212,315], [315,418], [418,521], [521,624]]

data = np.loadtxt("./data/train.csv", delimiter=',')
x_data = data[:,1:]
x_data = normalization(x_data)

y_data = data[:,0:1]

#define placeholder for inputs
xs = tf.placeholder(tf.float64, [None, len(x_data[0])])
ys = tf.placeholder(tf.float64, [None, 1])

# train
dprob = 0.0
learning_rate = 0.01
beta1 = 0.5

#build neural network
for i in range(len(split_dims)):
    if i == 0:
        logit, hidden_layer1 = add_neuron(xs, split_dims[i], activation_function = tf.nn.relu)
    else:
        templogit, tempprob = add_neuron(xs, split_dims[i], activation_function = tf.nn.relu)
        hidden_layer1 = tf.concat([hidden_layer1, tempprob], 1)

pd_Weights = tf.cast(tf.Variable(tf.random_normal([len(split_dims), 1])), tf.float64)
pd_biases = tf.cast(tf.Variable(tf.zeros([1]) + 0.1), tf.float64)
pd_Z = tf.matmul(hidden_layer1, pd_Weights) + pd_biases
pd_prob = tf.nn.sigmoid(pd_Z)
t_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pd_Z, labels=ys))
train_step = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss, var_list=t_vars)

#initialization
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

checkpoint_dir = "checkpoint"
remove(checkpoint_dir)
saver = tf.train.Saver(t_vars)
save(sess, saver, checkpoint_dir)

prediction_value = sess.run(pd_prob, feed_dict={xs: x_data})
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

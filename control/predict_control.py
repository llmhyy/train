import tensorflow as tf
import numpy as np
import train_util

def add_neuron(inputs, split_dim, activation_function = None):
    x = tf.cast(tf.slice(inputs, [0, split_dim[0]], [tf.shape(inputs)[0], split_dim[1] - split_dim[0]]),tf.float64)
    Weights = tf.cast(tf.Variable(tf.random_normal([split_dim[1] - split_dim[0], 1])), tf.float64)
    biases = tf.cast(tf.Variable(tf.zeros([1]) + 0.1), tf.float64)
    Wx_plus_b = tf.matmul(x, Weights) + biases
    return Wx_plus_b, activation_function(Wx_plus_b)

def normalization(data):
    for i in range(len(data[0])):
        max = np.max(data[:, i])
        min = np.min(data[:, i])
        for j in range(len(data)):
            if max == min:
                data[j][i] = 0
            else:
                data[j][i] = (max - data[j][i]) / (max - min)
    return data

split_dims = [[0,1], [0,3], [3,5], [5,6], [6,109], [109,212], [212,315], [315,418], [418,521], [521,624]]

data = np.loadtxt("./control_data/control_test.csv",delimiter=',')
x_data = data[:,1:]
x_data = normalization(x_data)

y_data = data[:,0:1]

# define placeholder for inputs
xs = tf.placeholder(tf.float64, [None, len(x_data[0])])
ys = tf.placeholder(tf.float64, [None, 1])

# train
dprob = 0.5
learning_rate = 0.001
beta1 = 0.5

#build neural network
for i in range(len(split_dims)):
    if i == 0:
        logit, hidden_layer1 = add_neuron(xs, split_dims[i], activation_function = tf.nn.relu)
    else:
        templogit, tempprob = add_neuron(xs, split_dims[i], activation_function=tf.nn.relu)
        hidden_layer1 = tf.concat([hidden_layer1, tempprob], 1)

pd_Weights = tf.cast(tf.Variable(tf.random_normal([len(split_dims), 1])), tf.float64)
pd_biases = tf.cast(tf.Variable(tf.zeros([1]) + 0.1), tf.float64)
pd_logit = tf.matmul(hidden_layer1, pd_Weights) + pd_biases
pd_prob = tf.nn.sigmoid(pd_logit)
t_vars = tf.trainable_variables()

#initialization
init = tf.global_variables_initializer()
sess = tf.Session()

checkpoint_dir = "checkpoint"
saver = tf.train.Saver(t_vars)
sess = train_util.load(sess, saver, checkpoint_dir)

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

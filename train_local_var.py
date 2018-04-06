import tensorflow as tf
import numpy as np

import os
import shutil
import train_util

# the 2nd dimension is probably [1, 3]
split_dims = [[0,1], [1,3], [3,5], [6,7], [7,110], [110,213], [213,316], [316,419], [419,522], [522,625]]

data = np.loadtxt("./data_data/local_var.csv", delimiter=',')
positive_data = []
negative_data = []
for i in range(len(data)):
    d = data[i]
    if d[0]==1 :
        positive_data.append(d)
    else:
        negative_data.append(d)

enhanced_data = []
loop = int(len(negative_data)/len(positive_data)) - 1
for i in range(loop):
    for j in range(len(positive_data)):
        enhanced_data.append(positive_data[j])

array = np.asarray(enhanced_data)
data = np.append(data, array, axis=0)



x_data = data[:,1:]
x_data = train_util.normalization(x_data)

y_data = data[:,0:1]

#define placeholder for inputs
xs = tf.placeholder(tf.float64, [None, len(x_data[0])])
ys = tf.placeholder(tf.float64, [None, 1])

# train
dprob = 0.0
learning_rate = 0.01
beta1 = 0.5

#build neural network
ws = []
bs = []
for i in range(len(split_dims)):
    if i == 0:
        logit, hidden_layer1, w, b = train_util.add_neuron(xs, split_dims[i], activation_function = tf.nn.relu)
        ws.append(w)
        bs.append(b)
    else:
        templogit, tempprob, w, b = train_util.add_neuron(xs, split_dims[i], activation_function = tf.nn.relu)
        hidden_layer1 = tf.concat([hidden_layer1, tempprob], 1)
        ws.append(w)
        bs.append(b)

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
train_util.remove(checkpoint_dir)
saver = tf.train.Saver(t_vars)
train_util.save(sess, saver, checkpoint_dir)

prediction_value = sess.run(pd_prob, feed_dict={xs: x_data})
train_util.printAccuracy(prediction_value, y_data)
# train.printWeight(split_dims, sess, ws, bs, pd_Weights, pd_biases)

import tensorflow as tf
import numpy as np
import train_util
import predict

# split_dims = [[0,3], [3,5], [5,6], [6,109], [109,212], [212,315], [315,418], [418,521], [521,624]]
split_dims = [[0,3],[3,5],[5,6]]

data = np.loadtxt("./control_data/control.csv", delimiter=',')

x_data = data[:,1:]
# x_data = train_util.normalization(x_data)

y_data = data[:,0:1]

#define placeholder for inputs
xs = tf.placeholder(tf.float32, [None, len(x_data[0])], name="Input")
normalized_xs = tf.nn.batch_normalization(xs, 0, 1, 0, 1, 0.001, name="Norm_Input")

ys = tf.placeholder(tf.float32, [None, 1], name="Label")

# train
dprob = 0.0
learning_rate = 0.05
iteration_time = 2000
beta1 = 0.5

#build neural network
ws = []
bs = []
for i in range(len(split_dims)):
    if i == 0:
        logit, hidden_layer1, w, b = train_util.add_neuron(normalized_xs, split_dims[i], activation_function = tf.nn.relu)
    else:
        templogit, tempprob, w, b = train_util.add_neuron(normalized_xs, split_dims[i], activation_function = tf.nn.relu)
        hidden_layer1 = tf.concat([hidden_layer1, tempprob], 1)
    ws.append(w)
    bs.append(b)

with tf.name_scope("2nd"):
    pd_Weights = tf.Variable(tf.random_normal([len(split_dims), 1]), name="W2")
    pd_biases = tf.add(tf.Variable(tf.zeros([1])), 0.1, name="b2")
    pd_Z = tf.add(tf.matmul(hidden_layer1, pd_Weights), pd_biases, name="Z2")
    pd_prob = tf.nn.sigmoid(pd_Z, name="Output")
t_vars = tf.trainable_variables()


# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pd_Z, labels=ys))
# train_step = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss, var_list=t_vars)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pd_Z, labels=ys))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=t_vars)

#initialization
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(iteration_time):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        training_accuracy = sess.run(cost, feed_dict={xs: x_data, ys: y_data})
        print("step %d, training accuracy %f" % (i, training_accuracy))

    checkpoint_dir = "checkpoint"
    train_util.remove(checkpoint_dir)
    saver = tf.train.Saver(t_vars)
    train_util.save(sess, saver, checkpoint_dir, "train_control")
    tf.summary.FileWriter("checkpoint/graph", sess.graph)

    prediction_value = sess.run(pd_prob, feed_dict={xs: x_data})
    # train.printWeight(split_dims, sess, ws, bs, pd_Weights, pd_biases)

print("training accuracy")
train_util.printAccuracy(prediction_value, y_data)

print("testing accuracy")
predict.testModel('checkpoint/train_control.meta',
                  'checkpoint',
                  'control_data/control_test.csv')

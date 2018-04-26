import tensorflow as tf
import numpy as np
from util import train_util


def testModel(model_file, model_dir, test_file, usecol):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    debug_info = np.loadtxt(test_file, delimiter=',', comments='@', usecols=[0,1,2], dtype='str')
    x_data = np.loadtxt(test_file, delimiter=',', comments='@', usecols=usecol)
    y_data = np.loadtxt(test_file, delimiter=',', comments='@', usecols=[3], ndmin=2)

    graph = tf.get_default_graph()
    pb_prob = graph.get_tensor_by_name("2nd/Output:0")
    xs = graph.get_tensor_by_name("Input:0")
    prediction_value = sess.run(pb_prob, feed_dict={xs: x_data})
    train_util.printAccuracy(prediction_value, x_data, y_data, debug_info)

def retrieveNNModel(model_file, model_dir):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    graph = tf.get_default_graph()

    return sess, graph

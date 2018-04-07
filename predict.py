import tensorflow as tf
import numpy as np
import train_util
import os.path

def testModel(model_file, model_dir, test_file):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    data = np.loadtxt(test_file, delimiter=',')
    x_data = data[:,1:]
    x_data = train_util.normalization(x_data)
    y_data = data[:,0:1]

    graph = tf.get_default_graph()
    pb_prob = graph.get_tensor_by_name("2nd/Output:0")
    xs = graph.get_tensor_by_name("Input:0")
    prediction_value = sess.run(pb_prob, feed_dict={xs: x_data})
    train_util.printAccuracy(prediction_value, y_data)

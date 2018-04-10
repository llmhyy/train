from sys import stdin
from sys import stdout

import numpy as np
import predict as pre

sess, graph = pre.retrieveNNModel('control/checkpoint/train_control.meta',
                  'control/checkpoint',
                  'control/control_data/control_test.csv')
pb_prob = graph.get_tensor_by_name("2nd/Output:0")
xs = graph.get_tensor_by_name("Input:0")

try:
    while (1):
       data = stdin.readline()

       x_data = np.fromstring(data, dtype=int, sep=',')
       slice_xs = x_data[4:]
       slice_xs = np.reshape(slice_xs, (1, len(slice_xs)))
       prediction_value = sess.run(pb_prob, feed_dict={xs:slice_xs})

       print ("@@PythonStart@@")
       print ("data=", prediction_value)
       print ("@@PythonEnd@@")
       stdout.flush()
    print ("finished!")
finally:
    sess.close()


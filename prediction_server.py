from sys import stdin
from sys import stdout

import numpy as np
from util import predict as pre

# print("start")

type = stdin.readline()
checkpoint_filename = ""
model_filename = ""
if "control" in type:
    checkpoint_filename = "control/checkpoint_control"
    model_filename = "control"
elif "field" in type:
    checkpoint_filename = "data/checkpoint_field"
    model_filename = "field"
else:
    checkpoint_filename = "data/checkpoint_local_var"
    model_filename = "local_var"

sess, graph = pre.retrieveNNModel(checkpoint_filename+'/' + model_filename + '.meta',
                  checkpoint_filename)
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


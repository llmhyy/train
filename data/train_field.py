import tensorflow as tf
import numpy as np
from util import predict, train_util, tensors

checkpoint_filename = "checkpoint_field"
model_filename = "field"
train_file = "./data_data/field.csv"
test_file = "./data_data/field_test.csv"

split_dims = [[0,1], [1,5], [5,9], [9,10], [10,113], [113,216], [216,319], [319,422], [422,525], [525,628]]
# split_dims = [[0,1], [1,4], [4,7], [7,8]]

usecol=[]
for i in range(4, 632):
    usecol.append(i)

x_data = np.loadtxt(train_file, delimiter=',', usecols=usecol)
y_data = np.loadtxt(train_file, delimiter=',', usecols=[3], ndmin=2)

positive_data = []
negative_data = []
for i in range(len(x_data)):
    d = x_data[i]
    if y_data[i][0]==1 :
        positive_data.append(d)
    else:
        negative_data.append(d)

enhanced_data = []
loop = int(len(negative_data)/len(positive_data)) - 1
for i in range((int)(loop)):
    for j in range(len(positive_data)):
        enhanced_data.append(positive_data[j])

x_array = np.asarray(enhanced_data)
x_data = np.append(x_data, x_array, axis=0)

y_array = np.ones((len(x_array), 1), dtype=float)
y_data = np.append(y_data, y_array, axis=0)

randomize = np.arange(len(y_data))
np.random.shuffle(randomize)
x_data = x_data[randomize]
y_data = y_data[randomize]

# x_data, y_data = shuffle(x_data, y_data, random_state=0)

#define placeholder for inputs
xs = tf.placeholder(tf.float32, [None, len(x_data[0])], name="Input")
ys = tf.placeholder(tf.float32, [None, 1], name="Label")

# train
dprob = 0.0
learning_rate = 0.05
iteration_time = 2000
beta1 = 0.5
random_seed = 0;
cost_threshold = 0.5

# random.seed(random_seed)
# np.random.seed(random_seed)
# tf.set_random_seed(random_seed)

training_tensor, prob_tensor, cost_tensor, ws1, bs1, ws2, bs2 = \
    train_util.buildNetwork(split_dims, xs, ys, learning_rate)

# graph = tf.get_default_graph()
# list_of_tuples = [op.values() for op in graph.get_operations()]
# print(list_of_tuples)

#initialization
# init = tf.initialize_all_variables()

tensors = tensors.Tensors(training_tensor, cost_tensor, prob_tensor, xs, ys, ws1, bs1, ws2, bs2)

prediction_value = train_util.train(tensors, x_data, y_data, cost_threshold,
                                    checkpoint_filename, model_filename, split_dims)

print("training accuracy")
train_util.printAccuracy(prediction_value, x_data, y_data)

print("testing accuracy")
predict.testModel(checkpoint_filename + '/' + model_filename + '.meta',
                  checkpoint_filename,
                  test_file, usecol)

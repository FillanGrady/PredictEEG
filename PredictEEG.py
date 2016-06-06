import LoadEEG
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

"""
Neural Network model
Input -> Convolution Layer -> Max Pool -> Convolution Layer -> Max Pool -> LTSM Layer -> Connected Layer -> Output
      256                256*6       128*6                128*6     64*6->432         60                 30
"""


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

if __name__ == '__main__':
    ne, ns = LoadEEG.setup(1)
    learning_rate = 0.001
    training_iters = 10
    display_step = 1

    n_samples_per_second = 256
    n_lstm_input = 12
    n_lstm_output = 384
    n_output = 31
    input_depth = 23

    x = tf.placeholder(tf.float32, [None, input_depth])
    y = tf.placeholder(tf.float32, [None, n_output])

    depths = {'conv1': 12, 'conv2': 12}
    weights = {'conv1': weight_variable([10, 1, input_depth, depths['conv1']]),
               'conv2': weight_variable([10, 1, depths['conv2'], depths['conv1']]),
               'full': weight_variable([n_lstm_output, n_output])}
    biases = {'conv1': bias_variable([depths['conv1']]),
              'conv2': bias_variable([depths['conv2']]),
              'full': bias_variable([n_output])}
    input = tf.reshape(x, [1, -1, 1, input_depth])
    h_conv1 = tf.nn.relu(conv1d(input, weights['conv1']) + biases['conv1'])
    h_pool1 = max_pool_2(h_conv1)

    h_conv2 = tf.nn.relu(conv1d(h_pool1, weights['conv2']) + biases['conv2'])
    h_pool2 = max_pool_2(h_conv2)

    flat = tf.reshape(h_pool2, [-1, n_lstm_input])
    s = tf.split(0, 64, flat)

    lstm = rnn_cell.BasicLSTMCell(num_units=n_lstm_output, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm, s, dtype=tf.float32)
    prediction = tf.nn.softmax(tf.matmul(outputs[-1], weights['full']) + biases['full'])

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(training_iters):
            eeg_data, time_data = ne.next_eeg_seconds(5)
            time_until = ns.time_until_next_seizure_array(time_data, n_output)
            sess.run(optimizer, feed_dict={x: eeg_data, y: time_until})
            if i % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: eeg_data, y: time_until})
                print "Iter: %.0f, Accuracy: %.4f" % (i, acc)

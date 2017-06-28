import tensorflow as tf
import mnist
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt

trainImg, trainLabel = mnist.get_mnist_data()
n_examples, n_features, t_steps = trainImg.shape
n_epochs = 1000
n_classes = 10
n_hidden = 128
learning_rate = 1e-2

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
    sequences: a list of lists of type dtype where each element is a sequence
    Returns:
    A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape

x = tf.placeholder(tf.float32, [None, n_features, t_steps])
y = tf.sparse_placeholder(tf.int32)
seqLen = tf.placeholder(tf.int32, [None])

xx = tf.transpose(x, (2, 0, 1)) # (t_steps, batch_size, n_features)

lstm_cell = rnn.BasicLSTMCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, xx, seqLen, dtype=tf.float32, time_major=True)
outputs = tf.reshape(outputs, [-1, n_hidden])
weight = tf.Variable(tf.random_normal([n_hidden, n_classes+1]))
bias = tf.Variable(tf.constant(0., shape=[n_classes+1]))
pred = tf.matmul(outputs, weight) + bias
pred = tf.reshape(pred, [t_steps, -1, n_classes+1])

loss = tf.nn.ctc_loss(y, pred, seqLen)
cost = tf.reduce_mean(loss)

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
decoded, log_prob = tf.nn.ctc_greedy_decoder(pred, seqLen)
label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    batch_x = trainImg[0]
    batch_x = np.expand_dims(batch_x, 0)
    batch_y = trainLabel[0]
    batch_y_sparse = sparse_tuple_from(np.expand_dims(batch_y,0))
    trainSeq = [mnist.N_DIGITS] * 1
    for epoch in range(n_epochs):
        feed = {x: batch_x, y: batch_y_sparse, seqLen: trainSeq}
        batch_cost, prediction, _ = sess.run([cost, pred, optimizer], feed_dict=feed)
        print('epoch: {}, batch_cost: {:.3f}'.format(epoch, batch_cost))
        prediction = np.transpose(prediction, (1, 2, 0))
        prediction = np.squeeze(prediction, axis=0)
        if epoch % 50 == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.title.set_text(str(batch_y))
            ax1.imshow(batch_x[0], cmap='gray')
            ax2.imshow(prediction[:, :5], cmap='gray')
            #ax2.imshow(prediction, cmap='gray')
            plt.show()


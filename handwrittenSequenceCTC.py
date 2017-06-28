import tensorflow as tf
import mnist
from tensorflow.contrib import rnn
import numpy as np
import time

trainImg, trainLabel = mnist.get_mnist_data()
n_examples, n_features, t_steps = trainImg.shape
batch_size = 1
n_classes = 10
n_hidden = 128
n_layers = 1
learning_rate = 1e-5
n_epochs = 1000
n_batches_per_epoch = int(n_examples/batch_size)

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
    for epoch in range(n_epochs):
        train_cost = train_ler = 0
        start = time.time()
        for batch in range(n_batches_per_epoch):
            batch_x = trainImg[batch*batch_size: (batch+1)*batch_size]
            batch_y = trainLabel[batch*batch_size: (batch+1)*batch_size]
            batch_y = sparse_tuple_from(batch_y)
            trainSeq = [mnist.N_DIGITS] * batch_size
            feed = {x: batch_x, y: batch_y, seqLen: trainSeq}
            batch_cost, _ = sess.run([cost, optimizer], feed_dict=feed)
            train_cost += batch_cost * batch_size
            train_ler += sess.run(label_error_rate, feed_dict=feed) * batch_size
            print('batch {} in epoch: {}, batch_cost: {:.3f}'.format(batch, epoch, batch_cost))
        train_cost /= n_examples
        train_ler /= n_examples
        print('epoch {}/{}, train_cost={:.3f}, train_ler={:.3f}, time={:.3f}'.format(epoch, n_epochs, train_cost, train_ler, time.time()-start))

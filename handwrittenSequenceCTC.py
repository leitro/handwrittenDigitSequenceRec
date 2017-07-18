import tensorflow as tf
import mnist
from tensorflow.contrib import rnn
import numpy as np
import time
import matplotlib.pyplot as plt

N_DIGITS_TR = 2 # Training digits length
N_DIGITS_TE = 5 # Testing digits length

class SeqLearn():
    def __init__(self, datasets):
        self.trainImg, self.trainLabel, self.testImg, self.testLabel = datasets
        self.n_examples, self.n_features, self.t_steps = self.trainImg.shape
        self.batch_size = 200
        self.n_classes = 10
        self.n_hidden = 32
        self.n_layers = 1
        self.learning_rate = 1e-2
        self.n_epochs = 400
        self.n_batches_per_epoch = int(self.n_examples/self.batch_size)
        self.model()

    def model(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_features, None]) # (batch_size, n_features, time_steps)
        self.y = tf.sparse_placeholder(tf.int32)
        self.seqLen = tf.placeholder(tf.int32, [None])

        #<CNN>
        batch_s = tf.shape(self.x)[0]
        conv = tf.reshape(self.x, shape=[batch_s, self.n_features, -1, 1])
        w_conv = tf.Variable(tf.random_normal([5, 5, 1, 32]))
        b_conv = tf.Variable(tf.constant(0., shape=[32]))
        conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b_conv)
        conv = tf.nn.relu(conv)
        #conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shapeConv = tf.shape(conv) # (batch_size, features/2, time_step/2, channels==32)
        xx = tf.transpose(conv, (2, 0, 1, 3)) # (time/2, batch, features/2, channels==32)
        xx = tf.reshape(xx, [-1, batch_s, self.n_features*32]) # (time/2, batch, features/2 * 32)
        #</CNN>

        lstm_cell = rnn.BasicLSTMCell(self.n_hidden)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, xx, self.seqLen, dtype=tf.float32, time_major=True)
        outputs = tf.reshape(outputs, [-1, self.n_hidden])
        weight2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_classes+1]))
        bias2 = tf.Variable(tf.constant(0., shape=[self.n_classes+1]))
        pred = tf.matmul(outputs, weight2) + bias2
        self.pred = tf.reshape(pred, [-1, batch_s, self.n_classes+1])

        loss = tf.nn.ctc_loss(self.y, self.pred, self.seqLen)
        self.cost = tf.reduce_mean(loss)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.cost)
        self.decoded, log_prob = tf.nn.ctc_greedy_decoder(self.pred, self.seqLen)
        self.decoded_long, log_prob = tf.nn.ctc_greedy_decoder(self.pred, self.seqLen, merge_repeated=False)
        self.label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.y))

    def train(self, test_flag=True, visual_flag=True):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for epoch in range(1, self.n_epochs+1):
                train_cost = train_ler = 0
                start = time.time()
                for batch in range(self.n_batches_per_epoch):
                    batch_x = self.trainImg[batch*self.batch_size: (batch+1)*self.batch_size]
                    batch_y = self.trainLabel[batch*self.batch_size: (batch+1)*self.batch_size]
                    batch_y = self.sparse_tuple_from(batch_y)
                    trainSeq = [N_DIGITS_TR*28] * self.batch_size
                    feed = {self.x: batch_x, self.y: batch_y, self.seqLen: trainSeq}
                    batch_cost, _, prediction = sess.run([self.cost, self.optimizer, self.pred], feed_dict=feed)
                    train_cost += batch_cost * self.batch_size
                    train_ler += sess.run(self.label_error_rate, feed_dict=feed) * self.batch_size
                train_cost /= self.n_examples
                train_ler /= self.n_examples
                print('epoch {}/{}, train_cost={:.3f}, train_ler={:.3f}, time={:.3f}'.format(epoch, self.n_epochs, train_cost, train_ler, time.time()-start))

                if test_flag and (epoch % 10 == 0 or epoch == 1):
                    feed = {self.x: self.testImg, self.y: self.sparse_tuple_from(self.testLabel), self.seqLen: [N_DIGITS_TE*28] * len(self.testLabel)}
                    batch_cost, ler = sess.run([self.cost, self.label_error_rate], feed_dict=feed)
                    print('###TEST### batch_cost {:.3f}, label error rate {:.3f}'.format(batch_cost, ler))

                if visual_flag and (epoch % 10 == 0 or epoch == 1):
                    test_batch_s = 4
                    feed = {self.x: self.testImg[:test_batch_s], self.y: self.sparse_tuple_from(self.testLabel[:test_batch_s]), self.seqLen: [N_DIGITS_TE*28] * test_batch_s}
                    decodedRes, prediction, decodedResLong = sess.run([self.decoded, self.pred, self.decoded_long], feed_dict=feed)
                    print('###TEST### batch_cost {:.3f}, label error rate {:.3f}'.format(batch_cost, ler))
                    sparseRes = decodedRes[0]
                    res = sess.run(tf.sparse_to_dense(sparseRes[0], sparseRes[2], sparseRes[1]))
                    print('decodedRes:', res)
                    sparseLong = decodedResLong[0]
                    resL = sess.run(tf.sparse_to_dense(sparseLong[0], sparseLong[2], sparseLong[1]))
                    print('decoded without merge:', resL)
                    prediction0 = np.transpose(prediction[:, 0, :], (1, 0))
                    prediction1 = np.transpose(prediction[:, 1, :], (1, 0))
                    prediction2 = np.transpose(prediction[:, 2, :], (1, 0))
                    prediction3 = np.transpose(prediction[:, 3, :], (1, 0))
                    fig = plt.figure()
                    ax1 = fig.add_subplot(421)
                    ax1.title.set_text(str(self.testLabel[0]))
                    ax1.imshow(self.testImg[0], cmap='gray')
                    ax2 = fig.add_subplot(423)
                    ax2.imshow(prediction0, cmap='gray')
                    ax1 = fig.add_subplot(422)
                    ax1.title.set_text(str(self.testLabel[1]))
                    ax1.imshow(self.testImg[1], cmap='gray')
                    ax2 = fig.add_subplot(424)
                    ax2.imshow(prediction1, cmap='gray')
                    ax1 = fig.add_subplot(425)
                    ax1.title.set_text(str(self.testLabel[2]))
                    ax1.imshow(self.testImg[2], cmap='gray')
                    ax2 = fig.add_subplot(427)
                    ax2.imshow(prediction2, cmap='gray')
                    ax1 = fig.add_subplot(426)
                    ax1.title.set_text(str(self.testLabel[3]))
                    ax1.imshow(self.testImg[3], cmap='gray')
                    ax2 = fig.add_subplot(428)
                    ax2.imshow(prediction3, cmap='gray')
                    plt.savefig('visTest/'+str(epoch)+'.jpg')
                    plt.close(fig)
                    #plt.show()

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        #Create a sparse representention of x.
        #Args:
        #sequences: a list of lists of type dtype where each element is a sequence
        #Returns:
        #A tuple with (indices, values, shape)
        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
        return indices, values, shape

if __name__ == '__main__':
    model = SeqLearn(mnist.get_mnist_data(N_DIGITS_TR, N_DIGITS_TE))
    model.train()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

N_DIGITS_TRAIN = 5
N_DIGITS_TEST = 8

def get_mnist_data(n_digits_train=N_DIGITS_TRAIN, n_digits_test=N_DIGITS_TEST):
    mnist = input_data.read_data_sets('.', one_hot=False)
    trainImg = mnist.train.images.reshape(-1, 28, 28)
    trainLabel = mnist.train.labels
    testImg = mnist.test.images.reshape(-1, 28, 28)
    testLabel = mnist.test.labels
    n_total = mnist.train.num_examples
    n_total_test = mnist.test.num_examples

    new_trainImg_list = []
    for i in range(n_total//n_digits_train):
        split_list = np.vsplit(trainImg[i*n_digits_train: (i+1)*n_digits_train], n_digits_train)
        split_list = [i.reshape(28, 28) for i in split_list]
        digits = np.hstack(split_list)
        new_trainImg_list.append(np.expand_dims(digits, 0))
    new_trainImg = np.vstack(new_trainImg_list)
    new_trainLabel = trainLabel.reshape(-1, n_digits_train)

    new_testImg_list = []
    for i in range(n_total_test//n_digits_test):
        split_list = np.vsplit(testImg[i*n_digits_test: (i+1)*n_digits_test], n_digits_test)
        split_list = [i.reshape(28, 28) for i in split_list]
        digits = np.hstack(split_list)
        new_testImg_list.append(np.expand_dims(digits, 0))
    new_testImg = np.vstack(new_testImg_list)
    new_testLabel = testLabel.reshape(-1, n_digits_test)
    return new_trainImg, new_trainLabel.tolist(), new_testImg, new_testLabel.tolist()

if __name__ == '__main__':
    trainImg, trainLabel, testImg, testLabel = get_mnist_data()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.title.set_text(str(trainLabel[0]))
    ax1.imshow(trainImg[0], cmap='gray')
    ax2.title.set_text(str(testLabel[0]))
    ax2.imshow(testImg[0], cmap='gray')
    plt.show()

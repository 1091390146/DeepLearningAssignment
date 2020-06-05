import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File(r'E:\untitled\DeepLearnningAssignment\L1W2A2\datasets\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    test_dataset = h5py.File(r'E:\untitled\DeepLearnningAssignment\L1W2A2\datasets\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    print('训练样本数={}'.format(train_set_x_orig.shape))
    print('训练样本对应的标签={}'.format(train_set_y_orig.shape))
    '''X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，第二维中取第0个数据，直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
        X[n,:]是取第1维中下标为n的元素的所有值。
        X[1,:]即取第一维中下标为1的元素的所有值。
        X[:,  m:n]，即取所有数据的第m到n-1列数据，含左不含右。
    '''
    print('前10张训练样本标签={}'.format(train_set_y_orig[:, :10]))
    print('测试样本数={}'.format(test_set_x_orig.shape))
    print('测试样本对应的标签={}'.format(test_set_y_orig.shape))
    print('{}'.format(classes))

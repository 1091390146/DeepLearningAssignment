import matplotlib.pyplot as plt
import numpy as np
'''
Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法。当我们面临机器学习问题时，便可根据下图来选择相应的方法。Sklearn具有以下特点：
简单高效的数据挖掘和数据分析工具
让每个人能够在复杂环境中重复使用
建立NumPy、Scipy、MatPlotLib之上
'''
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, X, y, single=True):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    '''
    根据传入的两个一维数组参数生成两个数组元素的列表。

    如果第一个参数是xarray，维度是xdimesion，

    第二个参数是yarray，维度是ydimesion。

    那么生成的第一个二维数组是以xarray为行，共ydimesion行的向量；

    而第二个二维数组是以yarray的转置为列，共xdimesion列的向量。
    '''
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    '''
    np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    
    首先声明两者所要实现的功能是一致的（将多维数组降位一维）。
    这点从两个单词的意也可以看出来，ravel(散开，解开)，flatten（变平）。
    两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，
    对拷贝所做的修改不会影响（reflects）原始矩阵，
    而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），
    会影响（reflects）原始矩阵。
    '''
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    # plt.ylabel('x2')
    # plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    if single:
        plt.show()


def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def load_planar_dataset():
    #seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
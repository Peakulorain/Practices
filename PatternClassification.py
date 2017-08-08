###python3.5.3
###tensorflow1.2.1
###windows10

###<<神经网络与机器学习>>Page94  计算机实验：模式分类

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(layername,inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.variable_scope(layername,reuse=None):
        Weights = tf.get_variable("weights",shape=[in_size, out_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[1, out_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
    #Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def produceData(r,w,d,num):
    r1 = r-w/2
    r2 = r+w/2
    #上半圆
    theta1 =  np.random.uniform(0, np.pi ,num)
    X_Col1 = np.random.uniform( r1*np.cos(theta1),r2*np.cos(theta1),num)[:, np.newaxis]
    X_Row1 = np.random.uniform(r1*np.sin(theta1),r2*np.sin(theta1),num)[:, np.newaxis]
    Y_label1 = np.ones(num) #类别标签为1
    #下半圆
    theta2 = np.random.uniform(-np.pi, 0 ,num)
    X_Col2 = (np.random.uniform( r1*np.cos(theta2),r2*np.cos(theta2),num) + r)[:, np.newaxis]
    X_Row2 = (np.random.uniform(r1 * np.sin(theta2), r2 * np.sin(theta2), num) -d)[:, np.newaxis]
    Y_label2 = -np.ones(num) #类别标签为-1,注意：由于采取双曲正切函数作为激活函数，类别标签不能为0
    #合并
    X_Col = np.vstack((X_Col1, X_Col2))
    X_Row = np.vstack((X_Row1, X_Row2))
    X = np.hstack((X_Col, X_Row))
    Y_label = np.hstack((Y_label1,Y_label2))
    Y_label.shape = (num*2 , 1)
    return X,Y_label

def produce_random_data(r,w,d,num):
    X1 = np.random.uniform(-r-w/2,2*r+w/2, num)
    X2 = np.random.uniform(-r - w / 2-d, r+w/2, num)
    X = np.vstack((X1, X2))
    return X.transpose()
###define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])
###添加隐藏层
l1 = add_layer("layer1",xs, 2, 20, activation_function=tf.tanh)
###添加输出层
prediction = add_layer("layer2",l1, 20, 1, activation_function=tf.tanh)

###MSE 均方误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
###优化器选取 学习率设置 此处学习率置为0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

###tensorflow变量初始化，打开会话
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def collect_boundary_data(v_xs):
    global prediction
    X = np.empty([1,2])
    X = list()
    for i in range(len(v_xs)):
        x_input = v_xs[i]
        x_input.shape = [1,2]
        y_pre = sess.run(prediction, feed_dict={xs: x_input})
        if abs(y_pre - 0) < 0.5:
            X.append(v_xs[i])
    return np.array(X)

def compute_testProbabilities(test_x_data):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: test_x_data})
    return y_pre

def main():
    x_data , y_label= produceData(10,6,-4,1000)
    ###训练
    for i in range(2000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_label})
    ###产生空间随机数据
    X_NUM = produce_random_data(10, 6, -4, 5000)
    ###边界数据采样
    X_b = collect_boundary_data(X_NUM)
    ###画出数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ###设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.scatter(x_data[:, 0], x_data[:, 1], marker='x')
    ###用采样的边界数据拟合边界曲线 7次曲线最佳
    z1 = np.polyfit(X_b[:, 0], X_b[:, 1], 7)
    p1 = np.poly1d(z1)
    x = X_b[:, 0]
    x.sort()
    yvals = p1(x)
    plt.plot(x, yvals, 'r', label='boundray line')
    plt.legend(loc=4)
    #plt.ion()
    plt.show()
    print('DONE!')

if __name__ == '__main__':
    main()
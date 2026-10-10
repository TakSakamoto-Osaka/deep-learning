import numpy as np
import sys, os
import matplotlib.pyplot as plt


sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
   
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch    


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    '''勾配（偏微分）を求める'''
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h 
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 値を元に戻す

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    '''勾配降下法'''
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == "__main__":
    '''
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    #y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    
    #print(sum_squared_error(np.array(y), np.array(t)))
    print(cross_entropy_error(np.array(y), np.array(t)))

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    print(x_train.shape)  # (60000, 784)
    print(t_train.shape)  # (60000, 10)
    
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()
   
    print(numerical_diff(function_1, 5))
    print(numerical_diff(function_1, 10)) 
   
    x0 = np.arange(-3, 3, 0.25)
    x1 = np.arange(-3, 3, 0.25)
    
    
    X, Y = np.meshgrid(x0, x1)
    Z = function_2(np.array([X, Y]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X0") 
    ax.set_ylabel("X1")
    ax.set_zlabel("f(x0, x1)")
    
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='blue', alpha=0.5)
    
    ax.grid(True)
    
    plt.show()
    
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))
    '''
    
    init_x = np.array([-3.0, 4.0])
    # print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
    # print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
    print(gradient_descent(function_2, init_x=init_x, lr=10, step_num=100))
    
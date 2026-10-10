import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # 重みをランダムに初期化
        self.W = np.random.randn(2, 3)  # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    # 損失関数の値を求める
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
    

def f(W):
    return net.loss(x, t)


if __name__ == "__main__":
    net = simpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    # 予測結果のクラス番号
    np.argmax(p)

    # 教師ラベル
    t = np.array([0, 0, 1])
    # 損失関数の値を求める
    loss = net.loss(x, t)
    print(loss)

    dW = numerical_gradient(f, net.W)
    print(dW)
    
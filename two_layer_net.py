import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    """2層のニューラルネットワーク

    Parameters
    ----------
    input_size : 入力サイズ
    hidden_size : 隠れ層のニューロン数
    output_size : 出力サイズ
    weight_init_std : 重みの標準偏差（ガウス分布）
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """推論（順伝播）"""
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """損失関数の値を求める"""
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        """認識精度を求める"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        """勾配（重みの更新量）を求める"""
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


if __name__ == "__main__":
    import numpy as np
    from dataset.mnist import load_mnist
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    train_loss_list = []
    
    # ハイパーパラメータ
    iters_num = 10000  # 繰り返しの回数
    train_size = x_train.shape[0]
    batch_size = 100  # ミニバッチのサイズ
    learning_rate = 0.1  # 学習率
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 勾配の計算
        grad = network.numerical_gradient(x_batch, t_batch)
        
        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)    
    
    pass

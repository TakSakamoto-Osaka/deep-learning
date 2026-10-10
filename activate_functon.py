import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


class Wave():
    def __init__(self, x, y, label, color):
        self.x = x
        self.y = y
        self.label = label
        self.color = color

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=np.int32)

def relu_function(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def draw(waves: Wave, title):
    # 波形プロット
    plt.figure(figsize=(8,5))
    
    for wave in waves:
        plt.plot(wave.x, wave.y, label=wave.label, color=wave.color)
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.grid(0, color="black", linewidth=0.5, linestyle="--")
    
    plt.legend()
    plt.show() 
    
    plt.close() 

if __name__ == "__main__":
    # 日本語フォントの設定
    '''
    rcParams["font.family"] = "Meiryo"
    
    waves = []
    
    x = np.arange(-5.0, 5.0, 0.01)   # -5.0から5.0まで0.01刻みでデータ生成
    
    y0 = sigmoid(x)          # シグモイド関数の値を計算
    waves.append(Wave(x, y0, "シグモイド関数", "blue"))
    
    y1 = step_function(x)
    waves.append(Wave(x, y1, "ステップ関数", "red"))
    
    y2 = relu_function(x)
    waves.append(Wave(x, y2, "ReLU関数", "green"))
    
    draw(waves, "活性化関数")
    '''
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))  # ソフトマックス関数の出力の合計は1になる
    
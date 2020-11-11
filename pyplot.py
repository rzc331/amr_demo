import numpy as np
# import matplotlib.pyplot as plt
from pylab import *
# input('1')


def draw_plot(X, y_pred, y_std, X_training, y_training, title='', X_next=False, X_next_idx=0):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 5))
        plt.plot(X, y_pred)
        plt.fill_between(X, y_pred - y_std, y_pred + y_std, alpha=0.2)
        # plt.scatter(X, y, c='k', s=20)
        '''Highlight observed points'''
        plt.scatter(X_training, y_training, c='green', s=100)
        if X_next:
            plt.vlines(X[X_next_idx], 0, y[X_next_idx], colors="r", linestyles="dashed")
        plt.title(title)
        plt.show()
        # plt.savefig('al_figures/initial')

# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8, 6), dpi=80)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(1,1,1)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)
# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plot(X, S, color="green", linewidth=1.0, linestyle="-")

plt.vlines(0.3, 0, 0.5, colors="r", linestyles="dashed")

# 设置横轴的上下限
xlim(-4.0,4.0)

# 设置横轴记号
xticks(np.linspace(-4,4,9,endpoint=True))

# 设置纵轴的上下限
ylim(-1.0,1.0)

# 设置纵轴记号
yticks(np.linspace(-1,1,5,endpoint=True))

show()


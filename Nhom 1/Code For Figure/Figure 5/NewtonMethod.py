# Hỗ trợ cả Python 2 và Python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2 * np.random.randn(1000, 1)

# Xây dựng Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)

def cost(w):
    return 0.5 / Xbar.shape[0] * np.linalg.norm(y - Xbar.dot(w), 2)**2

def grad(w):
    return 1 / Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)

def hessian(w):
    return 1 / Xbar.shape[0] * Xbar.T.dot(Xbar)

def newton_method(w_init, grad, hessian):
    H_inv = np.linalg.inv(hessian(w_init))
    w_new = w_init - H_inv.dot(grad(w_init))
    return [w_init, w_new]

# Phương pháp Newton
w_init = np.array([[2], [1]])
w_newton = newton_method(w_init, grad, hessian)

print(len(w_newton), w_newton[-1])

# Biểu đồ đường đồng mức
N = X.shape[0]
a1 = np.linalg.norm(y, 2)**2 / N
b1 = 2 * np.sum(X) / N
c1 = np.linalg.norm(X, 2)**2 / N
d1 = -2 * np.sum(y) / N
e1 = -2 * X.T.dot(y) / N

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(-10, 10, delta)
yg = np.arange(-10, 10, delta)
Xg, Yg = np.meshgrid(xg, yg)
Z = a1 + Xg**2 + b1 * Xg * Yg + c1 * Yg**2 + d1 * Xg + e1 * Yg

def save_gif_newton():
    it = len(w_newton)
    fig, ax = plt.subplots(figsize=(8, 8))    
    plt.cla()
    plt.axis([-10, 10, -10, 10])
    
    def update(ii):
        if ii == 0:
            plt.cla()
            CS = plt.contour(Xg, Yg, Z, 100)
            plt.clabel(CS, inline=0.1, fontsize=10)
            plt.plot(w_exact[0], w_exact[1], 'go', label='Giải pháp chính xác')
        else:
            plt.plot([w_newton[ii-1][0], w_newton[ii][0]], 
                     [w_newton[ii-1][1], w_newton[ii][1]], 'r-')
        plt.plot(w_newton[ii][0], w_newton[ii][1], 'ro', markersize=4) 
        xlabel = 'Lần lặp = %d/%d' % (ii, it-1)
        xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w_newton[ii]))
        ax.set_xlabel(xlabel)
        ax.set_title('Hồi quy tuyến tính với Phương pháp Newton')
        ax.legend()
        return ax
    
    anim1 = FuncAnimation(fig, update, frames=np.arange(0, it), interval=500)
    fn = 'LR_Newton_contours.gif'
    anim1.save(fn, dpi=100, writer='imagemagick')

save_gif_newton()

# Hàm mất mát 
print(cost(w_newton[-1]))
loss = np.zeros((len(w_newton), 1))
for i in range(len(w_newton)):
    loss[i] = cost(w_newton[i])
print(loss)

plt.plot(range(len(w_newton)), loss, 'b')
plt.xlabel('Lần lặp')
plt.ylabel('Mất mát')
plt.title('Hồi quy tuyến tính với Phương pháp Newton')
plt.show()

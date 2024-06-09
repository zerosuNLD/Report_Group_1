import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Hàm tính cost
def cost(x):
    m = A.shape[0]
    return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

# Hàm tính gradient
def grad(x, i):
    return A[i].reshape(-1, 1) * (A[i].dot(x) - b[i])

# Hàm kiểm tra gradient
def check_grad(x):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        g[i] = (cost(x1) - cost(x2))/(2*eps)
    
    g_grad = np.zeros_like(x)
    for i in range(A.shape[0]):
        g_grad += grad(x, i)
    g_grad /= A.shape[0]

    if np.linalg.norm(g-g_grad) > 1e-5:
        print("WARNING: CHECK GRADIENT FUNCTION!")

# Thuật toán Stochastic Gradient Descent
def sgd(x_init, learning_rate):
    x_list = [x_init]
    m = A.shape[0]

    iteration = 0
    while True:
        for j in range(m):
            x_new = x_list[-1] - learning_rate * grad(x_list[-1], j)
            x_list.append(x_new)
            if np.linalg.norm(grad(x_new, j)) / m < 1e-5:  # Điều kiện dừng SGD
                return x_list
        iteration += 1

# Dữ liệu

A = np.random.rand(1000, 1)
b = 4 + 3 * A + .2*np.random.randn(1000, 1)

# Thêm cột 1 vào A
ones = np.ones((A.shape[0], 1), dtype=np.int8)
A = np.concatenate((ones, A), axis=1)

# Đường khởi tạo ngẫu nhiên
x_init = np.array([[1.], [2.]])
check_grad(x_init)

# Chạy Stochastic Gradient Descent
learning_rate = 0.1
x_list = sgd(x_init, learning_rate)

# Tạo meshgrid cho biểu đồ đường đồng mức
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = cost(t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Vẽ biểu đồ đường đồng mức và điểm dữ liệu
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ đường đồng mức
CS = ax1.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 20))
line1, = ax1.plot([], [], 'ro-', markersize=5)  # Đường đi của SGD
ax1.set_title('Contour Plot and SGD Path')
ax1.set_xlabel('theta_0')
ax1.set_ylabel('theta_1')
ax1.set_xlim([-10, 10])
ax1.set_ylim([-1, 4])

# Biểu đồ dữ liệu
ax2.scatter(A[:, 1], b, label='Data Points')
line2, = ax2.plot([], [], 'r-', label='Regression Line')  # Đường hồi quy
ax2.set_title('Data and Regression Line')
ax2.set_xlabel('Feature')
ax2.set_ylabel('Target')
ax2.legend()

# Lưu các đường dự đoán cũ
prev_lines = []

# Hàm cập nhật cho animation
def update(i):
    # Cập nhật đường đi SGD trên biểu đồ đường đồng mức
    line1.set_data([x[0][0] for x in x_list[:i+1]], [x[1][0] for x in x_list[:i+1]])
    # Cập nhật đường hồi quy trên biểu đồ dữ liệu
    y_pred = A.dot(x_list[i])
    line2.set_data(A[:, 1], y_pred)
    
    # Vẽ các đường dự đoán cũ màu xám
    if i > 0:
        for line in prev_lines:
            line.remove()
        prev_lines.clear()
        for j in range(i):
            y_prev = A.dot(x_list[j])
            line, = ax2.plot(A[:, 1], y_prev, 'gray', alpha=0.5)
            prev_lines.append(line)
    
    return line1, line2

# Tạo animation
iters = np.arange(len(x_list))
line_ani = FuncAnimation(fig, update, iters, interval=100, blit=True)

# Lưu animation
fn = 'GD.gif'
line_ani.save(fn, dpi=100, writer='pillow')

plt.show()

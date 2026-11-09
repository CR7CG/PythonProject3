import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 目标函数、梯度、海瑟矩阵 =====================
def func(x):
    """目标函数 f(x,y) = x^4 + y^4 - x^2 - y^2"""
    return x[0]**4 + x[1]**4 - x[0]**2 - x[1]**2

def grad(x):
    """解析梯度"""
    return np.array([
        4 * x[0]**3 - 2 * x[0],
        4 * x[1]**3 - 2 * x[1]
    ])

def hess(x):
    """解析海瑟矩阵"""
    return np.array([
        [12 * x[0]**2 - 2, 0],
        [0, 12 * x[1]**2 - 2]
    ])

# ===================== 2. 纯牛顿法实现（无线搜索，步长=1） =====================
def pure_newton(x0, max_iter=20):
    """
    纯牛顿法迭代
    :param x0: 初始点，列表/数组
    :param max_iter: 最大迭代次数
    :return: 迭代轨迹数组, 每步函数值数组
    """
    x = np.array(x0, dtype=float)
    traj = [x.copy()]       # 保存迭代点轨迹
    f_history = [func(x)]   # 保存每步函数值

    for _ in range(max_iter):
        g = grad(x)         # 计算梯度
        H = hess(x)         # 计算海瑟矩阵
        d = -np.linalg.solve(H, g)  # 解线性方程组 H*d = -g，得到牛顿方向
        x = x + d          # 步长固定为1，无线搜索
        traj.append(x.copy())
        f_history.append(func(x))

    return np.array(traj), np.array(f_history)

# ===================== 3. 可视化函数 =====================
def plot_contour(traj_list, labels, title):
    """绘制函数等高线 + 迭代轨迹"""
    # 生成网格
    xg = np.linspace(-1.5, 1.5, 200)
    yg = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(xg, yg)
    Z = X**4 + Y**4 - X**2 - Y**2

    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, Z, levels=20, cmap='viridis')

    # 标记鞍点 (0,0)
    plt.plot(0, 0, 'go', markersize=10, label='鞍点 (0,0)')
    # 标记4个全局极小点
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            plt.plot(s1/np.sqrt(2), s2/np.sqrt(2), 'r*', markersize=12, label='全局极小点' if s1==1 and s2==1 else "")

    # 绘制各组迭代轨迹
    styles = ['r-o', 'b-s']
    for i, traj in enumerate(traj_list):
        plt.plot(traj[:, 0], traj[:, 1], styles[i], markersize=4,
                 linewidth=1.2, label=labels[i])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_f_curve(f_list, labels, title):
    """绘制函数值随迭代步数变化曲线"""
    plt.figure(figsize=(8, 5))
    for i, f_hist in enumerate(f_list):
        plt.plot(range(len(f_hist)), f_hist, linewidth=1.5, label=labels[i])

    plt.xlabel('迭代步数 k')
    plt.ylabel('f(x_k)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ===================== 4. 主程序：运行实验一 =====================
if __name__ == '__main__':
    # 两个初始点
    x0_saddle = [0.1, 0.1]   # 靠近鞍点
    x0_min = [1.0, 1.0]      # 靠近极小点

    # 运行纯牛顿法
    traj1, f1 = pure_newton(x0_saddle)
    traj2, f2 = pure_newton(x0_min)

    # ========== 打印结果 ==========
    print("=" * 50)
    print("【实验一：纯牛顿法结果】")
    print(f"初始点 (0.1, 0.1)：")
    print(f"  最终迭代点：{traj1[-1].round(6)}")
    print(f"  最终函数值：{f1[-1]:.6f}")
    print(f"  迭代步数：{len(f1)-1}")
    print()
    print(f"初始点 (1.0, 1.0)：")
    print(f"  最终迭代点：{traj2[-1].round(6)}")
    print(f"  最终函数值：{f2[-1]:.6f}")
    print(f"  迭代步数：{len(f2)-1}")
    print("=" * 50)

    # ========== 绘图 ==========
    # 图1：等高线迭代轨迹对比
    plot_contour([traj1, traj2],
                 ['初始点(0.1,0.1)', '初始点(1.0,1.0)'],
                 '实验一：纯牛顿法迭代轨迹')

    # 图2：函数值收敛曲线对比
    plot_f_curve([f1, f2],
                 ['初始点(0.1,0.1)', '初始点(1.0,1.0)'],
                 '实验一：纯牛顿法函数值变化曲线')

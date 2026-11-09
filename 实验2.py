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

# ===================== 2. 核心工具函数 =====================
def regularize_hessian(H, eps=1e-3):
    """
    海瑟矩阵正定修正：加 tau*I，保证最小特征值 >= eps
    返回: 修正后的矩阵B, 惩罚因子tau
    """
    eigvals = np.linalg.eigvalsh(H)
    lam_min = eigvals.min()
    tau = max(0.0, eps - lam_min)
    B = H + tau * np.eye(len(H))
    return B, tau

def backtracking(x, d, c1=1e-4, rho=0.5, alpha=1.0):
    """Armijo 回溯线搜索"""
    g = grad(x)
    fx = func(x)
    while func(x + alpha * d) > fx + c1 * alpha * np.dot(g, d):
        alpha *= rho
    return alpha

# ===================== 3. 算法实现 =====================
def pure_newton(x0, max_iter=20):
    """纯牛顿法（无线搜索，步长=1），用作对照组"""
    x = np.array(x0, dtype=float)
    f_history = [func(x)]
    for _ in range(max_iter):
        g = grad(x)
        H = hess(x)
        d = -np.linalg.solve(H, g)
        x = x + d
        f_history.append(func(x))
    return np.array(f_history)

def modified_newton(x0, max_iter=50, eps=1e-3):
    """修正牛顿法：正定修正 + Armijo线搜索"""
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    f_history = [func(x)]
    tau_history = []

    for _ in range(max_iter):
        g = grad(x)
        H = hess(x)
        B, tau = regularize_hessian(H, eps=eps)
        tau_history.append(tau)

        d = -np.linalg.solve(B, g)       # 解修正后的线性方程组
        alpha = backtracking(x, d)       # 线搜索求步长
        x = x + alpha * d

        traj.append(x.copy())
        f_history.append(func(x))

    return np.array(traj), np.array(f_history), np.array(tau_history)

# ===================== 4. 可视化 =====================
def plot_convergence(f_list, labels, title, f_opt=-0.5):
    """半对数误差收敛曲线：误差 = |f(x_k) - f*|"""
    plt.figure(figsize=(8, 5))
    for i, f_hist in enumerate(f_list):
        error = np.abs(np.array(f_hist) - f_opt)
        plt.semilogy(range(len(error)), error, linewidth=1.5, label=labels[i])

    plt.xlabel('迭代步数 k')
    plt.ylabel('|f(x_k) - f*|')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.show()

def plot_tau(tau_hist):
    """τ_k 随迭代步数变化曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(tau_hist)), tau_hist, 'b-o', markersize=4)
    plt.xlabel('迭代步数 k')
    plt.ylabel('τ_k')
    plt.title('实验二：惩罚因子 τ_k 变化曲线')
    plt.grid(alpha=0.3)
    plt.show()

# ===================== 5. 主程序：运行实验二 =====================
if __name__ == '__main__':
    x0 = [0.1, 0.1]   # 靠近鞍点初始点
    f_opt = -0.5

    # 运行两种算法
    f_pure = pure_newton(x0)
    traj_mod, f_mod, tau_hist = modified_newton(x0)

    # ========== 打印结果 ==========
    print("=" * 60)
    print("【实验二：修正牛顿法结果】")
    print(f"初始点：{x0}")
    print(f"修正牛顿最终点：{traj_mod[-1].round(6)}")
    print(f"修正牛顿最终函数值：{f_mod[-1]:.6f}")
    print(f"总迭代步数：{len(f_mod)-1}")
    print(f"\n前6步 τ 值：{tau_hist[:6].round(6)}")
    print("=" * 60)

    # ========== 绘图 ==========
    # 图1：纯牛顿 vs 修正牛顿 半对数收敛对比
    plot_convergence([f_pure, f_mod],
                     ['纯牛顿法', '修正牛顿法'],
                     '实验二：纯牛顿法 vs 修正牛顿法 收敛对比')

    # 图2：τ_k 变化曲线
    plot_tau(tau_hist)

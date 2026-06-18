from qutip import *
import numpy as np
from qudpy.Classes import *
import qudpy.plot_functions as pf
import ufss  # 图生成工具

w = 1.0
w0 = 1.0
g = 1  # 腔模-自旋相互作用的初始耦合强度，仅用于演示
gc = np.sqrt(w * w0)/2  # 临界耦合强度

kappa = 0.05
gamma = 0.15
M = 18    # 腔模的 Fock 基维数；耦合更强时可取更大
N = 6    # 自旋数目
j = N/2  # 自旋总角动量 J
n = N+1  # 总自旋算符的维数

print("critical coupling strength = ", gc)

a = tensor(destroy(M), qeye(n))
Jp = tensor(qeye(M), jmat(j, '+'))
Jm = tensor(qeye(M), jmat(j, '-'))
Jz = tensor(qeye(M), jmat(j, 'z'))

H0 = w * a.dag() * a + w0 * Jz  # 默认基础哈密顿量
H1 = (a + a.dag()) * (Jp + Jm)/np.sqrt(N)  # 腔内相互作用项
H = H0 + g * H1

print("dimensionality of Hilbert-space: ", H.shape)

# 与谐振腔热浴耦合时的平均热光子数
n_th = 0.25
c_ops = [np.sqrt(kappa * (n_th + 1)) * a, np.sqrt(kappa * n_th) * a.dag(),np.sqrt(gamma)*Jz]

#c_ops = [sqrt(kappa) * a, sqrt(gamma) * Jm]

g_vec = np.linspace(0.01, 1, 20)

# 哈密顿量 H = H0 + g * H1 对应的稳态
rho_ss_list = [steadystate(H0 + g * H1, c_ops) for g in g_vec]

# 计算腔中光子数等可观测量的期望值
n_ss_vec = expect(a.dag() * a, rho_ss_list)
n2_ss_vec = expect(a.dag() * a*a.dag() * a, rho_ss_list)
Jz_ss_vec = expect(Jz, rho_ss_list)

a_ss_vec = expect(a, rho_ss_list)



fig, axes = plt.subplots(1, 3, sharex=True, figsize=(12,4))

axes[0].plot(g_vec, n_ss_vec, 'r', linewidth=2, label="cavity occupation")
axes[0].set_ylim(0, max(n_ss_vec))
axes[0].set_ylabel("<n>", fontsize=16)
axes[0].set_xlabel("g", fontsize=16)

axes[1].plot(g_vec, Jz_ss_vec, 'r', linewidth=2, label="<Jz>")
axes[1].set_ylim(-j, j)
axes[1].set_ylabel(r"$\langle J_z\rangle$", fontsize=16)
axes[1].set_xlabel("g", fontsize=16)

axes[2].plot(g_vec, abs(a_ss_vec), 'r', linewidth=2, label="<a>")

fig.tight_layout()

rho_ss_sublist = rho_ss_list[::5]

xvec = np.linspace(-3, 3, 200)

fig_grid = (3, len(rho_ss_sublist))
fig = plt.figure(figsize=(3 * len(rho_ss_sublist), 9))

for idx, rho_ss in enumerate(rho_ss_sublist):
    # 对自旋自由度做偏迹，只保留腔模密度矩阵
    rho_ss_cavity = ptrace(rho_ss, 0)

    # 计算其 Wigner 函数
    W = wigner(rho_ss_cavity, xvec, xvec)

    # 绘制 Wigner 函数
    ax = plt.subplot2grid(fig_grid, (0, idx))
    ax.contourf(xvec, xvec, W, 100)

    # 绘制 Fock 态分布
    ax = plt.subplot2grid(fig_grid, (1, idx))
    ax.bar(np.arange(0, M), np.real(rho_ss_cavity.diag()), color="blue", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, M)


# 绘制不同耦合强度下的腔占据情况
ax = plt.subplot2grid(fig_grid, (2, 0), colspan=fig_grid[1])
ax.plot(g_vec, n_ss_vec, 'r', linewidth=2, label="cavity occupation")
ax.set_xlim(0, max(g_vec))
ax.set_ylim(0, max(n_ss_vec) * 1.2)
ax.set_ylabel("Cavity gnd occ. prob.", fontsize=16)
ax.set_xlabel("interaction strength", fontsize=16)

for g in g_vec[::4]:
    ax.plot([g, g], [0, max(n_ss_vec) * 1.2], 'b:', linewidth=2.5)


# 设置测试所需的双边图
# `DiagramGenerator` 类，简称 DG
DG = ufss.DiagramGenerator
# 初始化模块
R3rd = DG()  # DG 接受一个关键字参数，默认 detection_type='polarization'
# 图生成器需要知道相位匹配/相位循环条件
R3rd.set_phase_discrimination([(0, 1), (1, 0), (1, 0)])  # 设置重相位图 R1、R2、R3 的相位匹配条件
# 设置脉冲 0、1、2 和本振的脉冲宽度
d0 = 2
d1 = 4
d2 = 4
dlo = 6
# 设置各脉冲的时间窗口
t0 = np.array([-d0 / 2, d0 / 2])
t1 = np.array([-d1 / 2, d1 / 2])
t2 = np.array([-d2 / 2, d2 / 2])
tlo = np.array([-dlo / 2, dlo / 2])
all_pulse_intervals = [t0, t1, t2, tlo]

# 通过设置 DG 的 efield_times 属性传入这些脉冲时间窗口
R3rd.efield_times = all_pulse_intervals
# 在脉冲到达时刻 0、100、200 以及探测时刻 300 处生成图
[R3, R1, R2] = R3rd.get_diagrams([0, 100, 200, 300])
rephasing = [R1, R2, R3]
print('the rephasing diagrams are R1, R2 and R3 ', rephasing)

# 设置条件并生成非重相位图 R4、R5、R6
R3rd.set_phase_discrimination([(1, 0), (0, 1), (1, 0)])
[R6, R4, R5] = R3rd.get_diagrams([0, 100, 200, 200])
nonrephasing = [R4, R5, R6]
print('the non-rephasing diagrams are R4, R5 and R6', nonrephasing)


w  = 1.0
w0 = 1.0


gc = np.sqrt(w * w0)/2  # 临界耦合强度；此处等于 1/2
g = 0.5  # 耦合强度；此处取临界耦合的一半
kappa = 0.05
gamma = 0.15
#M = 10
#N = 2
j = N/2    # 这里是实数
n = N+1    # 对 qeye() 来说需要是整数

print("critical coupling strength = ",gc)

a = tensor(destroy(M), qeye(n))
Jp = tensor(qeye(M), jmat(j, '+'))
Jm = tensor(qeye(M), jmat(j, '-'))
Jz = tensor(qeye(M), jmat(j, 'z'))

H0 = w * a.dag() * a + w0 * Jz
H1 = (a + a.dag()) * (Jp + Jm)/np.sqrt(N)
H = H0 + g * H1

print("dimensionality of Hilbert-space: ",H.shape)

en,T = H.eigenstates()

Hd=H.transform(T)

# 与谐振腔热浴耦合时的平均热光子数
n_th = 0.25
c1 = np.sqrt(kappa * (n_th + 1)) * a
c2 = np.sqrt(kappa * n_th) * a.dag()
c3 = np.sqrt(gamma)*Jz



c1d =c1.transform(T)
c2d =c2.transform(T)
c3d =c3.transform(T)


# A 和 A.dag() 现在对应外部光场耦合导致的激发/退激发算符
# 因此也需要把它们变换到本征基中
A = a.transform(T)

mud = A + A.dag()



print("finding steady-states of the DM")
# 在本征基中把密度矩阵定义为稳态
rhoSS=steadystate(Hd, [c1d,c2d,c3d])


# 稳态系统的性质（要使用变换后的算符）
n_ss = expect(A.dag() * A, rhoSS)
n2_ss = expect(A.dag()*A*A.dag()*A,rhoSS)

print("Steady State Properties")
print("<n> = ",n_ss)
print("<n^2> = ",n2_ss)
print("<(n-<n>)^2> = ",n2_ss-n_ss*n_ss)


sys2 = System(H=Hd, a=A, u=mud, c_ops=[c1d, c2d, c3d], rho=rhoSS, diagonalize=False)
sys2.hbar=1
print("system has been intialized")

states = sys2.diagram_donkey([0, 75, 80, 155], [R1], r=10)

print("finished setup stage")


# 计算重相位图的二维相干响应
time_delays = [100, 10, 100]
scan_id = [0, 2]
response_list = []
states_list = []
diagrams = rephasing+nonrephasing
for k in range(6):
    states, t1, t2, dipole = sys2.coherence2d(time_delays, diagrams[k], scan_id, r=1/2, parallel=True)
    print('diagram ', k, ' done')
    response_list.append(1j*dipole)
    states_list.append(states)
spectra_list, extent, f1, f2 = sys2.spectra(np.imag(response_list))




qsave(response_list, 'dipole_6spin_6cavity_res_half_range_100_10_100_g_050')

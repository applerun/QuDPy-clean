from qutip import *
import numpy as np
from qudpy.Classes import *
import qudpy.plot_functions as pf
import ufss  # 图生成工具

# 在冲激近似下生成三阶重相位与非重相位双边图。

DG = ufss.DiagramGenerator
R3rd = DG()
# 设置重相位图 R1、R2、R3 的条件
R3rd.set_phase_discrimination([(0, 1), (1, 0), (1, 0)])
# 设置脉冲时间窗口
t_pulse = np.array([-1, 1])
R3rd.efield_times = [t_pulse]*4
# 在脉冲到达时刻 0、100、200 以及探测时刻 300 处生成图
[R3, R1, R2] = R3rd.get_diagrams([0, 100, 200, 300])
rephasing = [R1, R2, R3]
print('the rephasing diagrams are R1, R2 and R3 ', rephasing)

# 设置条件并生成非重相位图 R4、R5、R6
R3rd.set_phase_discrimination([(1, 0), (0, 1), (1, 0)])
[R6, R4, R5] = R3rd.get_diagrams([0, 100, 200, 200])
nonrephasing = [R4, R5, R6]
print('the non-rephasing diagrams are R4, R5 and R6', nonrephasing)

# 构造哈密顿量
hbar = 0.658211951  # in eV fs
E1 = 2  # eV
E2 = 2.1  # eV
w1 = E1/hbar  # 频率
w2 = E2/hbar  # 频率
j = -0.3/hbar  # 耦合强度
mu1 = 1.0  # 子系统 A 的偶极强度
mu2 = 1.0  # 子系统 B 的偶极强度
a = tensor(destroy(2), qeye(2))  # 子系统 A 的下降算符
b = tensor(qeye(2), destroy(2))  # 子系统 B 的下降算符
A = mu1*a + mu2*b  # 整个系统的总下降算符
mu = mu1*(a.dag()+a) + mu2*(b.dag()+b)  # 总偶极算符
H = hbar*(w1*a.dag()*a + w2*b.dag()*b)  # 基础哈密顿量
H += hbar*j*(a.dag()*b+b.dag()*a)  # 子系统之间的耦合项

# 通过系统-热浴耦合构造塌缩算符
kappa = 0.1  # 内部弛豫速率
kB = 8.617333262*1e-5  # 玻尔兹曼常数 eV/K
T = 300  # 温度，单位 K
kT = T*kB
beta = 1/kT
n1 = 1/(np.exp(E1*beta)-1)  # <n1>，未耦合态的热占据数
n2 = 1/(np.exp(E2*beta)-1)  # <n2>，未耦合态的热占据数
c1 = np.sqrt(kappa*(n1+1))*a  # 弛豫算符
c2 = np.sqrt(kappa*(n2+1))*b  # 弛豫算符
c3 = np.sqrt(kappa*n1)*a.dag()  # 激发算符
c4 = np.sqrt(kappa*n2)*b.dag()  # 激发算符
c_ops = [c1, c2, c3, c4]  # 塌缩算符列表

# 设置系统
rho = tensor(fock_dm(2, 0), fock_dm(2, 0))  # 哈密顿量对应的基态
sys = System(H=H, rho=rho, a=A, u=mu, c_ops=c_ops, diagonalize=True)

# 先对 R1 做一次单独试算
sys.diagram_donkey([0, 100, 110, 210], [R1])

# 计算二维相干响应
time_delays = [100, 10, 100]
scan_id = [0, 2]
response_list = []
total_diagrams = rephasing+nonrephasing
for k in range(6):
    states, t1, t2, dipole = sys.coherence2d(time_delays, total_diagrams[k], scan_id, r=1, parallel=True)
    response_list.append(1j*dipole)
spectra_list, extent, f1, f2 = sys.spectra(np.imag(response_list), resolution=1)
qsave(spectra_list, 'example1_res1_spectra_list')
qsave(response_list, 'exampel1_res1_response_list')
pf.multiplot(np.imag(spectra_list), extent, ['E emission', 'E absorption'], ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'], 'log',
             color_map='PuOr')

#spectra_list = [x[(len(x) // 2 + len(x) % 2):, :(len(x) // 2 + len(x) % 2)] for x in spectra_list]
#extent = [0, extent[1], extent[2], 0]

rephasing_spectra = spectra_list[:3]
rephasing_spectra.append(np.sum(spectra_list[:3], 0))
pf.silva_plot(rephasing_spectra, scan_range=extent, labels=['E emission', 'E absorption'],
              title_list=['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'], scale='linear', color_map='PuOr',
              interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='4', invert_y=False,
              diagonals=[True, True])

nonrephasing_spectra = spectra_list[3:]
nonrephasing_spectra.append(np.sum(spectra_list[3:], 0))
pf.silva_plot(nonrephasing_spectra, scan_range=extent, labels=['E emission', 'E absorption'],
              title_list=['$R_4$', '$R_5$', '$R_6$', '$R_{nonrephasing}$'], scale='linear', color_map='PuOr',
              interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='1', invert_y=False,
              diagonals=[True, True])

pf.silva_plot(rephasing_spectra, scan_range=extent, labels=['E emission', 'E absorption'],
              title_list=['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'], scale='log', color_map='PuOr',
              interpolation='spline36', center_scale=True, plot_sum=False, plot_quadrant='All', invert_y=True,
              diagonals=[False, True])

pf.silva_plot(nonrephasing_spectra, scan_range=extent, labels=['E emission', 'E absorption'],
              title_list=['$R_4$', '$R_5$', '$R_6$', '$R_{nonrephasing}$'], scale='log', color_map='PuOr',
              interpolation='spline36', center_scale=True, plot_sum=False, plot_quadrant='All', invert_y=False,
              diagonals=[False, True])

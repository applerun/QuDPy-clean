from qutip import *
import numpy as np
import matplotlib.pyplot as plt

class System:
    """
    `System` 类通过模型哈密顿量定义一个物理系统。
    类中的主要成员包括：
    n = 最大激发态编号
    H = 系统哈密顿量
    rho = 系统密度矩阵
    a = 系统下降算符
    u = 偶极算符
    c_ops = 描述系统与热浴耦合的塌缩算符
    e_ops = 需要计算期望值的观测量
    tlist = 时间演化所使用的时间步列表
    """
    # 一些常用物理参数的默认值
    hbar = 0.658211951  # eV fs
    h = 4.13566766  # eV fs
    E = 2  # eV
    f = E/h  # frequency in per fs
    w = E/hbar  # angular frequency 2*np.pi*f (the units are radians per fs.)

    current_states = None

    def __init__(self, n=None, H=None, rho=None, a=None, u=None, c_ops=None, e_ops=None, tlist=None, diagonalize=False):
        self.n = n if n is not None else 3  # 注意：能级数 = 最大激发态编号 + 1 = n+1
        self.rho = rho if rho is not None else fock_dm(self.n+1, 0)  # 初始密度矩阵，默认取基态
        self.a = a if a is not None else destroy(self.n+1)  # 下降算符
        self.u = u if u is not None else self.a.dag()+self.a  # 偶极算符
        self.H = H if H is not None else self.hbar*self.w*self.a.dag()*self.a  # 哈密顿量，默认是谐振子
        self.c_ops = c_ops if c_ops is not None else []  # 塌缩算符列表，默认空
        self.e_ops = e_ops if e_ops is not None else []  # 需要求期望值的算符列表，默认空
        self.tlist = tlist if tlist is not None else []  # 时间步列表，默认空
        self.diagonalized = diagonalize
        if self.diagonalized:
            print("diagonalizing Hamiltonian and transforming everything into eigen-basis except rho")
            evals, evecs = self.H.eigenstates()
            self.a = self.a.transform(evecs)
            self.u = self.u.transform(evecs)
            self.H = self.H.transform(evecs)
            self.c_ops = [c.transform(evecs) for c in self.c_ops]
            self.e_ops = [e.transform(evecs) for e in self.e_ops]
        print("system initialized")

    def diagram_donkey(self, interaction_times=None, diagrams=None, r=10):
        """
        计算并绘制一组双边图对应的单次密度矩阵演化。
        这个函数更适合用来做检查或教学演示。
        :param interaction_times: 脉冲到达时刻列表，最后一个条目对应本振探测前的时间段；
        注意第一束脉冲在 t=0 到达
        :param diagrams: 双边图列表（`ufss` 的 `DiagramGenerator` 输出格式）
        :param r: 时间分辨率（每飞秒多少个时间步）
        :return: None
        """
        if interaction_times is None:
            print("Error: interaction times not given")
        elif diagrams is None:
            print("Error: diagrams not given")
            return None

        # 设置仿真
        total_diagrams, total_interactions = np.shape(diagrams)[:2]
        print('total diagrams', total_diagrams, ', total interactions ', total_interactions)
        for diagram in diagrams:  # 遍历每个图
            rho = self.rho  # 设置初始密度矩阵（通常为基态）
            states = []
            # 依次遍历各个脉冲
            for pulse in range(len(interaction_times)-1):  # 遍历各脉冲
                # 施加当前脉冲对应的所有相互作用
                for x in diagram:  # 遍历单个图中的所有相互作用
                    print(x)
                    if x[1] == pulse:
                        if x[0] == 'Ku':
                            rho = (self.a.dag()*rho)
                        elif x[0] == 'Bu':
                            rho = (rho*self.a)
                        elif x[0] == 'Kd':
                            rho = (self.a*rho)
                        elif x[0] == 'Bd':
                            rho = (rho*self.a.dag())
                delta_t = interaction_times[pulse+1]-interaction_times[pulse]
                results = mesolve(
                    self.H, rho, np.linspace(interaction_times[pulse], interaction_times[pulse+1], delta_t*r),
                    self.c_ops, [])
                rho = results.states[-1]  # 当前演化的末态将作为下一段演化的初态
                states += results.states

            time_list = np.linspace(interaction_times[0], interaction_times[-1], len(states))
            plt.figure()
            plt.plot(time_list, np.imag(expect(self.u, states)))
            plt.plot(time_list, np.abs(expect(self.a.dag() * self.a, states)))
            plt.legend(['dipole', 'number'])
            plt.xlabel('Time (fs)')
            plt.ylabel('Value')
            plt.title('Expectation Values for '+str(diagram))
        plt.show()
        return None

    def coherence2d(self, time_delays=None, diagram=None, scan_id=None, r=10, parallel=False):
        """
        计算单个双边图在两个可扫描时延下的二维相干图。
        如果计算资源允许，可以并行执行。
        :param time_delays: 各相互作用之间的时延列表（即使某个时延为 0 也需要显式给出）
        :param diagram: 单个双边图（`ufss` 的 `DiagramGenerator` 输出格式）
        :param scan_id: 需要扫描的时延索引列表
        :param r: 时间分辨率（每飞秒多少个时间步）
        :param parallel: 是否启用并行计算，True 或 False
        :return: 密度矩阵列表、第一扫描轴时间数组、第二扫描轴时间数组，以及偶极响应
        """

        if len(time_delays) != len(diagram):
            print('time delays for each interaction not given')
            print('number of time delays', len(time_delays), ' number of interactions ', len(diagram))
            return None
        if len(scan_id)!=2:
            print('scan id not provided for two tunable delays')
            return None

        if parallel:
            from qutip import parallel as pp

        rho = self.rho  # 从系统对象中取初始态

        # 顺序处理相互作用和时延；若时延为零则直接进入下一步
        # 这个循环只进行到遇到第一个可扫描时延为止
        for i in range(scan_id[0]):
            rho = self.apply_pulse(rho, diagram[i])  # 施加脉冲相互作用

            # 若该时延非零，则在脉冲作用后做自由演化
            delta_t = time_delays[i]
            if delta_t > 0:
                coherence_time = np.linspace(0, delta_t, int(delta_t*r))
                results = mesolve(self.H, rho, coherence_time, self.c_ops, [])
                rho = results.states[-1]  # 只保留末态

        # 走到这里时，所有不需要扫描的脉冲和时延都已经处理完
        # 接下来施加第一个需要扫描的脉冲和时延，因此要保留整个演化过程中的所有态
        rho = self.apply_pulse(rho, diagram[scan_id[0]])
        delta_t = time_delays[scan_id[0]]
        t_list = np.linspace(0, delta_t, int(delta_t*r))
        results = mesolve(self. H, rho, t_list, self.c_ops, [])
        states = results.states

        # 继续施加后续相互作用，直到遇到下一个可扫描时延
        for i in range(scan_id[0]+1, scan_id[1]):
            states = [self.apply_pulse(state, diagram[i]) for state in states]  # 对所有态施加同一相互作用
            delta_t = time_delays[i]
            if delta_t > 0:
                coherence_time = np.linspace(0, delta_t, int(delta_t*r))
                # 逐个演化状态列表中的态，并仅保留末态
                if parallel:
                    self.tlist = coherence_time
                    states = pp.parallel_map(self.para_mesolve, states, only_last_state=True)
                    #states = [pp.parfor(self.para_mesolve, states, only_last_state=True)]
                    #print('shape of states is ', np.shape(states))
                else:
                    states = [mesolve(self.H, state, coherence_time, self.c_ops, []).states[-1] for state in states]

        # 走到这里时，只剩下最后一个相互作用和最后一个可扫描时延
        print('First scan done, starting second scan. Remaining time = First Scan Time x number of steps in second scan'
              + '/number of processors')
        states = [self.apply_pulse(state, diagram[scan_id[1]]) for state in states]
        delta_t = time_delays[scan_id[1]]
        t_list = np.linspace(0, delta_t, int(delta_t*r))
        final_states = []
        if parallel:
            self.tlist = t_list
            final_states.append(pp.parfor(self.para_mesolve, states, only_last_state=False))
            final_states = final_states[0]
        else:
            final_states = [mesolve(self.H, state, t_list, self.c_ops, []).states for state in states]

        dipole = np.array([expect(self.u, final_states[x][:]) for x in range(len(final_states))])
        #dipole = None
        #plt.figure()
        #plt.imshow(dipole.imag, origin='lower', interpolation='spline36', extent=[0, time_delays[scan_id[0]],
        #                                                                          0, time_delays[scan_id[1]]])
        #plt.show()
        print('second scan done')
        return final_states, np.linspace(0, time_delays[scan_id[0]], int(time_delays[scan_id[0]] * r)), t_list, dipole

    # 一些辅助函数，用来让 coherence2d 更易读
    def apply_pulse(self, rho, x):
        """
        一个在密度矩阵上施加算符的简单辅助函数。
        :param rho: 初始密度矩阵
        :param x: 要施加的算符信息
        :return: 作用后的密度矩阵
        """
        if x[0] == 'Ku':
            rho = (self.a.dag()*rho)#.unit()
        elif x[0] == 'Bu':
            rho = (rho*self.a)#.unit()
        elif x[0] == 'Kd':
            rho = (self.a*rho)#.unit()
        elif x[0] == 'Bd':
            rho = (rho*self.a.dag())#.unit()
        return rho

    def para_mesolve(self, rho, only_last_state=True):
        """
        为 `coherence2d` 提供并行计算支持的简单辅助函数。
        :param rho: 密度矩阵
        :param only_last_state: 为 True 时仅保留末态，否则保留完整状态序列
        :return: 状态列表或单个状态
        """
        if only_last_state:
            return mesolve(self.H, rho, self.tlist, self.c_ops, []).states[-1]
        else:
            return mesolve(self.H, rho, self.tlist, self.c_ops, []).states

    # 一些常用的频谱辅助函数

    def spectra(self, dipoles=None, resolution=10):
        """
        通过傅里叶变换把偶极响应列表转换为频谱。
        :param dipoles:
        :param resolution:
        :return: 频谱列表、两个坐标轴的范围，以及频率网格 f1 和 f2
        """
        if dipoles is None:
            print('Input data missing')
            return

        spectra = [np.fft.fftshift(np.fft.fft2(mu)) for mu in dipoles]
        # 这里乘以 2pi 是因为 fft 使用频率 freq，而 qutip 使用角频率 omega
        freq1 = np.fft.fftshift(np.fft.fftfreq(np.shape(spectra[0])[1], 1 / resolution)) * 2 * np.pi
        freq2 = np.fft.fftshift(np.fft.fftfreq(np.shape(spectra[0])[0], 1 / resolution)) * 2 * np.pi
        extent = [min(freq1), max(freq1), min(freq2), max(freq2)]
        f1, f2 = np.meshgrid(freq1, freq2)

        return spectra, extent, f1, f2

    def linear_spec(self, scan_time: int, diagram=None, resolution=10):
        """
        计算系统在若干初始相互作用之后的简单线性光谱。
        注意：如果想提高频率分辨率，可以增大 `scan_time`；如果想缩小频率范围，可以降低时间分辨率。
        :param scan_time: 需要模拟的时间区间
        :param diagram: 用于计算系统响应的双边图。注意：图中的所有相互作用都在 t=0 时施加。
        如果 `diagram=None`，则默认在 t=0 施加一次 `Bu` 相互作用。
        :param resolution: 仿真的时间分辨率
        :return: 偶极期望值、时间数组、频谱和频率数组
        """
        t_list = np.linspace(0, scan_time, resolution * scan_time)
        if diagram:
            rho = self.rho
            for x in diagram:
                rho = self.apply_pulse(rho, x)  # 逐项施加脉冲相互作用
        else:
            rho = self.rho * self.a
            # 这里通过在初态 rho 上施加一次 'Bu' 作用来生成线性响应
        dipole = mesolve(self.H, rho, t_list, self.c_ops, [self.u]).expect[0]

        plt.figure(figsize=(16, 6))
        plt.plot(t_list, np.imag(dipole))
        plt.plot(t_list, np.real(dipole))
        plt.legend(['Imaginary', 'Real'])
        plt.xlabel('Time (fs)')
        plt.ylabel('Dipole')
        plt.title('Expectation Values for linear response')
        plt.show()

        spec = np.fft.fftshift(np.fft.fft(dipole))

        freq = np.fft.fftshift(np.fft.fftfreq(np.shape(spec)[0], 1 / resolution)) * 2 * np.pi
        plt.figure(figsize=(16, 6))
        plt.plot(freq, np.imag(spec))
        plt.plot(freq, np.real(spec))
        plt.legend(['Imaginary', 'Real'])
        plt.xlabel('Freq (eV)')
        plt.ylabel('Dipole')
        plt.title('Spectrum from linear response (Bu)')
        plt.show()
        # print(spec[len(freq)//2:])
        # print(freq[len(freq)//2:])
        return dipole, t_list, spec, freq

    def pop_study(self, pop_time_list=None, pop_index=1, time_delays=None, diagram=None, scan_id=None, r=10, parallel=False):
        """
        对一组不同的 population time 计算某个双边图的非线性响应。
        :param pop_time_list: population time 列表
        :param pop_index: 产生 population 的那次相互作用在时延列表中的索引
        :param time_delays: 各相互作用之间的时延列表
        :param diagram: 需要模拟的双边图
        :param scan_id: 用于二维相干图扫描的时延索引
        :param r: 仿真的时间分辨率
        :param parallel: 是否启用并行计算
        :return: 每个 population time 对应的二维相干响应、t1 和 t2 时间轴、
        每个 population time 对应的频谱列表、频谱坐标范围以及频率网格 f1 和 f2
        """
        if len(time_delays) != len(diagram):
            print('time delays for each interaction not given')
            print('number of time delays', len(time_delays), ' number of interactions ', len(diagram))
            return None
        if len(scan_id) != 2:
            print('scan id not provided for two tunable delays')
            return None
        if len(pop_time_list) < 2:
            print('Less than two population times requested')

        pop_response = []
        for t_pop in pop_time_list:
            time_delays[pop_index] = t_pop
            states, t1, t2, dipole = self.coherence2d(time_delays, diagram, scan_id, r, parallel)
            pop_response.append(dipole)  # both real and imaginary parts are contained in it.

        spectra_list, extent, f1, f2 = self.spectra(np.imag(pop_response))

        return pop_response, t1, t2, spectra_list, extent, f1, f2

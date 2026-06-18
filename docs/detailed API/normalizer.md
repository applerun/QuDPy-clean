| 物理量                  |                    用户输入单位 |                                          中间物理求解单位 |                      内部代码单位 | 转换关系                                                  |
| -------------------- | ------------------------: | ------------------------------------------------: | --------------------------: | ----------------------------------------------------- |
| 时间                   |                        fs |                                                fs |               dimensionless | $t_{\mathrm{code}}=t_{\mathrm{fs}}/T_0$               |
| 时间步长                 |                        fs |                                                fs |               dimensionless | $dt_{\mathrm{code}}=dt_{\mathrm{fs}}/T_0$             |
| 能级差                  |                        eV |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\omega_{eg}=E_{eg}/\hbar$，再乘 $T_0$                   |
| 激光能量                 |                        eV |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\omega_L=E_L/\hbar$，再乘 $T_0$                         |
| 失谐量                  |               eV 或由两频率差得到 |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\Delta=\omega_{eg}-\omega_L$，再乘 $T_0$                |
| 偶极矩                  |                     Debye |                              参与计算 $\mu E_0/\hbar$ |                   通常并入 Rabi | 与电场合成 $\Omega$                                        |
| 电场强度                 |                     MV/cm |                              参与计算 $\mu E_0/\hbar$ | 通常并入 Rabi / field amplitude | 与偶极矩合成 $\Omega$                                       |
| Rabi frequency       |      由 $\mu E_0/\hbar$ 得到 |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\Omega_{\mathrm{code}}=\Omega_{\mathrm{fs}^{-1}}T_0$ |
| $T_1$                |                        fs |       $\gamma_1=\frac{1}{T_1}$，$\mathrm{fs}^{-1}$ |               dimensionless | $\gamma_{1,\mathrm{code}}=\gamma_1T_0$                |
| $T_\phi$             |                        fs | $\gamma_\phi=\frac{1}{T_\phi}$，$\mathrm{fs}^{-1}$ |               dimensionless | $\gamma_{\phi,\mathrm{code}}=\gamma_\phi T_0$         |
| $T_2$                |                        fs |       $\gamma_2=\frac{1}{T_2}$，$\mathrm{fs}^{-1}$ |               dimensionless | $\gamma_{2,\mathrm{code}}=\gamma_2T_0$                |
| 密度矩阵                 |                       无量纲 |                                               无量纲 |                         无量纲 | 不反变换                                                  |
| population           |                       无量纲 |                                               无量纲 |                         无量纲 | 不反变换                                                  |
| coherence            |                       无量纲 |                                               无量纲 |                         无量纲 | 不反变换                                                  |
| collapse operator 系数 | $\sqrt{\mathrm{fs}^{-1}}$ |                         $\sqrt{\mathrm{fs}^{-1}}$ |   $\sqrt{\text{code rate}}$ | 系数是 $\sqrt{\gamma_{\mathrm{code}}}$                   |

# DROPPS MDP 参数说明

本文档说明 `dps grompp` 当前支持的 MDP 参数、实际生效阶段和输出文件。DROPPS 的 MDP 语法借鉴 GROMACS，但参数集合和语义并不完全相同。

## 1. 工作流规则

MDP 只在生成 TPR 时读取：

```bash
dps grompp -f system.pdb -p system.top -m md.mdp -o run.tpr
dps mdrun -s run.tpr -o run -cpt 5
```

`mdrun` 读取的是 `run.tpr` 中保存的参数，不会再次读取 `md.mdp`。因此修改 MDP 后必须重新执行 `grompp`；仅重新执行 `mdrun` 不会反映修改。

`nsteps` 只表示 production 阶段的总步数。能量最小化和温度渐变不计入 `nsteps`；进入 production 时，步数和时间会从 0 开始记录。production 续跑会跳过最小化和温度渐变；若 checkpoint 位于温度渐变阶段，则从清单记录的温度段和段内步数继续。`nsteps` 始终是 production 目标总步数，而不是追加步数。

`mdrun` 的运行时选项优先于 TPR 中相应的值。例如 `-nsteps` 和
`--seed` 可以临时覆盖 production 总步数和随机种子，但不会修改 TPR。
平台、设备、输出路径、墙钟 checkpoint 和 `-maxh` 也都属于运行时选项；
使用 `dps mdrun -h` 查看完整列表。

## 2. 文件格式与单位

- 每行使用 `参数 = 值`；行尾可以用 `#` 添加注释。
- 参数名中的 `-` 和 `_` 等价，例如 `nst-energy` 与 `nst_energy`。
- 布尔值区分大小写，只能写成 `True` 或 `False`。
- 未写出的参数使用程序内置缺省值。
- 时间单位为 ps，长度为 nm，温度为 K，压强耦合单位为 bar，能量为 kJ/mol。
- 生效的 `nst-*` 输出间隔都以 production 步数计；对应参数为 `0` 时禁用该输出。checkpoint 例外：现在由 `mdrun -cpt` 按墙钟分钟控制。
- MDP 不是 GROMACS MDP 的完整实现；未知参数会导致 `grompp` 停止。

## 3. 积分、模拟长度与预处理

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `integrator` | `Langevin` | `Langevin` 使用 OpenMM `LangevinMiddleIntegrator`。`steep` 为独立能量最小化模式，调用 OpenMM 最小化器并输出最终 PDB/restart，不运行动力学、不生成 XTC；它不代表另行实现的最速下降算法。动力学开始前的最小化仍由 `minimize` 控制。 |
| `dt` | `0.01` | Langevin 积分步长，单位 ps。 |
| `nsteps` | `100000` | production 的目标总步数。续跑时不是“额外增加”的步数；可由 `mdrun -nsteps` 临时覆盖。 |
| `friction` | `1.0` | Langevin 摩擦系数，单位 ps^-1。 |
| `minimize` | `True` | 新运行开始前是否进行定体积能量最小化；续跑始终跳过。 |
| `forcetol` | `100.0` | 最小化力容差，单位 kJ mol^-1 nm^-1。 |
| `max-step` | `100000` | 最小化最大迭代次数。 |

## 4. 温度、初速度与升温

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `production-temperature` | `300.0` | production 目标温度。温度渐变期间，Langevin 恒温器、barostat、温度依赖 AH 参数、介电常数及 Debye 长度会同步使用当前阶段温度。 |
| `gen-vel` | `True` | 新运行时按 `initial-temperature` 生成 Maxwell-Boltzmann 初速度；续跑从 restart 恢复速度。TPR 本身不保存速度，因此新运行设为 `False` 会直接报错。 |
| `seed` | `1215` | Langevin 积分器、初速度和 Monte Carlo barostat 的随机种子；`0` 让 OpenMM 自动生成。可由 `mdrun --seed` 临时覆盖，续跑时 checkpoint 中的随机状态优先。 |
| `initial-temperature` | `150.0` | 初速度温度，也是升温起点。 |
| `warming-speed` | `1.0` | 初温与 production 温度不同时的升温或降温速率，单位 K/ns，必须大于 0。按不超过 1 K 的等温差阶段离散，步数取最接近的正整数。 |
| `keep-warming-trajectory` | `False` | 是否把温度渐变坐标单独写入 `<prefix>.warming.xtc`。每完成一个不超过 1 K 的阶段写一帧；渐变续跑会继续该文件。 |

若不需要独立温度渐变，将 `initial-temperature` 设置为与 `production-temperature` 相同。渐变阶段会产生额外计算量，但不会写入 production 的 XTC、log 或 EDR。其 checkpoint 同样受 `-cpt`、`-maxh` 和 Ctrl-C 保护；完成后 production 输出仍从 step 0、time 0 开始。

## 5. 键、约束与质心运动

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `bondtype` | `bond` | `bond` 使用谐和键；`constraint` 使用距离约束。运行时压力张量和构型压力重算不支持约束，因为当前不能得到约束 virial。 |
| `comm-mode` | `Linear` | `Linear` 定期去除质心平动；`none` 禁用。其他拼写会在 `grompp`/`mdrun` 校验时报错。 |
| `nstcomm` | `100` | `comm-mode = Linear` 时去除质心平动的间隔，单位步。 |

## 6. 非键相互作用

### 6.1 van der Waals

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `vdwtype` | `pLJ` | `pLJ` 对应 Ashbaugh-Hatch 拓扑；`MPiPi` 对应 Wang-Frenkel 拓扑。必须与拓扑的力场函数类型一致。目前不能用 `none` 正常关闭该项。 |
| `cutoff-scheme-lj` | `static` | `static` 使用固定截断；`dynamic` 对每一对粒子使用 `cutoff-lj-multi * sigma_pair`。 |
| `cutoff-lj` | `1.5` | `static` 模式的 LJ 截断，单位 nm。 |
| `cutoff-lj-multi` | `3.0` | `dynamic` 模式的无量纲截断倍数，必须大于 0。MPiPi/Wang-Frenkel 强制要求 `dynamic` 且值为 `3.0`，使势能终止于 `R=3 sigma`。 |
| `shift-lj` | `True` | 是否把 LJ 势平移到配对截断处为 0。 |

### 6.2 静电

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `coulombtype` | `yukawa` | `yukawa` 使用 Debye-Huckel 相互作用并要求匹配的拓扑；`no` 关闭静电。 |
| `cutoff-coul` | `1.5` | 静电截断，单位 nm。 |
| `shift-coul` | `True` | 是否把静电势平移到截断处为 0。 |
| `salt-conc` | `0.1` | 盐浓度，单位 mol/L。Debye 长度根据当前 Context 温度及拓扑介电模型动态计算；使用 `yukawa` 时必须为正值。 |

拓扑文件可以通过 `[ simulation-setting ]` 对 MDP 参数提出 NVT/NPT 的强制值或推荐值。强制值不匹配时 `grompp` 会停止，推荐值不匹配时会给出警告。因此更换力场后，应优先检查 `grompp` 的诊断信息，而不是机械沿用旧 MDP。

## 7. 系综与压强耦合

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `pcoulp` | `True` | `True` 添加 OpenMM Monte Carlo barostat，运行 NPT；`False` 为固定盒子的 NVT。 |
| `ref-P` | `1.0` | barostat 目标压强，单位 bar。 |
| `tau-P` | `5.0` | Monte Carlo 体积变化的尝试间隔，单位“步”。该名称沿用旧模板，但不是以 ps 表示的松弛时间。 |

Green-Kubo 黏度、平面表面张力等分析通常要求固定体积 production，因此应使用 `pcoulp = False`。NPT 平衡完成后，可重新 `grompp` 生成 NVT TPR 做 production。

## 8. 输出控制

checkpoint 不再使用 MDP 步数间隔。`mdrun -cpt N` 每 `N` 个墙钟分钟保存
一次，缺省为 5 分钟；每次同时生成原生 `.chk`、可移植 `.state.xml` 和校验
清单 `.restart.json`。温度渐变和 production 都会执行该机制；`-cpt 0` 关闭周期保存，但正常结束、`-maxh` 停止和
Ctrl-C 仍会保存 restart 文件。`-cpnum` 会额外保留带当前 Context step
编号的一组 restart 文件。原生 checkpoint 可精确续跑但要求 OpenMM 版本、
平台和硬件兼容；跨版本时自动回退 State，坐标、速度、盒子、时间和步数会
保留，但积分器内部随机数状态不会保留，因此轨迹不会逐位相同。

| 参数 | 内置缺省值 | 输出 | 说明 |
|---|---:|---|---|
| `keep-warming-trajectory` | `False` | `<prefix>.warming.xtc` | 温度发生变化时生成；每个不超过 1 K 的阶段一帧，可随温度渐变 checkpoint 续写。 |
| `nst-xout` | `1000` | `<prefix>.xtc` | 坐标轨迹间隔。新运行还会写 production step 0。设为 0 禁用。 |
| `nst-screenlog` | `1000` | 终端 | 固定进度字段的终端输出间隔。发布模板为减少终端输出，将其设为 `100000`。 |
| `nst-filelog` | `10000` | `<prefix>.log` | 固定 CSV 进度日志的间隔。字段固定为 step、time、progress、speed、elapsedTime、remainingTime。 |
| `nst-energy` | `10000` | `<prefix>.edr` | 热力学数据间隔。DROPPS EDR 是带版本和单位元数据的 CSV，不是 GROMACS 二进制 EDR。 |
| `energy-grps` | 见下文 | `<prefix>.edr` | 逗号分隔、大小写不敏感、自动去重。`step` 和 `time_ps` 总会写入，不需要选择。 |

可选的普通 EDR 项为：

| 项 | 单位 | 含义 |
|---|---|---|
| `potentialEnergy` | kJ/mol | 势能 |
| `kineticEnergy` | kJ/mol | 用当前力把 leapfrog 半步速度同步到当前坐标时刻后计算的动能 |
| `totalEnergy` | kJ/mol | 同一坐标时刻的势能与同步动能之和 |
| `temperature` | K | 由同步瞬时动能和自由度计算的温度 |
| `boxX`, `boxY`, `boxZ` | nm | 三个周期盒矢量的长度 |
| `volume` | nm^3 | 周期盒体积 |
| `density` | g/mL | 系统总质量密度 |

下列项会启动高成本的有限应变压力计算：

| 项 | 单位 | 含义 |
|---|---|---|
| `pressure` | kJ mol^-1 nm^-3 | 压力张量迹的三分之一 |
| `Pxx`, `Pyy`, `Pzz` | kJ mol^-1 nm^-3 | 压力张量对角分量 |
| `Pxy`, `Pxz`, `Pyz` | kJ mol^-1 nm^-3 | 压力张量非对角分量 |

压力项包含动理学项和构型 virial；它们要求正交盒、`bondtype = bond`，且每次输出要额外计算 12 次势能。若只需要盒子和温度，不要把任何压力项加入 `energy-grps`。

从 EDR 选择数据并导出 XVG：

```bash
# 先列出文件内已有的项
dps energy -f run.edr --list

# 不提供 --terms 时进入编号/名称交互选择
dps energy -f run.edr -o selected.xvg

# 非交互式选择
dps energy -f run.edr -o thermo.xvg --terms temperature volume -b 1000 -e 5000
```

## 9. 独立压力张量输出

| 参数 | 内置缺省值 | 作用与限制 |
|---|---:|---|
| `nst-stress` | `0` | 大于 0 时按该间隔写 `<prefix>.stress.xvg`；0 禁用。 |
| `stress-strain` | `0.0003` | 构型压力中心有限差分的应变量，必须满足 `0 < value < 0.02`。EDR 压力项也使用它。 |
| `stress-output` | `auto` | `auto` 写 `<mdrun prefix>.stress.xvg`，也可以指定路径。 |
| `stress-platform` | `auto` | 压力辅助 Context 的 OpenMM 平台。`auto` 依次尝试 CUDA、OpenCL、CPU、Reference。 |
| `stress-precision` | `double` | 辅助 Context 的计算精度；平台不支持该属性时会忽略。 |
| `stress-device` | `auto` | GPU 设备索引；`auto` 不指定。 |
| `stress-threads` | `0` | CPU 线程数；不大于 0 时使用 OpenMM 缺省值。 |

`nst-stress > 0` 和 EDR 压力项使用同一套压力算法，但属于两个独立 reporter；同时启用会重复计算，程序会发出警告。通常只选其中一种：需要统一 EDR 工作流时选择 EDR 压力项，需要直接给 `dps viscosity-gk` 使用的压力 XVG 时选择 `nst-stress`。

XTC 不保存速度，所以 `dps rerun` 只能从轨迹重算势能、盒子量和 `configPressure/configP*` 构型压力；它不能恢复动能、瞬时温度、完整压力张量或 Green-Kubo 所需的动理学贡献。

## 10. 已保留但尚未生效的参数

以下参数仍可被解析以兼容旧 MDP，但当前不会改变模拟行为；显式写入时会发出弃用警告。它们不出现在新模板的活动配置中。

| 参数 | 内置缺省值 | 当前状态 |
|---|---:|---|
| `tcoulp` | `Bussi` | 尚未用于选择恒温算法；当前始终由 `integrator = Langevin` 决定。 |
| `nstlist` | `10` | 尚未用于 OpenMM 邻居列表更新设置。 |
| `buffer` | `0.5` | 尚未用于截断或邻居列表配置。 |
| `nst-cp` | `0` | 仅为旧 MDP/TPR 保留；运行时忽略。使用 `mdrun -cpt` 配置墙钟 checkpoint。 |

旧参数 `screenlog-grps` 已弃用并被忽略，因为 log 字段现已固定。旧参数 `filelog-grps` 会在未设置 `energy-grps` 时作为兼容别名；旧 `nst-filelog` 会在未设置 `nst-energy` 时同时充当 EDR 间隔。新 MDP 应明确写出 `nst-energy` 和 `energy-grps`。

## 11. 常用配置片段

### NVT production

```ini
pcoulp = False
production-temperature = 300
initial-temperature = 300
gen-vel = True
```

### NPT 平衡

```ini
pcoulp = True
ref-P = 1
tau-P = 5
```

### Green-Kubo 压力张量

```ini
pcoulp = False
bondtype = bond
nst-stress = 10
stress-strain = 0.0003
```

### 低成本 EDR

```ini
nst-energy = 10000
energy-grps = potentialEnergy,kineticEnergy,totalEnergy,temperature,boxX,boxY,boxZ,volume,density
```

开始长模拟前，建议先用较小的 `nsteps` 运行短测试，核对 `grompp` 警告、实际生成的输出文件、EDR 项和性能，再提交 production 作业。

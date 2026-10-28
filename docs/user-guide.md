# DROPPS 用户指南

本文档面向使用 DROPPS 进行粗粒化生物分子建模、OpenMM 分子动力学模拟和轨迹分析的用户，内容对应 DROPPS 1.0.0。

## 1. DROPPS 能做什么

DROPPS（Distributed Rapid Operation Platform for Phase-separation Simulations）是一个命令行工具集，主要覆盖以下工作：

- 从蛋白质序列生成粗粒化构象和分子拓扑；
- 将多个分子装入周期性模拟盒；
- 根据 PDB、TOP/ITP 和 MDP 文件生成可运行的 TPR；
- 在 OpenMM 上运行、限时停止和续跑模拟；
- 转换轨迹并分析密度、接触、相共存、分子交换、MSD、表面张力和黏度等性质。

所有功能均通过 `dps` 入口调用：

```bash
dps COMMAND [OPTIONS]
```

## 2. 安装与验证

### 2.1 环境要求

- Python 3.9 或更高版本；
- Python 3.10+ 推荐使用 NumPy 2.x、OpenMM >=8.5,<9 与 MDAnalysis 2.9+；
- Python 3.9 使用兼容组合 NumPy 1.26.x、OpenMM 8.1.x 与 MDAnalysis 2.7.x；
- 如需 NVIDIA GPU 计算，系统必须已安装兼容的 NVIDIA 驱动；CUDA 运行库和
  OpenMM CUDA 平台可由 DROPPS 安装器自动安装。

建议使用独立虚拟环境。在源码目录中执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python scripts/install_dropps.py --editable
```

安装器通过 `nvidia-smi` 检测当前可见的 NVIDIA GPU，并根据驱动支持的最高
CUDA 版本安装 `cuda12` 或 `cuda13` extra。安装完成后，它会实际创建一个
OpenMM CUDA Context；创建失败则整个安装返回失败。如果当前环境必须提供
CUDA（例如 GPU 计算节点），使用：

```bash
python scripts/install_dropps.py --editable --require-cuda
```

也可以跳过自动检测，手动选择依赖组合：

```bash
python -m pip install -e '.[cuda12]'
# 或
python -m pip install -e '.[cuda13]'
```

没有可见 NVIDIA GPU 时，自动安装器安装基础 CPU/OpenCL 组合。`pip` 无法安装
系统级 NVIDIA 驱动；驱动缺失或版本过旧时，应先由系统管理员升级驱动。

如果必须在 OpenMM 8.1 环境中打开并迁移旧的原生 checkpoint，可使用兼容依赖组合：

```bash
python -m pip install -e '.[openmm81]'
```

不要把 OpenMM 8.1 与 NumPy 2.x 混装：OpenMM 8.1 的 XTC 扩展基于 NumPy 1.x ABI 构建。

安装完成后检查版本和命令列表：

```bash
dps --version
dps help commands
dps mdrun -h
```

如果终端提示 `dps: command not found`，请确认虚拟环境已经激活，并在该环境中重新执行安装命令。

### 2.2 命令行与终端输出约定

公开选项统一使用小写名称。长名称采用连字符分隔，例如 `--run-input`；简称只含
一至两个小写字母。所有 Input 和 Output 参数都有简称，且 Output 简称统一由
`o` 开头。常用约定包括输入结构/轨迹 `-f`、运行输入 `-s`、索引 `-n`、主输出
`-o` 和拓扑输出 `-op`。高频参数也会在不冲突时自动获得易记简称，例如
`-c/--center`、`-se/--selection`（或 `--select`）、`-r/--reference`、
`-dt/--delta-time`、`-tu/--time-unit`、`-ca/--center-axis`、
`-bw/--bin-width` 和 `-ff/--forcefield`。简称在每个命令内独立分配，发生冲突时
保留更常用的含义，以 `dps COMMAND -h` 为准。1.0.0 仍兼容部分旧版小写非标准
写法，但帮助和新文档不再展示；大写选项不属于新 CLI。

所有功能命令都会输出启动摘要、输入、输出、主要参数和完成状态。消息级别为
蓝色 `[INFO]`、绿色 `[OK]`、黄色 `[WARNING]`、红色 `[ERROR]` 和暗青色
`[DEBUG]`。颜色只在交互终端中自动启用，重定向到文件时不会写入 ANSI 控制符；
可用 `NO_COLOR=1` 或 `DROPPS_COLOR=never` 禁用，用
`DROPPS_COLOR=always` 强制启用。

排查问题时可把 `-db/--debug` 放在命令前或命令参数中的任意位置：

```bash
dps --debug check -s run.tpr -f run.xtc
dps check -s run.tpr -f run.xtc -db
```

debug 模式会在标准错误流中增加命令参数、工作目录、进程、Python/平台、关键
依赖版本、解析后参数、运行耗时和异常 traceback。

## 3. 认识常用文件

| 扩展名 | 用途 |
|---|---|
| `.pdb` | 粗粒化坐标、盒尺寸和最终结构 |
| `.itp` | 单个分子类型的粒子、键、角和力场参数 |
| `.top` | 系统拓扑，引用 ITP 并声明各分子数量 |
| `.mdp` | 模拟科学参数，只在 `grompp` 阶段读取 |
| `.tpr` | DROPPS 运行输入，包含系统、拓扑、坐标和参数 |
| `.xtc` | 压缩坐标轨迹 |
| `.edr` | DROPPS 的带元数据 CSV 能量文件，不是 GROMACS 二进制 EDR |
| `.ndx` | 分析所用的原子分组 |
| `.xvg`、`.xpm`、`.csv` | 分析结果 |
| `.chk` | OpenMM 原生 checkpoint，兼容环境中可精确续跑 |
| `.state.xml` | 可移植 OpenMM State，跨环境续跑的回退文件 |
| `.restart.json` | restart 校验和与兼容性信息 |

重要规则：`mdrun` 读取的是 TPR，而不是原始 MDP。修改 MDP、PDB、TOP 或 ITP 后，必须重新运行 `grompp`。

## 4. 快速开始：从序列到分析

下面的示例在一个新的工作目录中完成最小工作流。请把序列、分子数、盒尺寸和模拟参数替换为实际研究配置。

### 4.1 从序列生成单链模型

```bash
dps pdb2dps \
  -s "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW" \
  --forcefield HPS \
  --output-conformation chain.pdb \
  --output-topology chain.itp \
  --output-name PROT
```

不提供 `--forcefield` 时，DROPPS 会交互式列出可用力场。源码内置力场位于 `dropps/share/forcefields/`；当前工作目录也可放置自定义 `.ff` 文件，但不能与内置力场同名。

### 4.2 将多个分子装入模拟盒

```bash
dps genmesh \
  -f chain.pdb \
  -p chain.itp \
  -n 100 \
  -g 1.0 \
  -o system.pdb \
  -op system.top
```

`-f`、`-p` 和 `-n` 可以分别接收多个值，但三者的数量和顺序必须对应。例如：

```bash
dps genmesh \
  -f protein.pdb rna.pdb \
  -p protein.itp rna.itp \
  -n 100 20 \
  -o system.pdb \
  -op system.top
```

### 4.3 准备 MDP

从源码运行时，可复制项目模板：

```bash
cp dropps/share/templates/md.mdp md.mdp
```

至少核对以下参数：

```ini
integrator = Langevin
dt = 0.01
nsteps = 100000
production-temperature = 300
initial-temperature = 300
keep-warming-trajectory = False
pcoulp = False
nst-xout = 1000
nst-energy = 10000
```

MDP 中时间使用 ps、长度使用 nm、温度使用 K、压力使用 bar。布尔值必须写成区分大小写的 `True` 或 `False`。完整参数语义见 [MDP 参数说明](mdp-reference.md)。

### 4.4 生成运行输入

```bash
dps grompp \
  -f system.pdb \
  -p system.top \
  -m md.mdp \
  -o run.tpr
```

`grompp` 会构建 OpenMM System，并把坐标、拓扑、参数和源文件写入便携式 TPR v2。请先处理命令输出中的错误和力场建议，再开始长时间模拟。

### 4.5 先做短模拟

```bash
dps mdrun \
  -s run.tpr \
  -o run-smoke \
  --nsteps 1000 \
  --platform CPU
```

短模拟用于检查系统稳定性、输出间隔和运行性能。`--nsteps` 只覆盖本次运行的 production 目标，不会修改 TPR。

### 4.6 运行正式模拟

GPU 示例：

```bash
dps mdrun \
  -s run.tpr \
  --output-prefix run \
  --platform CUDA \
  --device-index 0 \
  --precision mixed \
  --checkpoint-minutes 10 \
  --max-hours 23.5
```

CPU 示例：

```bash
dps mdrun -s run.tpr -o run --platform CPU --threads 8
```

不指定平台时，`auto` 按 CUDA、OpenCL、CPU、Reference 的顺序尝试。GPU 选项不能与 `--threads` 同时使用。

默认输出如下：

| 路径 | 内容 |
|---|---|
| `run.log` | 固定字段的 CSV 进度日志 |
| `run.edr` | 热力学数据 |
| `run.xtc` | production 坐标轨迹 |
| `run.warming.xtc` | 仅在 `keep-warming-trajectory = True` 且发生升温时生成；每 1 K 一帧 |
| `run.chk` | 最新原生 checkpoint |
| `run.state.xml` | 最新可移植状态 |
| `run.restart.json` | restart 元数据和校验信息 |
| `run.pdb` | 最终结构 |
| `run.stress.xvg` | 仅在启用压力张量输出时产生 |

可用 `-x`、`-e`、`-g`、`--checkpoint-output`、`-c` 和
`--stress-output` 分别覆盖这些路径。

### 4.7 检查与分析

先确认 TPR、轨迹和索引能被正确读取：

```bash
dps check -s run.tpr -f run.xtc
```

随后运行分析，例如：

```bash
# 一维密度分布；运行时按提示选择分组
dps density -s run.tpr -f run.xtc -o density.xvg

# 二维 x-y 数密度图；沿 z 投影
dps densmap -s run.tpr -f run.xtc -o densmap.dat \
  --selection 0 --average-axis z --type number

# 列出 EDR 中可用的数据项
dps energy -f run.edr --list

# 非交互式导出温度和体积
dps energy -f run.edr -o thermo.xvg \
  --terms temperature volume
```

## 5. 模拟中断与续跑

### 5.1 自动 restart 文件

`mdrun` 默认每 5 个墙钟分钟保存一次 restart 文件。使用
`--checkpoint-minutes N` 调整间隔，使用 `--checkpoint-minutes 0` 关闭周期保存。
即使关闭周期保存，正常结束、`--max-hours` 停止或 Ctrl-C 中断时仍会保存最终状态。

保留每次保存的带步数副本：

```bash
dps mdrun -s run.tpr -o run --checkpoint-minutes 10 --number-checkpoints
```

### 5.2 从默认位置续跑

```bash
dps mdrun -s run.tpr -o run --checkpoint
```

DROPPS 先尝试 `run.chk`。如果 OpenMM 因版本、平台或硬件变化拒绝原生 checkpoint，则验证并回退到配对的 `run.state.xml`。

- 原生 checkpoint 可精确延续积分器内部状态；
- State 保留坐标、速度、盒子、时间、步数和 Context 参数，但不保留积分器内部随机数状态，因此统计上连续但不保证逐位一致。

也可指定文件：

```bash
dps mdrun -s run.tpr -o run --checkpoint /path/to/run.chk
dps mdrun -s run.tpr -o run --checkpoint /path/to/run.state.xml
```

续跑时，`nsteps` 始终表示 production 的目标总步数，而不是要追加的步数。默认 append 模式会先把比 checkpoint 更新的 XTC、log、EDR 和压力输出回滚到 checkpoint 步，再继续写入，以避免重复或乱序样本。

若希望保留原输出并写入新分段：

```bash
dps mdrun -s run.tpr -o run --checkpoint --no-append
```

新文件使用 `.partNNNN` 后缀，checkpoint 的主路径保持不变。

## 6. TPR 兼容性与旧文件迁移

新建 TPR 使用 v2 格式：它是带校验和的 ZIP 容器，OpenMM System 保存为 XML，其余数据保存为 JSON 或 NumPy 数组。TPR v2 适合在较新的 OpenMM 8.x 环境中读取旧环境生成的输入，但较新 OpenMM 生成的文件不保证能被更旧版本读取。

受信任的旧式 pickle TPR 可在原环境中转换：

```bash
dps convert-tpr -s legacy.tpr -o portable.tpr
```

pickle 在读取时可能执行代码，因此只转换来源可信的文件。比 `Atomtype_AH_DH`/`Atomtype_WF_DH` 更早的 TPR 不受支持，应从原始 PDB、TOP/ITP 和 MDP 重新生成。

## 7. 分组与索引

很多分析命令会提示选择原子组。DROPPS 的原子索引从 **0** 开始；默认至少包含 `System` 和各分子类型分组。

创建或扩展索引文件：

```bash
dps make_ndx -s run.tpr -o index.ndx
```

常见选择表达式：

| 表达式 | 含义 |
|---|---|
| `group 0` | 选择第 0 个已有分组 |
| `group System` | 按名称选择分组 |
| `mol PROT` | 按分子名称选择 |
| `res ALA,GLY` | 按残基名称选择 |
| `resid 1-20` | 按残基编号范围选择 |
| `chainid 0-9` | 按链编号范围选择 |
| `index 0-99` | 按原子编号范围选择 |
| `(mol PROT) & (resid 1-20)` | 取交集 |
| `(mol PROT) | (mol RNA)` | 取并集 |

### 7.1 轨迹转换与单帧导出

`trjconv` 统一承担轨迹选组、时间抽取、PBC 处理、居中、拟合以及
PDB/XTC 转换。所有帧选择都使用物理时间，不接受帧号或按固定帧数跳过。
`--time-unit` 默认为 `ns`，也可设为 `fs`、`ps`、`us`、`ms` 或 `s`。

提取最接近 100 ns 的单帧：

```bash
dps trjconv \
  -s run.tpr \
  -f run.xtc \
  -o frame.pdb \
  -b 100 -e 100 \
  --select "group 0"
```

当 `-b` 和 `-e` 相等时，DROPPS 只写距离该时间最近的已保存帧；若两帧
距离相同则选择较早者，并在终端打印实际帧号和时间。范围转换使用
`-b`、`-e` 和 `--delta-time`：

```bash
dps trjconv -s run.tpr -f run.xtc -o reduced.xtc \
  -b 100 -e 500 --delta-time 10 --time-unit ns --select "mol PROT"
```

PBC 模式含义与 GROMACS `trjconv` 对齐：

| 模式 | 含义 |
|---|---|
| `none` | 不改变周期镜像 |
| `whole` | 按拓扑键重建断裂分子，但不把它们放回主盒 |
| `atom` | 每个原子独立放回主盒 |
| `res` | 平移每个残基，使其质量中心位于主盒内 |
| `mol` | 先重建完整分子，再使每个分子的质量中心位于主盒内 |
| `nojump` | 顺序读取轨迹并消除跨帧周期跳跃 |

普通质量中心居中：

```bash
dps trjconv -s run.tpr -f run.xtc -o centered.xtc \
  -se "group 0" -p mol \
  -c mass -cs "mol PROT" -ca xyz
```

沿 `z` 轴逐帧定位并居中致密 slab：

```bash
dps trjconv -s slab.tpr -f slab.xtc -o centered.xtc \
  -se "group 0" -p mol \
  -c dense -cs "mol PROT" -ca z \
  --dense-phase-threshold 0.5 --density-bin-width 0.05
```

拟合模式为 `translation`、`transxy`、`rot+trans`、
`rotxy+transxy` 或 `progressive`。默认以 TPR 坐标为参考；
`--fit-reference first` 改用第一个处理帧。处理顺序固定为：重建分子或
去跳跃、居中、放回盒内、拟合、最终平移。若居中与拟合同时使用，拟合
可能把选择组移离盒中心，命令会打印警告。

PDB 范围输出默认写成多模型文件；`--separate` 可拆为编号 PDB，
`--conect` 可写出选择组内部的拓扑键。

### 7.2 拼接 XTC 与 EDR 分段文件

使用 `--no-append` 续跑时会生成 `partNNNN` 分段。`trjcat` 按首帧时间排序
输入 XTC，并把所有帧按时间写入一个新文件：

```bash
dps trjcat \
  -f run.xtc run.part0002.xtc run.part0003.xtc \
  -o complete.xtc
```

若多个分段含有相同时间，默认只保留后一分段中的帧；`--keep-duplicates` 可
显式保留重复时间。所有输入必须具有相同原子数，输出会保留每帧原有的
step、time、盒向量和 XTC 精度。`-b`、`-e` 和 `--delta-time` 可在拼接后按
物理时间筛选，`--time-unit` 默认为 `ps`。

`eneconv` 对 DROPPS CSV-EDR 执行相同的分段合并：

```bash
dps eneconv \
  -f run.edr run.part0002.edr run.part0003.edr \
  -o complete.edr
```

输入 EDR 必须具有相同的字段顺序和单位。目标温度、压强耦合、速度可用性和
压强估算方法等关键科学元数据也必须相容；仅在确认混合具有科学意义时才使用
`--allow-incompatible`。相同 step 和 time 的样本由后一个输入文件覆盖。

### 7.3 导出模拟方法报告

`report-methods` 从 TPR 中读取体系组成、积分器、系综、温度、截断、输出
间隔、软件版本、run/system ID 和嵌入的源文件名。无输出参数时打印到终端，
也可同时写普通文本和 LaTeX：

```bash
dps report-methods -s run.tpr
dps report-methods -s run.tpr -o methods.out -m methods.tex
```

报告末尾包含一段可继续人工编辑的英文 Methods 摘要；正式用于论文前仍应
结合实际运行时覆盖参数和完整研究设计进行核对。

## 8. 命令速查

所有 **43 个公开命令**的完整参数、默认值、输入输出、单位和示例见[完整命令参考](command-reference.md)。下表用于快速导航；逐命令页面由实际解析器生成，并由测试检查同步。

详细参数始终以当前安装版本的帮助为准：

```bash
dps help commands
dps help COMMAND
dps COMMAND -h
```

### 8.1 建模与文件处理

| 命令 | 用途 |
|---|---|
| `pdb2dps` | 从蛋白质序列生成粗粒化 PDB 和 ITP |
| `genelastic` | 向 ITP 添加基于距离的弹性网络 |
| `genmesh` | 把一种或多种分子装入三维网格模拟盒 |
| `addangle` | 向 ITP 添加角度约束 |
| `editconf` | 调整 PDB 盒尺寸并可按轴处理 PBC |
| `grompp` | 从 PDB、TOP/ITP 和 MDP 生成 TPR |
| `convert-tpr` | 把可信旧 TPR 转换为便携式 TPR v2 |
| `modifyres` | 同步修改 ITP 和 PDB 中的残基 |
| `trjconv` | 选择、处理并转换 PDB/XTC 轨迹 |
| `gsd2xtc` | 把 GSD 轨迹转换为 XTC |
| `trjcat` | 按时间顺序拼接多段 XTC，并处理重复边界帧 |
| `eneconv` | 拼接、筛选 DROPPS CSV-EDR 能量文件 |
| `report-methods` | 从 TPR 输出模拟方法与来源信息，可写文本或 LaTeX |

### 8.2 模拟

| 命令 | 用途 |
|---|---|
| `mdrun` | 新建或续跑 OpenMM 分子动力学模拟 |
| `rerun` | 从 XTC 重新计算构型能量、盒信息和构型压力 |

XTC 不保存速度，因此 `rerun` 不能恢复动能、瞬时温度或完整压力张量。

### 8.3 分析

| 命令 | 用途 |
|---|---|
| `make_ndx` | 交互式创建或扩展 NDX 分组 |
| `check` | 检查 TPR、XTC 和 NDX |
| `energy` | 从 DROPPS EDR 选择数据并导出 XVG |
| `density` | 计算一维质量或电荷密度 |
| `densmap` | 计算二维质量、数目或电荷密度图 |
| `coexistence` | 分析平面 slab 的稠相、稀相与界面宽度 |
| `exchange` | 分析跨 slab 界面的分子交换和驻留时间 |
| `phase-msd` | 计算去除相平移后的分相条件 MSD |
| `gyrate` | 计算链回转半径和分布 |
| `polystat` | 面向大量蛋白拷贝计算端到端距离、回转形状和主链统计 |
| `angle` | 计算等价蛋白链的主链角度 |
| `distance` | 统一计算固定、链内、链间、最小和最大距离 |
| `rdf` | 计算原子、残基或分子中心之间的径向分布函数和累计配位数 |
| `contact` | 计算残基层接触图和接触数时间序列 |
| `contact.ng` | 以内存友好的流式聚合计算接触图 |
| `timecorr` | 计算接触相关函数和寿命矩阵 |
| `cstat` | 从已有接触图计算残基层统计量 |
| `msd` | 计算所选原子组的均方位移 |
| `rmsd` | 以可配置的逐分子或整体 fit/输出方式计算构象 RMSD |
| `rmsf` | 计算逐分子 fit 后的原子或残基位置涨落 |
| `sasa` | 计算力场半径和周期边界感知的粗粒化 SASA |
| `assembly` | 分析组装体的形成、尺寸、组成和形状 |
| `pbcontact` | 统计分子直接或经接触网络连到自身周期镜像的比例 |
| `pdb2bond` | 向 PDB 添加 CONECT 记录 |
| `surftension` | 从 TPR 和 XTC 计算平面界面表面张力 |
| `viscosity-gk` | 从运行时压力张量计算 Green-Kubo 零剪切黏度 |

#### `gyrate` 稠相/稀相分辨的单链回转半径

原有的 `-ov`、`-oa` 和 `-oh` 仍按固定 index group 计算。对于平面 slab 相分离体系，使用
`-op/--output-phase` 启用动态分相：

```bash
dps gyrate -s slab.tpr -f slab.xtc -n groups.ndx \
  -op phase-rg.xvg -ref 0 -sel 1 2 \
  --axis z --phases dense dilute \
  --center-mode frame --density-bin-width 0.05 --blocks 5
```

`-ref` 定义用于定位和拟合稠相 slab 的参考组；每个 `-sel` 组会按分子拆分。程序逐帧重建跨
周期边界的分子，分别计算质量加权 Rg 和分子质心，再根据拟合出的相边界把该帧中的分子归为
`dense`、`interface` 或 `dilute`。因此发生相间交换的分子会随时间改变类别，而不会被固定在
初始 index group。需要界面统计时，在 `--phases` 中加入 `interface`。

以 `phase-rg.xvg` 为例，会产生：

- `phase-rg.xvg`：各组、各相逐帧的单链平均 Rg；该帧无成员时写为 `nan`；
- `phase-rg.observations.csv`：逐帧、逐分子的坐标、相态和 Rg；可用
  `--no-phase-observations` 关闭；
- `phase-rg.summary.csv`：各组、各相的均值、标准差、中位数、范围、占有率与连续区块 SEM；
- `phase-rg.histogram.xvg`：使用同一 Rg 网格的分相概率密度；
- `phase-rg.profile.xvg`：参考密度、双 tanh 拟合及稠相/稀相截断位置。
- `phase-rg.groupNN-<组名>.values.csv`：为每个 `-sel` 组单独输出两列原始值
  `dense_rg_nm` 和 `dilute_rg_nm`；两相样本数不同时，较短列末尾留空。

该模式假设正交周期盒中存在单个平面 slab，并要求所选分子部分沿每个周期方向的跨度小于半个
盒长，以保证最小镜像重建唯一。`-bw` 控制 Rg 直方图宽度；`--density-bin-width` 控制参考
密度的空间分箱，两者不要混淆。

#### `densmap` 二维密度图

`densmap` 参考 `gmx densmap` 的平面模式：`-aver z` 沿 z 平均并保留 x-y
网格，`-aver y` 保留 x-z，`-aver x` 保留 y-z。它额外支持
`-tp mass|number|charge`，因此可用于液滴轮廓、非平面畴、界面横向不均匀和
多组分分选。例如同时输出两个组分的质量密度：

```bash
dps densmap -s run.tpr -f run.xtc -o components.dat \
  -sel 1 2 -aver z -tp mass -bin 0.05 \
  --center-mode frame -selfit 1 -oxpm components.xpm
```

`-unit volume`（兼容写法 `nm-3`）按每帧真实三维网格体积归一化；`area`
（`nm-2`）只按投影面积归一化；`count` 输出每格平均权重。NPT 盒涨落会逐帧
归一化，再在周期分数坐标中平均。`-xmin/-xmax` 可只统计平均方向上的一个物理
切片，体密度按该切片的实际厚度归一化。

输出 `.dat` 的首行是第二网格轴的 cell-center 坐标，首列是第一网格轴坐标，
其余为密度矩阵；`-oxpm` 可另外生成快速预览。多个 `-sel` 会生成带序号和组名
的独立文件。默认 `--center-mode none` 保留原轨迹位置；对会整体漂移或跨越周期
边界的紧致液滴，使用 `frame`。固定的非平面畴或多个对称畴通常不要强制居中。

### 8.4 多链体系的统一距离分析

`distance` 不要求为每条链创建 NDX 组。用户只选择一次分子类型和两个区域，
程序再按 TPR 中的 chain ID 自动拆分：

```bash
dps distance -s run.tpr -f run.xtc \
  -ref "(mol PROT) & (resid 10)" \
  -sel "(mol PROT) & (resid 50)" \
  --scope intra-chain --position cog --mode pair \
  -ostat intra.csv --statistics-scope both \
  -oh intra-hist.xvg
```

`--scope intra-chain` 只连接同一条链上的两个实体；若有 100 条完整链，则生成
100 个通道。改用 `inter-chain` 后只连接不同链，生成有向的
`A(chain i)-B(chain j), i != j`，即 9900 个通道。缺少目标残基的链会被报告并
跳过，不会因列表位置发生错配。

`--statistics-scope pooled` 将所有链对和帧汇总为一行，`channel` 为每个链对
分别做时间统计，`both` 同时输出两者。`-oav` 输出每帧跨链通道的均值和总体
标准差；仅在需要原始通道时间序列时使用 `-oall` 或 `-oxyz`。总体统计和直方图
采用分块流式累计，不保存 `帧数 × 链对数` 的完整矩阵。

端点可选择 `atom`、`cog` 或 `com`。`--mode pair` 固定配对位置，`min/max`
在每个允许的实体对内部取原子对极值。若按残基名称选中了每条链内的多个残基，
加 `--grouping residue` 即可把每个 `(chain ID, residue ID)` 作为独立实体。

旧 `idist` 和 `odist` 暂时保留为隐藏兼容入口，会打印弃用警告并调用统一内核。

### 8.5 大量蛋白体系的聚合物统计

`polystat` 参考 `gmx polystat`，但默认面向包含几十到数千个蛋白拷贝的
DROPPS 体系。只需选择一次分子类型，程序会按 topology molecule/chain ID
拆分所有拷贝：

```bash
dps polystat \
  -s run.tpr \
  -f run.xtc \
  -o polystat.xvg \
  --select "mol PROT"
```

主 XVG 的列数取决于 selection 和分子 cohort 数量，不随蛋白拷贝数增加。
每个 cohort 默认输出：

- `Ree RMS/mean/std`：端到端距离的均方根、算术均值和蛋白总体标准差；
- `Rg RMS/mean/std`：回转半径的均方根、算术均值和蛋白总体标准差；
- `g1/g2/g3 RMS`：每个蛋白先对角化回转张量，再平均特征值所得的三个
  主回转半径；
- `kappa2 mean`：相对形状各向异性的蛋白均值。

逐蛋白先对角化可以避免大量随机取向蛋白在实验室坐标系中被平均成
“近似球形”。`Ree RMS` 和 `Rg RMS` 仍与 `gmx polystat` 的总体量直接可比。

体系中可以同时选择多种分子：

```bash
dps polystat -s run.tpr -f run.xtc -o mixed.xvg \
  --select "mol PROT_A,PROT_B"
```

不同 molecule name、拓扑或选择长度形成不同 cohort，不会静默混合。
同一 molecule name 的不同拷贝若具有不同拓扑或局部选择签名，默认报错，
以捕获缺失残基或错误 NDX；只有确认需要分别处理时才使用
`--allow-heterogeneous-cohorts`。

轨迹每帧先沿拓扑键将完整分子从正交或三斜周期盒中重建，再计算所有指标。
已经预先展开的轨迹可使用 `--no-pbc`。默认回转张量使用质量权重；
`--weight uniform` 改为等权。非连续选择若只需尺寸和形状，可以使用
`--metrics rg shape`。

主链选择为按 topology index 排列的连续成键路径时，还可输出内部距离和
持久长度：

```bash
dps polystat -s run.tpr -f run.xtc -n index.ndx \
  -sel 1 -o polystat.xvg \
  -i internal-distance.csv \
  -p persistence.xvg \
  --tangent-output tangent-correlation.csv
```

`internal-distance.csv` 给出序列间隔 `s` 上的观测数、实际平均轮廓长度、
`<R²(s)>`、其平方根和 `<R²(s)>/s`。密集 lag 使用 FFT 自相关计算，避免
对每个蛋白执行显式的长度平方级 pair loop。蛋白主链的切向相关默认使用
每个 lag；仅在需要复现 all-trans 聚合物约定时使用 `--lag-step 2`。
`persistence.xvg` 使用切向相关首次越过 `1/e` 的位置，同时输出 backbone
steps 和按实际轮廓长度插值得到的 nm。未在可用 lag 内越过的帧写为 `NaN`，
并在终端报告 censored 数量。

默认不写 `帧数 × 蛋白数` 的矩阵。只有显式指定
`--molecule-output molecules.csv` 才流式写出逐蛋白长表；预计超过 1 GiB
或一亿行时会停止并提示，确认后可添加 `--allow-large-output`。

### 8.6 可配置的逐分子/整体选择 RMSD

`rmsd` 的选择对象是“跨所有分子的同一个区域”，而不是一个单独分子。
例如，对所有 `PROT` 分子的 20--80 号残基计算 RMSD：

```bash
dps rmsd \
  -s run.tpr \
  -f run.xtc \
  -o rmsd.xvg \
  --select "(mol PROT) & (resid 20-80)"
```

程序会按 TPR 中的 molecule/chain ID 自动拆分该选择，不需要为每个分子
建立一个 NDX 组。也可以一次给出一个或多个已有组：

```bash
dps rmsd -s run.tpr -f run.xtc -n index.ndx -o rmsd.xvg -sel 3 4
```

每帧先沿拓扑键把完整分子从周期盒中展开，再取所选区域。fit 和计算输出
粒度由两个互相独立的参数控制：

- `--fit-mode none`：不移除物理平移或旋转；
- `--fit-mode molecule`：每个分子分别做 Kabsch fit，默认值；
- `--fit-mode selection`：每个选择中的全部原子共同做一次 Kabsch fit；
- `--output-mode molecule`：按分子计算并输出 RMSD，默认值；
- `--output-mode selection`：把选择中全部原子的加权残差合并为一个 RMSD。

六种组合均可使用。例如，把整个选择作为一个整体 fit 并输出一个 RMSD：

```bash
dps rmsd -s run.tpr -f run.xtc -o rmsd-system.xvg -sel 3 \
  --fit-mode selection --output-mode selection
```

`fit=molecule + output=selection` 会合并逐分子 fit 后的原子残差，并非简单
平均分子 RMSD；`fit=selection + output=molecule` 则先整体 fit，再按分子统计
同一次 fit 的残差。即使使用 `fit=none`，程序仍会消除周期镜像表示差异，
避免过盒产生虚假的盒长跳变，但不会移除亚盒尺度的真实平移和旋转。

默认把每个分子与它自身在第一个分析帧的构象比较；`-ref TIME_NS` 可选择
最接近指定时间的参考帧。默认所有所选粒子等权，`--mass-weighted` 会在
中心化、fit 和 RMSD 中统一使用质量。

分子输出模式下，`rmsd.xvg` 给出逐帧分子均值和总体标准差，`rmsd.csv`
给出分子长表；整体选择输出模式下，两者各包含每个选择的一条结果。只需
XVG 时可添加 `--summary-only`。多粒子分子必须在 TPR 中有连通的键网络，
才能无歧义地处理 PBC。

### 8.7 多分子位置涨落 RMSF

`rmsf` 参考 `gmx rmsf` 的位置总体标准差、质量加权 fit、残基协方差平均和
B-factor 定义，并针对 DROPPS 常见的多拷贝体系默认逐分子独立 fit：

```bash
dps rmsf \
  -s run.tpr \
  -f run.xtc \
  -o rmsf.xvg \
  --select "(mol PROT) & (resid 20-80)"
```

程序会沿拓扑键把完整分子从周期盒中展开，然后把每个分子与它自身在参考帧的
坐标做质量加权 Kabsch fit。默认参考是第一个分析帧，`-ref TIME_NS` 可选择
最接近指定时间的保存帧。`--fit-mode selection` 把一个选择中的全部分子整体
fit，接近 `gmx rmsf` 的单组模型；`none` 保留真实的亚盒尺度平移和旋转。

默认逐原子输出。`-res`（等同于 `--grouping residue`）先对同一残基中各原子的
协方差张量做质量加权平均，再由张量迹开平方，不是对原子 RMSF 做算术平均。
`--fit-weighting uniform` 和 `--group-weighting uniform` 可分别改成等权 fit 与
等权残基汇总。

```bash
dps rmsf -s run.tpr -f run.xtc -n index.ndx -o rmsf-residue.xvg \
  -sel 3 -res -od reference-deviation.xvg \
  -ox average.pdb -oq reference.pdb
```

`rmsf.xvg` 对每个选择/分子类型输出等价位点在不同分子拷贝间的均值和总体
标准差；`rmsf.csv` 保留每个分子和位点的 RMSF、参考偏差、B-factor 与六个
协方差分量。`-od` 输出相对参考坐标的时间均方根偏差，`-ox/-oq` 分别把平均
对齐坐标和参考坐标写成带各向同性 B-factor 的 PDB。PDB 输出要求只有一个
顶层选择，但该选择仍可包含任意数量的分子。只需摘要时使用 `--summary-only`。

完整算法、输出定义和限制见 [RMSF 设计文档](rmsf-design.md)。

### 8.8 粗粒化溶剂可及表面积

`sasa` 参考 `gmx sasa`，把“构造表面的粒子”和“需要汇总输出的表面子集”
分开。`-surface` 中的全部粒子共同遮挡并构成一张表面；`-output` 只从
这张表面提取一个或多个子集，不会为每个子集重新计算孤立表面：

```bash
dps sasa \
  -s run.tpr \
  -f run.xtc \
  -n index.ndx \
  -o sasa.xvg \
  -surface "group 1 | group 2" \
  -output "group 1" "group 2"
```

每个 `-output` 选择都必须是 `-surface` 的子集。surface 总面积始终是
XVG 的第一条曲线，随后按命令行顺序写入额外 output。`-output` 可以重复，
例如 `-output 1 2 -output "res ARG | res LYS"`。

在终端中省略 `-surface` 时，程序显示现有索引组并接受一个 DROPPS 选择
表达式；直接回车表示全部原子。随后可以交互输入任意多个 output，空行结束。
在非交互运行中，缺省 surface 自动取全部原子且不添加额外 output。这里的
“全部原子”直接来自 TPR 拓扑，不依赖自定义 NDX 的第 0 组。

默认把力场 `sigma` 的一半作为珠子几何半径，并使用 `0.14 nm` 探针和
960 个确定性、成对反向的 Fibonacci 球面点。可用参数包括：

- `--radius-convention sigma-half`：半径为 `sigma/2`，默认值；
- `--radius-convention lj-minimum-half`：半径为 `2^(1/6)*sigma/2`；
- `--radius-scale`：统一缩放所有半径；
- `--fixed-radius`：不用力场 sigma，给所有珠子指定同一半径；
- `--probe-radius`：探针半径，设为 0 时得到裸珠可及表面；
- `--sphere-points`：偶数且至少为 4，生产计算建议先检查点数收敛。

Ashbaugh--Hatch 力场使用逐类型 sigma；Wang--Frenkel/MPiPi 使用同类型
自相互作用 `sigma_ii`。这些都是粗粒化几何约定，输出头会记录采用的半径
规则、缩放、探针和球面点数，不应在没有额外标定时把结果直接解释为水合
自由能。

每一帧都使用自身的盒向量和周期最小镜像。即使 OpenMM 原始坐标离开主盒
许多倍，程序也先在 float64 中归一化，再建立周期邻居；不会通过只枚举
`-1/0/+1` 三个坐标平移来猜测镜像。最大扩展球重叠 cutoff 必须小于当前
盒最短晶格矢量的一半，否则最小镜像不唯一，程序会停止。XVG 单位为
`nm^2`，并在头部给出每条曲线的轨迹均值和总体标准差。

### 8.9 周期镜像自相互作用分析

`pbcontact` 用于检查有限周期盒是否使分子或组装体与自身镜像发生接触。程序先依据拓扑中的键连接把每个分子完整展开，再用指定 cutoff 建立带周期晶格位移的分子接触图，因此分子仅仅跨过盒边界不会被误判。

判据是：如果接触图中存在一个回到同一分子、但累计晶格位移不为零的闭合路径，该连通组分就与自己的周期镜像相连。这个判据同时包含：

- 分子直接接触自身镜像；
- 经另一个分子连到自身镜像；
- 经两个、三个或任意更多中间分子连到自身镜像。

例如，分析索引组 1 中的接触原子：

```bash
dps pbcontact \
  -s run.tpr \
  -f run.xtc \
  -sel 1 \
  -c 0.7 \
  -mc 1 \
  -o image_check
```

默认生成：

| 文件 | 内容 |
|---|---|
| `image_check.frames.csv` | 每帧的镜像分子数、直接/仅间接镜像分子数、镜像百分比和绕盒组分数 |
| `image_check.summary.csv` | 整条轨迹的汇总百分比 |

汇总文件给出两个主要口径：

- `frame_percentage`：至少存在一个镜像相互作用的帧数，占分析帧数的百分比；
- `molecule_percentage`：发生镜像相互作用的“分子-帧”观测数，占全部“分子-帧”观测数的百分比。

若还需要定位具体分子，可添加 `--molecule-output image_check.molecules.csv`。其中 `direct_mirror_interaction=0` 且 `mirror_interaction=1` 表示该分子只通过其他分子的接触网络连到自己的镜像。

`-mc/--minimum-contacts` 是形成一条特定“分子—分子镜像”边所需的最少原子对接触数，可用来排除偶然的单粒子接触。cutoff 必须小于当前周期盒最短晶格矢量的一半；否则最小镜像不唯一，程序会停止并指出出错帧。多粒子分子必须在 TPR 拓扑中有连通的键网络，才能无歧义地展开。

### 8.10 径向分布函数与累计配位数

`rdf` 参考 `gmx rdf`，用一个参考选择计算到一个或多个目标选择的径向分布：

```bash
dps rdf -s run.tpr -f run.xtc \
  -ref "mol PROT" \
  -sel "mol SOL" \
  -o protein-solvent.xvg \
  -cn protein-solvent-cn.xvg
```

`-sel` 可重复，输出中每个目标占一条曲线。默认保留所选原子作为位置；分子
COM-COM RDF 可写为：

```bash
dps rdf -s run.tpr -f run.xtc \
  -ref "mol PROT" -sel "mol PROT" \
  --position com --grouping chain \
  --exclude same-molecule -o protein-com.xvg
```

位置类型支持 `atom/cog/com`，分组支持 `all/chain/residue/atom`；
`--ref-position`、`--sel-position` 和两侧 grouping 选项可分别覆盖公共设置。
`--norm rdf` 输出无量纲 `g(r)`，`number-density` 输出径向数密度，`none`
输出每个参考位置的距离计数密度。`-cn` 始终从原始计数计算累计配位数，不受
`--norm` 选择影响。

周期边界默认开启，并支持正交和 triclinic 盒。省略 `--rmax` 时，程序扫描所有
选中帧，按最小安全周期半径自动选择并对齐到完整 bin；显式半径超过任一帧的
安全值会报错。`--geometry xy` 计算平面内 RDF，`--surface chain|residue` 计算
目标位置到每个参考分子/残基最近所选原子的距离。排除模式支持
`topology`（ITP 的 1-2/1-3 排除）和 `same-molecule`；与 GROMACS 一致，排除
不会修正归一化分母。

## 9. 常见问题

### `grompp` 报未知 MDP 参数

DROPPS 只实现自己的 MDP 参数子集，不等同于 GROMACS。删除不支持的参数，并查阅 [MDP 参数说明](mdp-reference.md)。参数名中的 `-` 与 `_` 等价，但布尔值必须写成 `True` 或 `False`。

### 找不到 OpenMM 或 MDAnalysis

确认当前使用的是安装 DROPPS 的同一个虚拟环境：

```bash
which python
which dps
python -m pip show dropps openmm MDAnalysis
```

### GPU 没有被使用

显式指定平台并观察启动输出：

```bash
dps mdrun -s run.tpr -o run --platform CUDA -gpu_id 0 --precision mixed
```

如果 CUDA 不在 “Available OpenMM platforms” 列表中，需要先修复 OpenMM 与 GPU 驱动环境。

### 分析命令选错了分组

先运行 `dps check` 查看默认分组，必要时用 `dps make_ndx` 创建命名分组。请特别注意 DROPPS 原子索引从 0 开始。

### 长任务如何适配作业调度时限

使用略短于队列时限的 `-maxh`，让 DROPPS 在限制到达前保存 restart：

```bash
dps mdrun -s run.tpr -o run -maxh 23.5 -cpt 10
```

下一次作业使用相同 TPR 和输出前缀，并添加 `-cpi`。

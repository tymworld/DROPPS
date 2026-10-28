> Historical audit of the 0.5.0 beta line. Current validation is in [release-checklist](../release-checklist.md).

# DROPPS 发行版本审计与代码更新进度

最后审计：2026-08-14
审计范围：历史 `dist/` 制品、`release/0.5.0b4/` beta 制品、当前源码、测试与构建配置

## 1. 结论

`dist` 中共有 14 个版本、28 个正式命名的 Python 制品。每个版本都同时提供 wheel 和 sdist；全部制品均通过压缩包完整性、版本元数据、wheel `RECORD` 哈希与大小校验。每个版本的 wheel 与 sdist 所含 `dropps/` 代码也逐文件一致。

当前项目版本为 `0.5.0b4`，beta wheel、sdist 与 SHA-256 清单位于
`release/0.5.0b4/`。发布脚本验证了版本元数据、禁止文件、wheel/sdist 载荷、
wheel 版本导入和双制品独立安装。当前代码进度定义为：

- `0.4.2`：OpenMM 8.1 旧运行栈的最后正式版。
- `0.5.0b3`：上一个 beta 基线。
- `0.5.0b4`：当前 beta 测试版本；源码和制品已经准备完成，仍需补齐 OpenMM 8.5 与 CUDA 硬件矩阵后再评估正式版。

## 2. 审计摘要

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 历史制品盘点 | 通过 | `dist/` 中 14 个版本、28 个历史制品均已审计 |
| 0.5.0b4 beta 制品 | 通过 | wheel、sdist 与 `SHA256SUMS` 齐备 |
| 名称和版本元数据 | 通过 | 文件名与 `Name: dropps`、`Version: 0.5.0b4` 一致 |
| 发布载荷清洁 | 通过 | 不含缓存、AppleDouble、Synology 元数据或 backups |
| Python 3.9 语法解析 | 通过 | 当前 163 个 Python 文件以 3.9 AST 全量解析 |
| 当前回归测试 | 通过（单环境） | Python 3.12 / OpenMM 8.1 / MDAnalysis 2.10：352/352 通过 |
| 安装冒烟 | 通过 | wheel/sdist 独立安装；wheel 的 `-v`、`-db`、`trjconv -c/-se` 通过 |
| 版本控制追踪 | 阻塞 | `.git/refs` 仍受 NAS 元数据污染，无法可靠创建标签 |

说明：版本差异统计仅比较 `dropps/` 有效载荷，排除 `.DS_Store`、`._*`、`@eaDir`、`__pycache__` 和构建元数据。`A/M/D` 分别表示相对上一发行版新增、修改和删除的有效载荷文件数。

## 3. 版本演进总表

| 版本 | 类型 | Python 文件 | A/M/D | 制品审计 | 主要进展 |
| --- | --- | ---: | ---: | --- | --- |
| `0.1.1` | 正式版 | 42 | 基线 | 通过 | 建模、运行和基础轨迹分析工作流 |
| `0.2.0` | 正式版 | 43 | 1/6/0 | 通过 | 新增 `modifyres`，扩展力场与系统构建 |
| `0.2.1` | 正式版 | 43 | 0/2/0 | 通过 | 修复密度时间步单位和 ITP 角度第三原子索引 |
| `0.3.0` | 正式版 | 45 | 2/5/0 | 通过 | 新增 `trjconv`、`gsd2xtc`，更新轨迹处理 |
| `0.3.1` | 正式版 | 45 | 1/2/0 | 通过 | 角度力常数改用 radian 单位；加入 HPSC 力场 |
| `0.3.2` | 正式版 | 45 | 0/3/1 | 通过 | 更新 PTM 参数/输入形式；移除 HPSC 力场 |
| `0.4.0b0` | beta | 51 | 11/32/2 | 通过 | 新 CLI 入口、接触/组装分析、CALVADOS2/MPiPi 力场 |
| `0.4.0b1` | beta | 51 | 0/3/0 | 通过 | 补全 Wang–Frenkel 参数校验；修正 MPiPi 蛋白/RNA 键 |
| `0.4.0` | 正式版 | 51 | 0/0/0 | 通过 | 与 `b1` 代码相同；放宽 MDAnalysis 和 `bio` 依赖下限 |
| `0.4.1` | 正式版 | 73 | 23/40/1 | 通过 | 运行恢复/能量/应力张量；相共存、交换、相 MSD、表面张力、黏度等分析 |
| `0.4.2` | 正式版 | 73 | 0/0/0 | 通过 | 与 `0.4.1` 代码相同；标记为 OpenMM 8.1 最后支持版并恢复 Poetry 构建 |
| `0.5.0b1` | beta | 75 | 2/9/0 | 通过 | TPR v2 可移植格式、旧 TPR 转换、运行恢复兼容升级 |
| `0.5.0b2` | beta | 80 | 6/21/1 | 通过 | Python 3.9/GPU 安装方案、统一 `trjconv`、RMSD、周期镜像相互作用、帮助系统 |
| `0.5.0b3` | beta | 83 | 3/23/0 | 通过 | 统一物理时间选择和密度几何；对齐 density/coexistence；补齐分析依赖 |
| `0.5.0b4` | beta | 102 | 19/10/0 | 通过 | 完成分析工具集、命令架构、CLI 统一、发布自动化、文档和制品清理 |

## 4. 逐版本代码更新记录

### 0.1.1

首个保留在 `dist` 中的完整发行基线。包含 PDB/ITP 建模、弹性网络、网格装箱、`grompp`、`mdrun`，以及索引、密度、回转半径、接触、距离、角度和 MSD 等分析。Git 历史中可见 `0.1.0` 发行提交，但 `dist` 没有 `0.1.0` 制品，因此本审计从 `0.1.1` 开始。

### 0.2.0

- 新增 `modifyres`，支持在结构与拓扑中应用残基修饰。
- 更新系统构建和力场读取逻辑。
- 调整 HPS/HPST 力场数据。

### 0.2.1

- `density` 将 OpenMM 时间步显式转换为 ps 后再计算帧号，修复量纲混用。
- ITP 角度读取由错误的 `j-j` 索引修正为 `j-k` 索引。

### 0.3.0

- 新增 `trjconv` 和 `gsd2xtc`，开始形成统一轨迹转换能力。
- 更新轨迹抽象、ITP 读取和接触图实现。
- 旧 `contact_number` 从命令注册表中停用。

### 0.3.1

- 角度力常数从 `kJ/mol/degree²` 修正为 `kJ/mol/radian²`。
- 新增 HPSC 力场文件。

### 0.3.2

- `pdb2dps -pt` 改为一次接收多个修饰，并明确修饰字符串格式。
- `modifyres` 同步采用 radian 角度力常数。
- 更新 HPS 磷酸化残基的尺寸、疏水性、电荷、键和角度参数。
- 移除上一版本加入的 HPSC 力场。

### 0.4.0b0

- 首次在发行元数据中提供 `dps = dropps.dps:main` 控制台入口。
- 新增 `contact`、`cstat`、`assembly`、`pdb2bond` 等分析命令。
- 将旧接触图/接触数实现标记为 deprecated。
- 新增 CALVADOS2、MPiPi、MPiPi_PTM 力场。
- 大范围更新索引、轨迹、参数、系统构建和既有分析命令。

### 0.4.0b1

- Wang–Frenkel 非键参数改为覆盖完整原子类型集合，并对缺失 pair 显式报错。
- 动态 LJ 截断增加无效 `sigma` 防护。
- MPiPi 将蛋白键和 RNA 键拆分为不同平衡长度。

### 0.4.0

与 `0.4.0b1` 的 `dropps/` 有效载荷完全相同。发布变化仅在版本和依赖元数据：MDAnalysis 下限由 2.9 降至 2.7，`bio` 下限由 1.8 降至 1.6.2。

### 0.4.1

这是一次大规模功能扩展：

- 模拟运行：新增可恢复运行、进度/能量输出、应力张量报告和 configurational rerun。
- 相分离分析：新增 `coexistence`、`exchange`、`phase-msd`。
- 热力学分析：新增 `energy`、`surftension`、`viscosity-gk`。
- 接触分析：新增 `contact.ng` 和 `timecorr`。
- 命令基础设施：新增统一参数解析器、运行配置、重启和报告器模块。
- `pdb2cgps` 对外命令名迁移为 `pdb2dps`。

该版本 wheel 由 setuptools 83 构建，sdist 还包含 `setup.py`、`setup.cfg` 和 `dropps.egg-info`；这与 `pyproject.toml` 声明的 Poetry 后端以及其他版本的构建路径不一致。

### 0.4.2

`dropps/` 有效载荷与 `0.4.1` 完全相同。主要发行变化是：

- 项目描述明确标记为“OpenMM 8.1 最后支持版”。
- wheel 构建器恢复为 poetry-core；sdist 不再携带 setuptools/egg-info 附加文件。

### 0.5.0b1

- 引入版本化 ZIP 容器 TPR v2，使用 OpenMM XML、JSON 和 NumPy 数组替代对 pickle 类路径的依赖。
- 新增 `convert-tpr`，用于将受信任的旧 TPR 转换为可移植格式。
- 更新 `grompp`、`mdrun` 和重启逻辑以支持新旧 TPR 迁移。
- 包内新增可靠的 `__version__` 获取逻辑。
- 默认依赖范围扩展到 OpenMM 8.x 和 NumPy 1/2，并提供 `latest`、`openmm81` extras。

### 0.5.0b2

- 最低 Python 版本调整为 3.9，并按 Python 版本拆分 NumPy/OpenMM/MDAnalysis 依赖。
- 新增 CUDA 12、CUDA 13 extras 和 `scripts/install_dropps.py` GPU 检测/验证安装器。
- 重写 `trjconv`：按物理时间选帧，统一 PBC、center、fit、PDB/XTC 输出。
- 删除旧 `extract` 命令，其单帧能力由 `trjconv` 覆盖。
- 新增 PBC-aware `rmsd` 和周期镜像相互作用命令 `pbcontact`。
- 新增命令帮助基础设施，并改进 CLI 错误与版本显示。
- 改进 warm-up/production 检查点恢复和压力张量时间切片处理。
- 将依赖名从 `bio` 更正为 `biopython`。

### 0.5.0b3

- 新增共享 `time_selection`，按实际轨迹时间戳处理 `-b/-e/-dt`，并显式检查非均匀采样。
- 新增 `density_core`，统一逐帧盒向量、体积、周期分数坐标和稠密区中心定位。
- `density` 与 `coexistence` 共享中心、阈值和 bin 定义，避免参考密度曲线口径不一致。
- `density` 改为逐帧使用实际正交盒与体积后再平均。
- 更新多项分析命令的时间选择与均匀采样校验。
- 补齐 `scipy`、`pandas`、`openpyxl`、`gsd` 运行依赖。

## 5. 当前 `0.5.0b4` 更新进度

本 beta 版本在 `0.5.0b3` 基线上进一步统一 CLI：所有功能命令具有一致的生命周期摘要，
`INFO/OK/WARNING/ERROR/DEBUG` 按级别着色且重定向不含 ANSI；全局
`-db/--debug` 输出运行环境、解析参数、耗时和 traceback。公开参数统一为小写
简称加小写 kebab-case 长名称；部分旧小写非标准别名从帮助隐藏，大写选项已移除。所有
公开 Input/Output 均保证具有无冲突的一至两字母小写简称，且 Output 简称由
`o` 开头，例如 `genmesh` 的 `-o/--output-conformation` 与
`-op/--output-topology`；高频参数按命令内冲突分配简称，例如
`trjconv -c/--center` 与 `-ff/--forcefield`。新增测试会比较每个命令解析器的
Namespace 与执行模块实际读取的 `args.*` 字段，并逐项检查简称格式。

| 发布门槛 | 状态 | 证据/下一步 |
| --- | --- | --- |
| 版本号统一为 `0.5.0b4` | 完成 | `pyproject.toml`、wheel、sdist 与 CLI 一致 |
| wheel + sdist 齐备 | 完成 | `release/0.5.0b4/` 中两个制品均存在且可完整读取 |
| 制品内部完整性 | 完成 | wheel 元数据、sdist 元数据、禁止文件和 SHA-256 均通过 |
| wheel/sdist 独立安装 | 完成 | 两种安装均成功，`dps --version` 返回 0.5.0b4 |
| Python 3.9 语法兼容 | 完成 | 163 个 Python 文件 AST 全量解析通过 |
| CPU 回归测试 | 部分完成 | OpenMM 8.1 单环境 352/352 通过；仍需两端版本矩阵 |
| OpenMM 8.1 迁移测试 | 完成（现有环境） | 旧 TPR 转换、读取、恢复与短程运行均在测试中通过 |
| OpenMM 8.5 当前栈测试 | 未完成 | 安装 `.[latest]`，验证 CLI、TPR v2、分析与恢复 |
| CUDA 12/13 安装测试 | 未完成 | 在对应 NVIDIA 驱动机器验证安装器和真实 CUDA Context |
| Git 追踪 | 未完成 | 将源码冻结提交与正式制品哈希绑定；制品不进入源码 Git |
| Git 发行标签 | 未完成 | 清理非法 refs 后，为可追溯发行创建标签 |
| 发布目录清洁 | 完成 | `release/0.5.0b4/` 只有两个白名单制品和 SHA-256 清单 |

结论：`0.5.0b4` 已达到“源码、测试、beta 制品准备完成”的状态；未完成项属于
Git 可追溯性和跨运行栈/硬件验证，因此当前仍应作为 beta 测试版本。

## 6. 已发现问题与风险

### 高优先级

1. **Git 引用被 NAS/macOS 元数据污染。** `.git/refs` 和 `.git/objects` 内存在 `@eaDir`、`._*` 文件；`git show-ref` 会报非法 ref。清理前无法可靠建立发行标签和完整追溯链。清理 `.git` 前必须先备份仓库并采用可恢复操作。
2. **测试矩阵尚不完整。** 当前 352 项测试只在 Python 3.12 / OpenMM 8.1 / MDAnalysis 2.10 环境验证；OpenMM 8.5 / NumPy 2、Python 3.9 和 CUDA 12/13 实机仍需单独验收。

### 中优先级

1. **旧版本 Python 元数据过宽。** `0.1.1–0.4.0` 声明 `Requires-Python >=3.10`，但源码使用 Python 3.12 引入的宽松 f-string 语法；在 Python 3.9 解析时每版出现 8–11 个语法错误，Python 3.10/3.11 同样不支持该语法。应在历史兼容表中标记这些版本实际需要 Python 3.12+，或重建修复版。
2. **历史制品含系统文件。** `0.1.1–0.4.0` 的 wheel 与 sdist 均包含 `dropps/.DS_Store` 和 `dropps/share/.DS_Store`。`0.4.1` 起已清除。
3. **构建链不一致。** `0.4.1` 使用 setuptools 83 构建，其他版本使用 poetry-core；这降低了复现性。
4. **`dist` 是历史归档而非发布 staging。** 本次上传必须只使用干净的 `release/0.5.0b4/` 白名单，不能批量上传历史 `dist/`。
5. **缺少已列出的 Git 发行标签。** 当前 `git tag -n` 未列出任何版本标签，历史版本与源提交主要依赖提交消息和制品内容关联。

## 7. 建议的正式版验收顺序

1. 备份并清理 `.git` 内的 `@eaDir`/`._*` 非 Git 数据，确认 `git show-ref` 和 `git fsck` 恢复正常。
2. 使用 Python 3.9 + OpenMM 8.1、Python 3.10+ + OpenMM 8.5 两组 CPU 环境补齐完整测试矩阵。
3. 在 CUDA 12/13 机器验证安装器、真实 CUDA Context 和短程模拟。
4. 冻结源提交，把提交 ID 与 `release/0.5.0b4/SHA256SUMS` 一并保存。
5. 创建 `v0.5.0b4` 附注/签名标签，只上传清单中的两个白名单制品。

## 8. 制品 SHA-256

以下哈希可用于后续镜像、传输和发布核对。

| 版本 | wheel SHA-256 | sdist SHA-256 |
| --- | --- | --- |
| `0.1.1` | `6bd87a85b57ed1faf94ad1ba122e09d7c4d48d77c39ee49a4e693ff625388621` | `77f86359ad8db4a9d05a6f035533209bfdd505858662e6f6053c091ba04f0f63` |
| `0.2.0` | `c48445d9ea64b789fa7899a23798d9dfed6bf9a02d497fc3059a6deb362a3b6e` | `247a94b2f5ce89d5a79e5a37b6d4bf9aa817ef6ea2982ecddc28d0a80a1dd33c` |
| `0.2.1` | `b6cd5380381797cd8f41edfb19b3fc56ae60c6692dcd429a86ec74aba4365634` | `86330e132f4e645e211cf3392717e922b1e5613daf526de5a6cd2f5d4d9dabac` |
| `0.3.0` | `4aab7a7c89a7436d2cee035d14f08dd1dba853405d26998f4c7485e6d8237369` | `7f4db2948487cbceaf08582db56d38fa53e36d7004f2e3c703d4f6721c5d4ad6` |
| `0.3.1` | `e5f696ba2afa59d94fed984885ef6078625694f0a8c91291771cf2ea099bebe8` | `5f9556790a82217790b89fff54c7631e2488cae4bc81e595d1d34d3c302aab07` |
| `0.3.2` | `19798b67a876ee4dee3702e428cc9261379b8e3cd3b6429f936ae5300fc726ae` | `a0ab721bfbb56f673cf61e7862fcb812e5049108ec1600b47a36d87aae8dc249` |
| `0.4.0b0` | `41704cf929c8efebd05a8fa155298c30aef094acb989162ad903a622d0c8af0a` | `967950d315a2cff059742b85df93520a8e3c6f25556d81fd8c2b2fd77daa406b` |
| `0.4.0b1` | `9eeb844bd3899d648a78a981990a0bdbc12f9e532381febcc1636948d462a3b9` | `36b532568fafd0a42079319cbbc452461d7498be85d9978b5144473f12225ca1` |
| `0.4.0` | `6cdb81d39461a9ca5dc1c8862abed1a0c48e03d6e2890202a0ad60c602cea58b` | `5f53db69351299e1acd34dd2602c4c66d8c4740ecc7f04c7d2ed3b813c7bcab5` |
| `0.4.1` | `f12cd972e214d7b30b42410da39a5cae5558c1488f21cd2eb001188e4de38356` | `f5c3bc4d61033fa05263ebff3068cfb4b1925a9a97209eb20efaddd7433ab883` |
| `0.4.2` | `706afb5e2c5db083cdbf6d2a8621aefae0a4bd3c020566d875c51587eb2bd6c3` | `9e2dc96abe607ad503bc1f9a74f978f947a23f2b2e891e4e0c07649bed20e550` |
| `0.5.0b1` | `6e67e915b66b36367e22a35781feda7e9dffd8cabbea38abe40b9e0fb5ad3c93` | `d3650528a8aa2873160905c85245f295f4ea972fd11c6a8b5deadde135ae02b2` |
| `0.5.0b2` | `eed2c4d0861fd31b0177f9647ccb6a5ddf022c61a2e988f5490c21f808155f3a` | `5a9914661057853afffd79c426b0237af9ede5e732481c34aed38c622812da4c` |
| `0.5.0b3` | `c47c8a3f7a77610bf6669c9be505c2aee40dda20da37ef88adb0f4ec46dd884d` | `fe35b62bd1ae471cbca0b2accf6f35729d0a12156c7f878699b3057079ca6b87` |
| `0.5.0b4` | 见 `release/0.5.0b4/SHA256SUMS` | 见 `release/0.5.0b4/SHA256SUMS` |

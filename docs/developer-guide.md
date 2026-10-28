# DROPPS 开发文档

本文档面向维护 DROPPS、添加命令、修改模拟内核或扩展分析功能的开发者，内容对应 DROPPS 1.0.0。整体模块边界和依赖方向另见 [架构说明](architecture.md)。

## 1. 项目概览

DROPPS 是一个以 Python 实现的命令行应用。公开入口由 `pyproject.toml` 注册：

```toml
[project.scripts]
dps = "dropps.dps:main"
```

一次普通调用的路径为：

```text
dps argv
  -> dropps.dps.main
  -> dropps.share.all_commands 查找命令
  -> 命令模块的 getargs_* 解析参数
  -> dropps.share.console 输出统一生命周期与诊断
  -> 命令模块的主函数执行任务
```

项目把建模/运行命令、分析命令、文件格式和共享运行设施分开，但所有命令仍通过同一个 `dps` 入口发布。

## 2. 开发环境

### 2.1 安装

```bash
git clone https://github.com/tymworld/DROPPS.git
cd DROPPS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python scripts/install_dropps.py --editable
```

当前依赖约束定义在 `pyproject.toml`：Python >=3.9、tqdm >=4.67 和 Biopython >=1.80。Python 3.10+ 使用 NumPy >=1.26.4,<3、OpenMM >=8.1,<9 和 MDAnalysis >=2.9；Python 3.9 使用兼容组合 NumPy >=1.26.4,<2、OpenMM >=8.1,<8.2 和 MDAnalysis >=2.7,<2.8。`cuda12` 与 `cuda13` extras 在 Python 3.10+ 安装对应的 OpenMM CUDA 平台；`latest` 是无 CUDA 强制要求的 NumPy 2.x/OpenMM >=8.5,<9 组合；`openmm81` 用于 Python 3.9 兼容、旧环境兼容和迁移。

`scripts/install_dropps.py` 是面向最终用户的硬件感知安装入口。修改它时应保持以下性质：仅把成功的 `nvidia-smi` 设备查询视为 NVIDIA GPU 可见；根据驱动报告的最高 CUDA 兼容版本选择 extra；GPU 可见但驱动低于 CUDA 12 时拒绝安装；CUDA 安装后必须创建真实的 OpenMM CUDA Context。GPU 节点的严格安装检查可使用：

```bash
python scripts/install_dropps.py --editable --require-cuda
```

### 2.2 基本检查

```bash
dps --version
dps help commands
python -m unittest discover -s tests -p 'test_*.py'
```

完整命令导入会加载 OpenMM、MDAnalysis 及其分析依赖；只安装部分依赖时，`dps --version` 仍可工作，但命令列表和测试套件可能无法导入。

运行单个测试文件或测试方法：

```bash
python -m unittest tests.test_mdrun_config
python -m unittest \
  tests.test_mdrun_config.MdrunConfigurationTests.test_individual_output_paths_override_prefix_defaults
```

测试方法名称以当前源码为准，可先运行完整测试模块查看可用测试。

## 3. 目录结构

```text
DROPPS/
├── dropps/
│   ├── dps.py                 # CLI 入口
│   ├── commands/              # 建模、转换、grompp、mdrun、rerun
│   ├── analysis/              # 轨迹和热力学分析命令
│   ├── fileio/                # PDB、ITP、TPR、XVG、XPM 等格式
│   ├── share/                 # 参数、系统构建、续跑、reporter、索引等
│   └── hp/                    # 启动横幅、常量和文案
├── tests/                     # unittest 测试与小型夹具
├── scripts/                   # 维护脚本
├── docs/                      # 用户、开发和 MDP 文档
├── online/                    # 在线构建相关 notebook
├── pyproject.toml             # 包元数据、依赖、入口与构建配置
└── requirements.txt           # 推荐运行依赖的简化列表
```

`dropps/backups/` 不参与打包，应视为历史参考而不是生产实现。

## 4. CLI 架构

### 4.1 命令对象与注册表

每个命令以 `single_command` 保存四项信息：命令名、参数解析函数、执行函数和描述。`dropps/share/all_commands.py` 把命令分成 modeling、simulation 和 analysis 三组，并建立统一注册表。

`dropps/dps.py` 负责：

1. 快速处理 `--version`，此路径不加载重型科学依赖；
2. 无参数时转换为 `help`；
3. 从注册表取出解析函数和执行函数；
4. 打印启动信息；
5. 解析命令参数、输出统一执行摘要并调用实现；
6. 将旧模块输出映射为统一级别，报告完成状态和可选 debug 信息。

### 4.2 共享参数解析器

所有公开命令必须使用 `dropps.share.argument_parser.ArgumentParser`，不要直接使用标准库 `argparse.ArgumentParser`。共享解析器会：

- 把选项分为 `Input`、`Output` 和 `Parameters`；
- 在帮助中标记 required 或 default；
- 为文件和前缀生成一致的 metavar；
- 标记 required mutually-exclusive group；
- 自动提供全局 `--debug`；
- 要求公开选项具有小写 kebab-case `--long-name`；
- 保证所有公开 Input/Output 选项具有一个无冲突的小写一至两字母简称；
- 强制所有 Output 简称由 `o` 开头；
- 帮助和 usage 只展示小写标准简称与标准长名称，继续解析旧别名；
- 禁用长选项缩写，避免新增选项后产生歧义。

新增参数时，长名称是稳定的公开接口，必须使用 `--lowercase-kebab-case`。
短名称只能是单横线加一至两个小写字母，例如 `-f`、`-op`。Input/Output 的语义
简称集中维护在 `_ROLE_SHORT_OPTIONS`；命令没有显式声明时，解析器会按语义和
确定性备选顺序补齐且避免冲突，所有 Output 简称必须以 `-o` 开头。高频
Parameters 的公共简称包括 center 的 `-c`、selection 的 `-se`、reference 的
`-r`、delta-time 的 `-dt`、time-unit 的 `-tu`、center-axis 的 `-ca`、
bin-width 的 `-bw` 和 forcefield 的 `-ff`，集中维护在
`_COMMON_SHORT_OPTIONS`。两套映射都不得覆盖其他选项已经占用的简称。不要为新
参数添加 `-gpu_id`、`-sel` 这类超过两个字母或含下划线的形式。兼容旧拼写时，
可把旧别名放在同一个 `add_argument` 中，解析器会自动从帮助和描述中隐藏并
替换它。

不常见的选项可显式指定分区：

```python
parser.add_argument(
    "--destination",
    section="output",
    help="Output path.",
)
```

帮助信息的一致性由 `tests/test_argument_parser.py` 和 `tests/test_cli_help.py` 约束。

### 4.3 终端输出与 debug

`dropps.share.console` 是运行状态输出边界。CLI 为每个功能命令统一打印命令名、
输入、输出、参数和完成状态；旧模块中的 `##`、`WARNING:` 和 `ERROR:` 会在命令
边界映射为 `[INFO]`、`[WARNING]` 和 `[ERROR]`。新代码应直接调用
`console.info()`、`console.success()`、`console.warning()` 或
`console.error()`，不要继续创造新的文本前缀。

`[INFO]`、`[OK]`、`[WARNING]`、`[ERROR]`、`[DEBUG]` 分别使用蓝、绿、黄、红、
暗青色；非 TTY 输出不带 ANSI。必须尊重 `NO_COLOR` 和
`DROPPS_COLOR=auto|always|never`。警告、错误和 debug 写 stderr，正常状态和成功
写 stdout；但不能仅因为第三方库或进度条使用 stderr 就把内容升级为 warning。
机器可读的文件内容不要经过状态日志接口。

`--debug` 可位于命令前后，输出 argv、工作目录、Python/平台、关键依赖版本、
解析结果、耗时和 traceback。正常模式只显示简洁异常以及重新加 `--debug` 的
提示。新增耗时阶段可以添加 debug 细节，但不应把逐帧数据全部写到终端。

## 5. 添加一个命令

以下流程适用于 `dropps/commands/` 或 `dropps/analysis/`。

### 5.1 建立模块

```python
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command


prog = "example"
desc = "Describe the command in one sentence."


def getargs_example(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-f", "--input", required=True, help="Input data file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output result file."
    )
    return parser.parse_args(argv)


def example(args):
    # Validate inputs, perform the calculation, then write the result.
    ...


example_commands = single_command(
    prog, getargs_example, example, desc
)
```

### 5.2 导入并注册

如果模块位于 `dropps/commands/example.py`：

1. 在 `dropps/share/all_commands.py` 直接导入 `example_commands`；
2. 在同一文件的适当分组加入 `example_commands`。

分析模块同理，加入 `commands_analysis`。`dropps.commands` 和 `dropps.analysis`
的包初始化器必须保持轻量，不应在其中重新导入所有命令。

### 5.3 更新命令目录

README 的命令段由源码生成，不要手工编辑标记之间的内容：

```bash
python scripts/update_readme_commands.py
```

脚本通过 AST 读取注册表、描述和 required 参数，不导入 OpenMM 等运行依赖。

### 5.4 添加测试

至少覆盖：

- `-h` 能成功渲染，并包含正确的 Input/Output/Parameters 分区；
- 必填参数和互斥参数被解析器强制执行；
- 纯计算逻辑的正常值、边界值和无效值；
- 输出扩展名、元数据、单位和列顺序；
- 如果修改续跑或文件格式，覆盖 round trip、损坏文件和兼容性路径。

新增纯数值分析时，优先把算法放入 `*_core.py`，使其只依赖 NumPy 等轻量组件；命令模块负责轨迹读取、选择、进度、参数检查和输出。现有 `coexistence_core.py`、`exchange_core.py`、`phase_msd_core.py`、`surface_tension_core.py` 与 `viscosity_gk_core.py` 可作为范例。

## 6. 数据和文件格式

### 6.1 建模输入链

```text
PDB + TOP/ITP + MDP
        |
        v
      grompp
        |
        v
      TPR v2
        |
        +--> mdrun --> XTC / EDR / log / restart / final PDB
        |
        +--> analysis commands --> XVG / XPM / CSV / PDB
```

`dropps/share/build_system.py` 根据结构、拓扑和参数构建 OpenMM System；`dropps/commands/grompp.py` 负责收集输入源文件并调用 `write_tpr`。

### 6.2 TPR v2

`dropps/fileio/tpr_reader.py` 是 TPR 的唯一读写边界。TPR v2 是 ZIP 容器，关键成员包括：

| 成员 | 内容 |
|---|---|
| `manifest.json` | 格式版本、DROPPS/OpenMM/Python 版本、哈希和标识 |
| `system.xml` | OpenMM System XML |
| `topology.json` | OpenMM 拓扑的可移植表示 |
| `positions.npy` | nm 单位坐标数组，禁止 pickle |
| `parameters.json` | 解析后的 MDP 参数 |
| `pdb.json` | 初始 PDB 元数据 |
| `itp.json` | 分子 ITP 数据 |
| `sources/...` | `grompp` 收集的原始输入文件 |

所有主要成员都由 manifest 中的 SHA-256 校验和保护。`run_id` 标识完整运行输入；`system_id` 忽略步数、随机种子和输出设置等续跑无关字段，用于判断系统兼容性。

开发约束：

- 新代码通过 `read_tpr`/`write_tpr` 访问 TPR，不要直接假设 ZIP 内部布局；
- 新 TPR 不得使用 pickle；
- 旧 pickle 只允许作为受信任迁移输入，因为反序列化可以执行代码；
- 修改成员格式时递增格式版本，并同时添加 round-trip、checksum 和旧版本测试；
- 写入保持原子性，失败时不能留下看似完整的目标文件。

### 6.3 EDR、日志和分析输出

DROPPS `.edr` 是版本化 CSV，而不是 GROMACS EDR。实现位于 `dropps/share/energy_reporter.py`，包含固定头部元数据、字段名、单位和数据行。新增字段时应同时更新：

- 字段注册和单位；
- reporter 计算；
- `dps energy` 的选择与导出；
- MDP 文档和相关测试。

XVG/XPM 写入优先复用 `dropps/fileio/` 中的 helper，避免不同命令产生不兼容的标题、单位或矩阵方向。

## 7. 模拟运行与 restart

### 7.1 科学配置和运行配置分离

MDP 参数在 `grompp` 阶段由 `dropps/share/parameters.py` 解析并写入 TPR。`mdrun` 的平台、输出路径、墙钟 checkpoint、最大运行时长以及临时 `nsteps`/seed 覆盖由 `dropps/share/mdrun_config.py` 解析。

修改配置时保持这个边界：

- 改变体系物理意义或积分行为的设置进入 MDP/TPR；
- 与机器、调度、路径或单次执行有关的设置进入 `mdrun` CLI；
- CLI 覆盖不应改写原 TPR。

### 7.2 输出路径

`resolve_output_paths` 从 `-o/--output-prefix/-deffnm` 产生默认 `.log`、`.edr`、`.xtc`、`.chk`、最终 `.pdb` 和可选 `.stress.xvg`，再应用单文件覆盖。新增输出时，应同步考虑：

- 默认路径和显式覆盖；
- 新运行、append 续跑与 `--no-append` 分段；
- checkpoint 之后的回滚或截断；
- 帮助、用户文档和测试。

### 7.3 restart 契约

一次 checkpoint 保存三类文件：原生 `.chk`、便携 `.state.xml` 和 `.restart.json`。相关原子写入、校验、XTC 截断和 reporter 恢复逻辑位于 `dropps/share/restart.py`。

关键语义：

- 原生 checkpoint 优先，失败后才回退 State；
- State 回退不是逐位可复现续跑；
- production 的 `nsteps` 是总目标，不是新增步数；
- 续跑跳过最小化、初速度生成和升温；
- append 前把比 checkpoint 更新的输出回滚，避免重复样本；
- `--no-append` 使用下一个 `.partNNNN`；
- 正常结束、`-maxh` 和 Ctrl-C 都必须保存最终 restart。

对该路径的修改至少运行 `tests/test_mdrun_config.py`、`tests/test_mdrun_runtime.py` 和 `tests/test_mdrun_resume.py`。

## 8. 轨迹分析架构

`dropps/share/trajectory.py` 的 `trajectory_class` 负责：

- 通过 `read_tpr` 恢复拓扑、坐标、质量、电荷和粒子参数；
- 建立 MDAnalysis Universe；
- 流式加载 XTC；
- 加载默认分组和可选 NDX；
- 把时间窗口转换为帧窗口；
- 执行 DROPPS 选择表达式。

选择和 NDX 实现在 `dropps/share/indexing.py`。DROPPS 内部原子索引从 0 开始，表达式支持 `mol`、`name`、`abbr`、`res`、`resid`、`chainid`、`index`、`group` 以及 `&`、`|`、括号。

分析命令设计建议：

- 大轨迹默认逐帧或分块处理，避免无必要的完整轨迹驻留内存；
- 在算法入口显式检查盒形状、链长、分组重叠、采样间隔和空选择；
- 内部明确使用 nm、ps、K、bar 和 kJ/mol，并在输出头部写单位；
- 周期性边界条件处理放在选择和测量之前，并为跨盒轨迹添加测试；
- 同一结果同时输出 XVG/CSV 时，确保数值、时间轴和元数据一致。

## 9. 测试组织

测试使用标准库 `unittest`。当前主要覆盖面如下：

| 测试 | 关注点 |
|---|---|
| `test_argument_parser.py`、`test_cli_help.py` | CLI 分区、默认值、required 和命令目录 |
| `test_dps.py` | 版本快速路径和启动横幅 |
| `test_tpr_compatibility.py` | TPR v2 round trip、旧 pickle 和 system identity |
| `test_mdrun_config.py` | 路径、平台、运行时覆盖和分段编号 |
| `test_mdrun_runtime.py` | 墙钟控制、平台选择和 checkpoint 调度 |
| `test_mdrun_resume.py` | 中断、恢复、append/rollback 和跨环境 State |
| `test_energy*.py` | EDR I/O、字段选择和 MDP 能量参数 |
| `test_stress_tensor_reporter.py` | 压力张量、reporter 和单位 |
| `test_surface_tension.py`、`test_viscosity_gk.py` | 界面张力和 Green-Kubo 黏度 |
| `test_coexistence*.py`、`test_exchange*.py`、`test_phase_msd*.py` | slab 分析及纯数值内核 |
| `test_contact_ng*.py` | 邻居搜索、布局和流式接触聚合 |
| `test_polystat*.py` | 多分子 cohort、回转张量、FFT lag、持久长度、PBC 和流式输出 |
| `test_sasa*.py` | SASA 球面采样、力场半径、周期邻居、surface/output 和 XVG |
| `test_regressions.py` | 拓扑、文件 I/O 和边界回归 |

建议测试顺序：

```bash
# 先运行被改动模块的测试
python -m unittest tests.test_energy

# 再运行完整测试套件
python -m unittest discover -s tests -p 'test_*.py'

# 最后检查命令目录是否仍由源码生成且没有意外差异
python scripts/update_readme_commands.py
python scripts/update_readme_commands.py --check
git diff --check
git diff
```

涉及 GPU 的逻辑应能通过 mock 或 CPU/Reference 小系统验证核心行为；不要让单元测试默认依赖特定 GPU。

## 10. 代码与错误处理约定

- 使用四空格缩进和 UTF-8；公开行为应有简短 docstring 或清晰的命令帮助；
- 路径优先使用 `pathlib.Path` 或 `os.fspath`，写文件时显式指定 UTF-8；
- 共享逻辑放入 `dropps/share/` 或 `dropps/fileio/`，不要在多个命令中复制；
- 纯计算函数应返回结构化结果，由 CLI 层负责打印和写文件；
- 新状态消息使用 `dropps.share.console` 的级别函数；机器可读正文保持原格式；
- 新代码优先抛出带上下文的 `ValueError`、`FileNotFoundError`、`RuntimeError` 或格式专用异常，便于测试和上层处理；
- 不要吞掉异常。若需要补充上下文，使用 `raise ... from exc` 保留根因；
- 对长循环使用 tqdm 或有节制的进度报告，避免每步打印；
- 不要依赖未排序的 glob、set 或字典遍历来决定科学输出顺序；
- 所有兼容性行为都应写入测试，不只写注释。

现有部分旧模块仍使用 `print` 加 `quit()` 的错误路径；CLI 兼容层会标准化其
显示和退出状态。新增可测试核心逻辑应优先使用异常，并由统一 CLI 边界输出。

## 11. 文档和发布检查

### 11.1 文档更新触发条件

| 变更 | 同步更新 |
|---|---|
| 新增、删除或重命名命令 | 注册表、README 自动目录、用户指南 |
| 修改 CLI 参数或默认值 | `-h`、测试、用户指南示例 |
| 修改终端输出或 debug | `test_console.py`、`test_dps.py`、用户指南约定 |
| 修改 MDP 参数 | 模板、`docs/mdp-reference.md`、测试 |
| 修改输出格式 | 文件格式实现、用户/开发文档、兼容测试 |
| 修改 TPR/restart 语义 | 迁移说明、续跑文档、round-trip/恢复测试 |

### 11.2 构建发布包

项目使用 `poetry-core` 作为 PEP 517 backend。准备发布时：

```bash
python -m pip install build
python scripts/prepare_release.py --expected-version 1.0.0
```

发布前至少检查：

1. `pyproject.toml` 中的版本和依赖约束正确；
2. `dps --version` 与目标版本一致；
3. README 命令目录已重新生成；
4. 完整测试套件通过；
5. wheel 和 sdist 中不包含 `__pycache__`、`.pyc`、`._*`、`@eaDir` 或 `dropps/backups/`；
6. 在干净虚拟环境中安装 wheel，并运行版本、帮助和一个最小工作流。

许可证由 `LICENSE` 和 `NOTICE` 和 `pyproject.toml` 中的 `Apache-2.0` 声明；分发前确认目标渠道符合项目许可。

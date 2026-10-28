# DROPPS 1.0.0 架构

## 1. 设计目标

DROPPS 是单入口、多命令的科学计算 CLI。架构首先保证模拟输入和输出可追溯，
其次把依赖 OpenMM/MDAnalysis 的编排层与可独立测试的数值核心分离。

依赖方向固定为：

```text
dps CLI
  -> command registry
     -> commands / analysis adapters
        -> share / fileio services
           -> *_core numerical kernels
              -> NumPy/SciPy and standard library
```

下层模块不得反向依赖 CLI 注册表。`dropps.commands` 和 `dropps.analysis` 的
`__init__.py` 保持轻量；所有公开命令只在
`dropps/share/all_commands.py` 注册一次。

## 2. 模块职责

| 路径 | 职责 | 不应承担的职责 |
| --- | --- | --- |
| `dropps/dps.py` | 进程入口、版本快速路径、命令分发、统一生命周期、退出码 | 科学计算和文件格式实现 |
| `dropps/commands/` | 建模、转换、运行和报告命令的参数/编排 | 可复用数值算法 |
| `dropps/analysis/` | 分析命令适配器与 `*_core.py` 数值核心 | TPR 格式私有读写 |
| `dropps/fileio/` | PDB/ITP/TPR/XVG/XPM 的读写边界 | CLI 输出和交互 |
| `dropps/share/` | 系统构建、轨迹、选择、参数、restart、reporter 等跨命令服务 | 具体命令注册 |
| `dropps/hp/` | 版本横幅和终端文案 | 运行状态或科学逻辑 |
| `scripts/` | 安装、文档同步、发布检查 | 包运行时依赖 |
| `tests/` | 单元、集成、兼容性与科学回归测试 | 生产数据存储 |

## 3. 三条核心数据流

### 建模与模拟

```text
PDB + TOP/ITP + MDP
        -> grompp -> TPR v2
        -> mdrun  -> XTC + EDR + log + restart + final PDB
```

`grompp` 冻结科学配置；`mdrun` 仅应用机器、路径、限时和本次运行覆盖。运行时
覆盖不能修改原 TPR。

### TPR 与 restart

`dropps/fileio/tpr_reader.py` 是 TPR 的唯一边界。TPR v2 以 ZIP 保存 OpenMM XML、
JSON、NumPy 数组、源输入和 SHA-256 清单，不使用 pickle。旧 pickle 仅用于可信
迁移。restart 由原生 checkpoint、便携 State XML 和校验 JSON 共同组成。

### 轨迹分析

命令适配器负责读取、选择、PBC、进度和输出；`*_core.py` 负责无 I/O 的数组
算法。共享 `time_selection` 和密度几何确保不同分析使用同一时间轴、盒定义和
周期边界语义。

### CLI 输出流

`dropps/share/argument_parser.py` 统一选项命名、Input/Output/Parameters 分区和
帮助格式；`dropps/share/console.py` 统一命令生命周期、输出级别、TTY 颜色、
stderr 路由与 debug 上下文。命令实现可以保留机器可读正文，但运行状态必须落入
`DEBUG/INFO/OK/WARNING/ERROR` 五级。全局 `--debug` 由入口提取，因此可位于
命令名之前或之后。

## 4. 扩展契约

新增命令时必须：

1. 使用共享 `ArgumentParser`，并将算法拆到可测试核心；
2. 在 `all_commands.py` 直接导入并注册一次；
3. 在 `command_help.py` 提供可解析示例和任务分组；
4. 运行 `scripts/update_readme_commands.py` 同步 README；
5. 为参数、边界、单位、元数据和失败路径添加测试。

公开选项必须有 `--lowercase-kebab-case` 长名称；可选简称只能使用一至两个
小写字母，所有 Output 简称必须由 `o` 开头。部分历史小写非标准简称仅作为隐藏
兼容别名，不得用于新示例；大写选项不再属于公开或兼容接口。

文件格式、restart 语义或科学单位变化必须同时更新兼容性测试和用户文档。

## 5. 发布边界

正式制品只包含 `dropps/` 运行代码、内置力场/模板、许可证、README，以及 sdist
中的安装脚本。`backups/`、测试、缓存、AppleDouble、Synology 元数据和历史
`dist/` 内容均不属于发布载荷。发布制品统一写入 `release/<version>/`，并附带
`SHA256SUMS`。

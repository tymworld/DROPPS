# 文章未提及但确认保留的命令

审计对象为 `11-submission/02-Proteins/manuscript.docx` 的正文、图 3 中的工作流标签，以及其 Supporting Information。`01-JCTC` 中同名稿件的正文内容与本次核对稿一致。下列 8 个辅助命令仍注册在 DROPPS 1.0 的 CLI 中，但未在上述文章中以 DROPPS 命令或图示模块出现。作者已确认全部保留，文章无需逐一介绍。

| 命令 | 当前用途 | 决定 | 理由 |
|---|---|---|---|
| `help` | CLI 命令目录与详细帮助 | 保留 | CLI 基础设施，不应按科研模块处理。 |
| `convert-tpr` | 将可信的旧 pickle TPR 转为便携、安全校验的 TPR v2 | 保留 | 支撑文章所述“本地建系、远端运行”的兼容迁移；删除会使旧数据难以安全升级。 |
| `rerun` | 对既有 XTC 重算构象观测量 | 保留 | 有科研价值，但正文没有描述。 |
| `energy` | 从 DROPPS EDR 选择观测量并导出 XVG | 保留 | 正文提到性质提取，但没有点名此命令。 |
| `exchange` | 平板界面的分子交换与相驻留时间 | 保留 | 正文没有描述。 |
| `cstat` | 对预计算接触图做残基级统计 | 保留 | 正文虽讨论接触统计结果，但没有给出该命令。 |
| `rmsd` | 支持 PBC 与拟合的 RMSD | 保留 | 正文没有描述。 |
| `pdb2bond` | 根据 TPR/TOP 向 PDB 写入 `CONECT` | 保留 | 正文没有描述。 |

## 作者决定

1. 上述 8 个命令全部保留在 DROPPS 1.0 中。
2. 它们属于 CLI 基础设施、兼容迁移或辅助科研工具，不要求在文章正文或 Supporting Information 中逐一介绍。
3. `convert-tpr` 仍只应处理可信来源的旧 TPR；新 TPR 使用 ZIP/XML/JSON/NumPy 格式及 SHA-256 完整性校验。

## 已执行的审核决定

已按作者要求删除以下 7 个命令及其专用实现、帮助和文档：`coexistence`、`phase-msd`、`contact.ng`、`timecorr`、`pbcontact`、`surftension`、`viscosity-gk`。仍被 `density`、`exchange`、`rmsd` 或 `msd` 使用的共享数值/PBC 工具未删除。

## 文章中出现、但命名需要作者统一的地方

- 正文在“构型与拓扑编辑”段写 `dps angle` 用于“引入角势”，但后文又将 `dps angle` 定义为轨迹角度分析。代码以 `dps addangle` 添加角势、以 `dps angle` 做分析。建议正文前一处改为 `dps addangle`。
- 图示中若仍使用 `add-angle`、`modify-res` 或 `make-ndx`，建议统一为实际 CLI 名称 `addangle`、`modifyres`、`make_ndx`，避免读者照图输入后失败。

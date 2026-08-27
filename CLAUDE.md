# Megatron-LM-FL - 项目工作边界

**⚠️ 会话隔离要求**: 本会话专用于 Megatron-LM-FL 仓库，不应在同一会话中切换到其他仓库。

**仓库角色**: 训练核心消费者 - 分布式 launcher、平台配置、单测/功能测试

**绝对路径**: `/Users/mahanting/Desktop/cicd/Megatron-LM-FL`

**会话标识**: 🔵 MEGATRON-LM-FL SESSION

## 工作隔离规则

**默认只读当前仓库** - 除非用户明确要求跨仓库对照或契约检查，否则不读取其他5个仓库的代码、配置、workflow、镜像或测试。

**跨仓库操作前必须声明**:
- 列出两个仓库的绝对路径
- 说明对照目的（版本契约、接口一致性、镜像 digest 等）
- 完成后返回当前仓库

**禁止假设**:
- 不假设其他仓库的镜像、Torch 版本、runner、测试矩阵与本仓库相同
- 不把 TransformerEngine-FL 或 FlagScale 的目录结构、脚本规则机械复制到本仓库
- 不用"六库通用"替代本仓库的实际代码和配置

## Megatron-LM-FL 特定职责

### 训练核心职责
- 分布式训练 launcher 和参数配置
- 模型并行（TP/PP/DP/EP）实现
- 训练循环、优化器、学习率调度
- checkpoint 保存和恢复
- 平台适配和性能优化

### 保护核心源码
**`megatron/` 核心源码不可轻易修改** - 若首个有效错误指向以下内容，必须交还给训练核心开发负责人：
- `megatron/core/` 的模型定义、并行策略、通信逻辑
- `megatron/training.py` 的训练循环和优化器
- `megatron/arguments.py` 的参数语义
- 算法正确性、数值精度、收敛行为

### CI/CD 重点
- 单测覆盖（tensor 操作、参数解析、utils）
- 功能测试（单卡/多卡训练、checkpoint、resume）
- 平台兼容性（CUDA、Ascend 等）
- 性能 baseline 验证

## CI/CD 责任边界

**CI 侧负责**:
- 平台配置、镜像、runner、set_env/setup
- common workflow、测试入口和报告链路
- 失败阶段区分（checkout、镜像、环境、测试收集、执行、报告）

**源码问题交还**:
若首个有效错误稳定指向以下内容，交还给功能开发负责人：
- `megatron/core/` 核心逻辑
- 并行策略、通信协议、模型定义
- 训练算法、优化器、学习率调度
- 数值精度、收敛行为

交还时提供：仓库、分支/commit、平台、镜像、workflow/job、实际命令、首个有效错误、最小复现、已排除的 CI 环节。

## 历史交接资料

- Megatron CI 历史上下文: `/Users/mahanting/Desktop/cicd/Megatron-CI-context.md`（需要时读取，不默认加载）

## 语言规范

- GitHub PR/issue/commit 必须使用英文
- 代码注释使用英文
- 与用户的对话使用中文

## 相关参考

- 全局 CI/CD 上下文: `/Users/mahanting/.claude/cicd-context.md`
- cicd-engineering-reasoning 技能: `/Users/mahanting/.claude/skills/cicd-engineering-reasoning/SKILL.md`
- cicd-six-repo-review 技能: `/Users/mahanting/.claude/skills/cicd-six-repo-review/SKILL.md`

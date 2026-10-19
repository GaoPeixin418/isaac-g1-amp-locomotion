# 进度看板

最后更新：2026-10-19 17:05

## 备份机制（已启用）

三层保护，防止任何一端丢失：

| 层 | 位置 | 说明 |
|---|---|---|
| 服务器 | `/data`（200G 数据盘） | 环境/代码/训练产物都在数据盘，实例重启不丢 |
| 本地 | `backup/server/<时间戳>/` | `scripts/sync_from_server.py` 每 10 分钟自动拉回日志/脚本/配置 |
| Git | 本地仓库 + **GitHub ✅** | `scripts/backup_loop.py` 自动 commit + push；仓库 <https://github.com/GaoPeixin418/isaac-g1-amp-locomotion>（分支 `main`） |

> 注：`backup/` 本地快照已移出 git（每轮一个新目录会撑爆仓库），git 只存源码/文档/脚本。

已启动的后台进程：
- `scripts/wait_install.py` —— 每 120s 查安装日志，完成/失败自动通知
- `scripts/backup_loop.py` —— 每 600s 同步 + commit（配了 remote 就 push）

## 总览

| 阶段 | 内容 | 状态 | 时间 |
|---|---|---|---|
| P0 | 技术调研：AMP 实现 / 数据集 / 版本兼容 | ✅ 完成 | 10:19-10:45 |
| P1 | 服务器开通 + 环境安装 | ✅ **完成** | 10:45-16:34（踩了 8 个坑） |
| P2 | legged_lab + rsl_rl(AMP) + G1 模型/动作数据 | ✅ **完成** | 16:28-16:57 |
| P3 | 参考动作体检 + motion viewer 目视 | ⏳ 明早评估前做 | — |
| P4 | AMP 行走训练 | 🔄 **训练中** | 17:02 开跑，ETA 22:00 |
| P5 | 走跑切换（速度指令覆盖测试） | ⏳ 待 P4 | — |
| P6 | 量化评估 + 录视频 + 技术报告 + 仓库 | ⏳ 待 P5 | — |

## 当前：P4 正式训练（4096 envs × 5000 iter）

- 任务：`LeggedLab-Isaac-AMP-G1-v0`（velocity-conditioned AMP）
- 数据：`MotionData/g1_29dof/amp/walk_and_run/`（26 个 pkl：走/跑/转身/切换）
- 实测：**3.49s/iter，GPU 80%，显存 10.9G/24G**，ETA ≈ 5h（22:00 完成）
- checkpoint：`/data/legged_lab/logs/rsl_rl/g1_amp/<时间戳>/`（rsl_rl 定期保存）
- 日志：`/data/logs/train_main.log`（本地监控进程每 90s 检查，异常自动通知）
- 首迭代指标：`Episode_Reward/style: 0.0482`（风格奖励在算）、`bad_orientation: 1.0`（初始全摔，正常）

## P1 环境安装（已完成，8 个坑全记录在开发日志）

| 子步骤 | 状态 |
|---|---|
| conda env py3.11 + torch 2.7.0+cu128 | ✅ |
| isaacsim 5.1.0（EULA `yes Yes \|` 管道） | ✅ |
| IsaacLab v2.3.2（flatdict 需 setuptools 降 80.9） | ✅ |
| rsl_rl AMP fork 3.2.0（PIP_CONSTRAINT 锁 torch） | ✅ |
| legged_lab 0.1.0 + G1 USD + 26 个 AMP pkl | ✅ |
| 训练链路验证（2 iter smoke：reward 有值、checkpoint 落盘） | ✅ |

## 风险与降级

| 风险 | 概率 | 降级方案 |
|---|---|---|
| isaacsim 20GB 下载慢/中断 | 中 | pip 缓存已落盘，重跑可续；换清华镜像/nvidia 源 |
| legged_lab 与 Lab 2.3.2 有 API 冲突 | 中 | 切保底 `humanoid_amp`（03 脚本） |
| git lfs 拉不出 G1 USD/动作 | 低（LFS 实测通） | 从 HF 镜像 `lvhaidong/LAFAN1_Retargeting_Dataset` 下载 |
| retarget 数据有关节越界/穿地 | 中 | `verify_motion.py` 体检 + 目视，必要时换 clip |
| 两天内 AMP 不收敛 | 中 | 砍到单一速度高质量行走，跑作为加分项 |

## 下一步（P1 完成后立即执行）

1. `setup/02_install_legged_lab.sh`
2. `scripts/verify_motion.py` 体检参考动作
3. 启动第一次 AMP walk 训练，**尽早开跑**，因为收敛需要时间

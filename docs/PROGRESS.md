# 进度看板

最后更新：2026-10-19 20:00

## 服务器成本与关机计划 ⏱

> 目标：把 GPU 机时花在刀刃上。**一次性把服务器上所有必需的活干完，是省钱的关键**——
> 因为重开机虽然不用重装环境（`/data` 保留），但每次开机都意味着又要为启动+等待付费。

### 今晚时间线（实际）

| 时刻 | 事件 | 机时消耗 |
|---|---|---|
| 10:45 开机 | — | — |
| 10:45-16:34 | 环境安装（8 个坑） | 5.8 h |
| 16:34-17:02 | legged_lab + 数据 + 训练体检 | 0.5 h |
| 17:02-21:10 | **正式训练** 4096 envs × 5000 iter | 4.1 h |
| 21:10-21:40 | 终版量化评估 + 多速度视频 + 产物拉回本地 | 0.5 h |
| **21:40 关机** | | **合计 ≈ 10.9 h** |

### 关机前必须完成的保全清单

| 产物 | 位置 | 状态 |
|---|---|---|
| 最终 checkpoint `model_5000.pt` | 服务器 `/data/legged_lab/logs/rsl_rl/g1_amp/2026-10-19_16-37-21/` + 本地 `backup/2026-10-19_16-37-21/` | ⏳ |
| 中期 checkpoint `model_3600.pt`（20MB，保险） | 本地 `backup/2026-10-19_16-37-21/model_3600.pt` | ✅ 已拉回 |
| 终版走跑视频 | 本地 `backup/video/` | ⏳ |
| 量化数据 `gait_model_*.npz` | 本地 `backup/eval_out/` | ✅ iter3200 版已在，终版待更新 |
| 训练日志 | 本地 `backup/server/<时间戳>/data/logs/` | ✅ 每 10 分钟自动同步 |
| 源码/文档/脚本 | **GitHub** `isaac-g1-amp-locomotion` | ✅ 自动 push |

### 关键平台规则（决定能不能放心关机）

- **关机 ≠ 释放**：`/data`（200G 数据盘，已用 37G）在关机后保留，环境/env/checkpoint/数据集全在
- **下次开机无需重装**：重开后直接 `cd /data/legged_lab` 就能跑，省掉今天那 5.8 小时
- ⚠️ **关机约 7 天实例会被自动释放** → 若一周内不再用，务必确认上面的保全清单全绿

### 更新：用户选「等跑完 5000」+「明天可能再训/调 style 权重」

**关键原则：训练跑完那一刻 GPU 就空转了，空转 = 纯烧钱。**
所以只有两条路：① 立刻关机 ② 立刻排第二轮实验。绝不让它挂着过夜等明天。

采纳方案：**今晚 21:40 关机 → 明早按需开机**。理由——重开无需重装环境
（`/data` 保留 21G env + 数据集），开机后 5 分钟就能开训，比通宵空转省得多。

#### 🔑 找到 style 权重的旋钮（明早调参用）

`/data/legged_lab/source/.../amp/config/g1/agents/rsl_rl_ppo_cfg.py`：

```python
amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
    hidden_dims=[1024, 512], activation="elu",
    style_reward_scale=2.0,      # 风格奖励的绝对缩放
    task_style_lerp=0.5,         # ← 任务 vs 风格 的插值系数，AMP 原文的核心权重
)
```

`task_style_lerp` 含义：`r_final = (1-λ)·r_task + λ·r_style`
- λ↑ → 更像人，但可能站不稳、跟不上速度指令
- λ↓ → 更稳，但退回「能走但僵硬」的老路
- **面试价值**：做 λ = 0.3 / 0.5 / 0.7 三档消融，对比 duty-factor 曲线与 CoT，
  是能讲出道理的 ablation（比单纯「我调了参数」强得多）

#### 明早开工（一条命令，脚本已就绪）

```bash
# 续训（从 model_5000 接着训到 7000，约 1.6h）
MODE=resume MAX_ITER=7000 LERP=0.5 nohup bash /data/setup/08_train_resume.sh > /data/logs/train_resume.log 2>&1 &

# 加重风格权重做对照（约 1.6h）
MODE=resume MAX_ITER=7000 LERP=0.7 nohup bash /data/setup/08_train_resume.sh > /data/logs/train_lerp07.log 2>&1 &
```

`scripts/rsl_rl/train.py` 支持 `--resume --load_run <run>`；`08_train_resume.sh`
会自动 sed 改配置并存 `.bak`，改没改对用 `git diff` 自查即可。

⚠️ 无论如何，**别让实例连续关机超过 7 天**（平台会自动释放 /data）。

### 明天是否可以全天关机？

**取决于上面的选择**：若明天只做本地工作（剪辑/报告）→ 全天关机 = 0 成本。
若要做 style 消融 → 明早开机跑 1-2 轮即可，**跑完立刻关**。

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
| P3 | 训练链路冒烟（跳过独立 motion viewer） | ✅ 完成 | 17:02 |
| P4 | AMP 走跑训练 4096 × 5000 | 🔄 **训练中** | 17:02 开跑，**ETA 21:10** |
| P5 | 走跑切换（6 速度档指令覆盖测试） | ✅ **已完成** | 19:20（iter3200 评估） |
| P6 | 量化评估 + 视频 + 技术报告 + 仓库 | 🔄 进行中 | 量化报告已出，终版待 iter5000 |

## 当前：P4 正式训练（20:00 实测快照）

- 任务：`LeggedLab-Isaac-AMP-G1-v0`（velocity-conditioned AMP）
- 数据：`MotionData/g1_29dof/amp/walk_and_run/`（**30 段**，总时长 156 s，含 walk_to_run / run_to_walk 过渡）
- 实测：**2.94 s/iter，GPU 85%，显存 11.1 G/24 G**
- 进度（19:57）：已跑 3:14:21，**ETA 1:12:50 → 约 21:10 完成**
- checkpoint 实际路径 ⚠️ 不是 `rsl_ramp`：
  `/data/legged_lab/logs/rsl_rl/g1_amp/2026-10-19_16-37-21/model_*.pt`（每 200 iter 一个，最新 `model_3600.pt`，20 MB/个）
- 日志：`/data/logs/train_main.log`

### 训练曲线趋势

| 指标 | iter 400 | iter 2000 | 判读 |
|---|---|---|---|
| 摔倒率（重心过低） | 32.3% | **1.3%** | 站稳了 |
| 跑满全程比例 | 60.2% | **95.3%** | — |
| 速度跟踪得分 | 0.145 | **0.710** | 听得懂指令 |
| 速度跟踪误差 | 2.13 m/s | **0.615 m/s** | — |
| AMP style 奖励 | 0.662 | **0.788** | 风格在学 |

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

## 下一步（训练完成后立即执行，约 30 分钟）

1. 用 `model_5000.pt` 重跑 `scripts/eval_gait.py`（6 速度档），更新 `docs/量化评估结果.md`
2. 录终版视频（建议多录一档慢走 + 一档快跑，便于剪辑对比）
3. 把 `model_5000.pt` + 视频 + npz 全部拉回本地
4. **21:40 关机**

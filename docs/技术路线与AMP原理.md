# 技术路线与 AMP 原理

> 目标：用 AMP（Adversarial Motion Priors）让宇树 G1 在 Isaac Lab 中做出**拟人态走跑**，
> 替代此前"PPO + 手写奖励"那种能走但僵硬的步态。
> 最后更新：2026-10-19 17:10

---

## 1. 为什么是 AMP：问题的本质

### 1.1 手写奖励的天花板

PPO 做双足行走的常规做法是手写一堆奖励项：

```
r = w1·(速度跟踪) + w2·(别摔倒) + w3·(关节别超限) + w4·(能耗小) + w5·(脚抬够高) + ...
```

问题在于"好看的步态"是个**高维、难以逐条列举**的性质。你能想到的项（步频、步长、抬脚高度）
只是冰山一角，真正决定观感的是整条轨迹的**分布特征**——手臂摆动与腿的相位关系、
落地时重心的起伏节奏、躯干的微小扭转。这些既列不全，权重也无从推导，
只能靠实验试错。结果是收敛到一个"每一项都勉强及格、整体很机械"的局部最优。

### 1.2 AMP 的思路转变

**不定义什么叫好看，改为让判别器从数据里学。**

给一批真人动捕数据（LAFAN1，已 retarget 到 G1 骨架），训一个小网络 D 去区分
"这段关节运动来自真人还是来自机器人"。策略的奖励里加一项：

```
r_style = -log(1 - D(s_t, s_{t+1}))
```

判别器越觉得像真人，策略拿到的分越高。判别器自己在训练中不断变强，
策略就得越来越像才能骗过它——这就是对抗。

**关键差异**：手写奖励只能惩罚你**想到了**的特征；判别器惩罚的是**数据里真实存在**的
全部统计特征。所以它天然能覆盖你列不全的那些细节。

---

## 2. 判别器在学什么（数学）

### 2.1 输入不是单帧，是 transition

判别器看的是 `(s_t, s_{t+1})` 连续两帧，而不是单帧姿态。原因很直接：
单帧只能判断"这个姿势是否人类能做到"，两帧才能判断"这个**运动**是否人类风格"。
一个僵硬的机器人可以摆出和人类一样的静态姿势，但从一个姿势到下一个姿势的**变化方式**会露馅。

输入特征（legged_lab 实现）是关节角、关节速度、根部线速度/角速度等运动学量，
**不含世界坐标系的绝对位置**——这样学到的才是风格本身，而不是"在场地某个角落走路"。

### 2.2 训练目标：最小二乘 GAN

AMP 原文（Peng et al., 2021）用的是 LSGAN 形式，比原始 GAN 的梯度更稳定：

```
判别器损失：
  L_D = E_参考[(D(s,s') - 1)²] + E_策略[(D(s,s') + 1)²] + w_gp · E_参考[‖∇D‖²]

风格奖励：
  r_style = -log(1 - D(s,s'))     （D→1 时奖励→+∞，D→0 时奖励→0）
```

第三项是**梯度惩罚**（gradient penalty），防止判别器在参考样本附近变得陡峭，
这是 GAN 训练里对抗模式崩塌的标准手段。

### 2.3 两个网络的更新节奏

- 策略（actor-critic）用 PPO 更新，采样来自 4096 个并行环境
- 判别器用 Adam 独立更新，正负样本分别来自**参考动作库**和**当前策略的 rollout buffer**
- 两者交替；判别器通常需要每轮多更新几次才能跟上策略的变化

---

## 3. AMP vs DeepMimic（面试高频追问）

| 维度 | DeepMimic | AMP（本项目） |
|---|---|---|
| 参考数据用法 | 逐帧目标姿态，需**相位变量** phase 对齐 | 只做分布匹配，**不需要 phase** |
| 奖励 | 显式姿态跟踪误差 ‖q − q̂‖² | 判别器输出的风格分 |
| 多动作切换 | 一条 clip 一个策略；切换需额外门控 | 多 clip 共用一个判别器，按速度指令自动选风格 |
| 保真度 | 高（几乎复刻参考动作） | 略低（只匹配分布，不逐帧对齐） |
| 泛化性 | 弱 | 强——这是我们选它的原因 |

**为什么不需要 phase 很重要**：phase（"现在走到周期的第几拍"）是 DeepMimic 类方法的
硬前提，一旦要做走↔跑切换、变速，phase 的对齐和插值就变成一团麻烦事。
AMP 靠"速度指令 + 判别器"的组合，让策略自己学会"要跑快点时该摆成什么样"。

本项目用的 `LeggedLab-Isaac-AMP-G1-v0` 奖励里**同时**包含：
- `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`——速度指令跟踪（任务）
- `style`——判别器风格分（风格）

所以它能按命令切换走/跑，这是纯模仿方案做不到的。

---

## 4. 数据链路

```
LAFAN1（真人动捕 BVH）
    ↓  retargeting：人体骨架 → G1 29 DOF 骨架
G1 关节角序列（.pkl）
    ↓  打包
/data/legged_lab/MotionData/g1_29dof/amp/walk_and_run/   ← 26 段
    ↓
训练时两条用途：
  ① 判别器随机采样连续两帧 → 正样本
  ② episode 起始状态从参考动作中抽取 → 加速早期探索（reference state initialization）
```

**本项目跳过了 retarget 环节**：`legged_lab` 仓库用 `git lfs pull` 自带了宇树 G1 的
USD 模型（29 DOF，103 MB）和已 retarget 好的动作数据。这省掉了全流程里最容易
卡两天的步骤，也是 2 天能出结果的决定性因素之一。

参考动作构成（26 段 pkl）：走、跑、转身、侧步、走跑切换。

---

## 5. 技术选型（决策表）

| 决策点 | 选择 | 备选 | 为什么 |
|---|---|---|---|
| 框架 | `legged_lab` | `humanoid_amp`（保底） | legged_lab 的 AMP **带速度指令**，能按命令切换走/跑；自带 G1 USD + 已 retarget 动作 |
| 机器人 | 宇树 G1（29 DOF） | H2 | GMR 等 retarget 工具目前不支持 H2；G1 生态最全，2 天内出结果概率最高 |
| 参考数据 | LAFAN1 retarget 版 | AMASS + GMR 自处理 | 省掉 retarget 环节；AMASS 作为后续增强备选 |
| RL 后端 | rsl_rl（AMP fork） | skrl | legged_lab 主线就用 rsl_rl |
| Sim 版本 | Isaac Sim **5.1.0** | 4.5 | IsaacLab v2.3.2 硬要求 Sim 5.x；5.x wheel 只有 cp311 |
| Lab 版本 | IsaacLab **v2.3.2** | — | legged_lab 锁定版本 |

**版本锁**（改任何一个都会连锁崩）：
```
Python 3.11.17  →  torch 2.7.0+cu128  →  isaacsim 5.1.0  →  IsaacLab v2.3.2
```
其中 torch 版本是硬约束：isaacsim 5.1 按 torch 2.7 编译，
pip 解析依赖时曾试图拉 torch 2.14.1 + CUDA 13 覆盖它（已用 `PIP_CONSTRAINT` 锁死）。

---

## 6. 当前训练配置（实测）

| 项 | 值 |
|---|---|
| 任务 ID | `LeggedLab-Isaac-AMP-G1-v0` |
| 并行环境 | 4096 |
| 总迭代 | 5000 |
| 实测速度 | 2.96 s/iteration |
| GPU | RTX 3090 24G，占用 11.0 G / 24 G，利用率 58% |
| 启动时间 | 2026-10-19 16:37 |
| 预计完成 | 约 21:00（ETA 3:55 @ 17:02） |

### 6.1 训练曲线实测（iter ≈ 400）

| 指标 | 数值 | 读法 |
|---|---|---|
| `Episode_Reward/style` | 0.048 → **0.662** | 风格奖励在涨，判别器给出的"像人"评分提高 ✅ |
| `Episode_Reward/track_lin_vel_xy_exp` | 0.145 | 速度跟踪还弱，满分 1.0 |
| `Episode_Reward/track_ang_vel_z_exp` | 0.183 | 转向跟踪同样偏弱 |
| `Metrics/base_velocity/error_vel_xy` | 2.13 m/s | 速度跟踪误差仍大 |
| `Episode_Termination/time_out` | 0.602 | 60% 的 episode 能跑满时长 ✅ |
| `Episode_Termination/base_height` | 0.323 | ⚠️ 32% 因重心过低终止——还没站稳 |
| `Episode_Termination/bad_orientation` | 0.093 | 9% 姿态崩掉 |

**诚实判读**：现在才 400/5000 迭代（8%），`style` 从 0.048 涨到 0.662 说明
对抗训练机制**确实在工作**，但摔倒率 41%（32% + 9%）说明步态还没稳定。
这是早期正常状态，关键看 1500-2000 迭代后 `base_height` 终止率能否降下来。

---

## 7. 审核清单（你可以逐项核验）

```bash
# 服务器（root@jq1.9gpu.com:12250）

# ① 训练是否真在跑
tail -30 /data/logs/train_main.log
nvidia-smi

# ② checkpoint 是否真的落盘（每 200 iter 一个）
ls -la /data/legged_lab/logs/rsl_rl/g1_amp/2026-10-19_16-37-21/

# ③ 参考动作数据是否完整
ls /data/legged_lab/MotionData/g1_29dof/amp/walk_and_run/ | wc -l   # 应为 26

# ④ G1 模型是否正确（29 DOF）
ls -la /data/legged_lab/        # g1_29dof 相关 usd，约 103 MB

# ⑤ 版本是否锁对
/data/envs/isaaclab/bin/python -c "import torch; print(torch.__version__)"   # 2.7.0+cu128
```

本地侧：
- `docs/PROGRESS.md`——阶段进度看板
- `docs/开发日志.md`——按时间记录，8 个踩坑按"现象→排查→真凶→规律"写
- `backup/server/<时间戳>/`——每 10 分钟从服务器拉回的日志快照
- GitHub：<https://github.com/GaoPeixin418/isaac-g1-amp-locomotion>

---

## 8. 诚实红线

- **本项目全部在仿真中完成，未上真机。** 报告和面试表述中不得出现 sim2real 相关结论。
- 参考动作 retarget 的质量未经逐帧目视验证（时间所限，依赖 legged_lab 官方数据）。
  若最终步态异常，retarget 数据是首要怀疑对象。
- `style` 奖励高 ≠ 步态一定好看。最终判断必须结合**录制的视频**，
  以及步频/步长/关节角 RMSE 等量化指标（明日完成）。

---

## 9. 待完成

- [ ] 训练跑满 5000 iter（约 21:00）
- [ ] Play 模式加载 checkpoint 录视频
- [ ] 速度指令覆盖测试：0.5 / 1.0 / 1.5 / 2.5 m/s 观察走→跑切换
- [ ] 量化评估：步频、步长、足部抬升、关节角 RMSE、CoT 能耗
- [ ] 判别器准确率曲线（证明风格是学到的，不是调出来的）
- [ ] 一页技术报告 + README 整理

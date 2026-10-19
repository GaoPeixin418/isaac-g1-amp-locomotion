# G1 + AMP 拟人态走跑

> 用 Adversarial Motion Priors (AMP) 在宇树 G1 上做拟人态走/跑控制，参考动作用 LAFAN1 数据集。
> Is**纯仿真项目**（Isaac Sim + Isaac Lab），**未做 sim2real 部署**，任何对外表述都不声称真机验证。

## 1. 为什么做这个

面试双足机器人运动控制岗时，面试官看了我基于 PPO 的人形机器人行走 demo，评价是"有点丑"。
根因：纯 PPO 从零探索出的步态只有"能走"，没有人类的步态风格（步频、摆臂、重心起伏、足部轨迹）。

AMP 的做法：不逐帧跟踪参考动作，而是训练一个 **discriminator** 去区分"生成的动作片段"和"参考动捕片段"，
把它的判别输出作为**风格奖励**叠加到任务奖励上。好处是不需要 phase 变量、不需要严格对齐时间轴，
还能保留按速度指令自由切换走/跑的能力。

## 2. 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| 机器人 | Unitree G1（29 DOF） | 开源 retarget 数据、USD 模型、AMP 实现最全；H2 生态不成熟 |
| 算法 | AMP（PPO + discriminator 风格奖励） | 面试官点名；且支持 velocity command 走跑切换 |
| 参考数据 | LAFAN1（宇树官方已 retarget 到 G1 的 CSV） | 跳过"人形动捕→G1 骨架"这个最耗时环节 |
| 框架 | `legged_lab`（主线）+ `humanoid_amp`（保底） | legged_lab 的 AMP 带速度指令；humanoid_amp 是纯风格模仿 |
| 仿真 | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 | legged_lab v2.3.x 的硬要求；Sim 5.x 必须 Python 3.11 |

## 3. 环境（云端服务器）

| 项 | 值 |
|---|---|
| 机器 | 9gpu 租用实例，RTX 3090 24GB / Xeon 8350C 12 核 / 32GB 内存 |
| 系统 | Ubuntu 22.04.4 |
| 磁盘 | 系统盘 59G（紧张）+ **数据盘 200G → `/data`（所有环境都装这）** |
| conda env | `/data/envs/isaaclab`（Python 3.11.17） |
| 网络 | GitHub / GitHub LFS / HuggingFace / pypi.nvidia.com 实测全通 |

## 4. 快速复现

```bash
# 1) 环境（约 60-90 min，后台跑）
bash setup/01_install_env.sh

# 2) legged_lab + rsl_rl(AMP fork) + G1 模型与参考动作
bash setup/02_install_legged_lab.sh

# 3) 参考动作体检（关节限位/骨盆高度/速度区间/突变帧）
python scripts/verify_motion.py --dir ./motions/lafan1

# 4) 训练 AMP 行走
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 --headless --num_envs 4096

# 5) 评估 + 录视频
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Flat-G1-Play-v0 \
    --headless --num_envs 64 --video --checkpoint <path>
```

## 5. 目录结构

```
g1-amp-locomotion/
├── setup/      安装脚本（01 环境 / 02 legged_lab / 03 humanoid_amp 保底）
├── data/       数据集下载
├── scripts/    远程操作(remote.py)、数据体检、进度监控、备份
├── config/     训练配置（环境就绪后落成文件并纳入 git）
├── docs/       进度看板、开发日志、踩坑记录
├── backup/     从服务器拉回的资料（日志/脚本/配置）
└── report/     一页技术报告 + 面试问答准备
```

## 6. 文档索引

- `docs/PROGRESS.md` —— 进度看板（当前阶段 / 阻塞 / 下一步）
- `docs/开发日志.md` —— 按时间追加的开发记录与踩坑（现象→排查→真凶→规律）

## 7. 诚实红线

- 只做仿真，不声称 sim2real
- 量化指标必须来自实测（步频/步长/关节 RMSE/CoT），不编造
- 未跑完的实验在报告里标"未验证"，不写成已完成

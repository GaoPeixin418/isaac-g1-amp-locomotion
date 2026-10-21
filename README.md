# G1 + AMP 拟人态走跑

> 用 Adversarial Motion Priors (AMP) 在宇树 G1 上做拟人态走/跑控制，参考动作用 LAFAN1 数据集。
> **纯仿真项目**（Isaac Sim + Isaac Lab），**未做 sim2real 部署**，任何对外表述都不声称真机验证。
> 全部结论来自 6 个速度档的确定性 rollout，数据与分析脚本都在仓库里，可复现。

## 1. 一句话结论

在 Unitree G1 上，只有把风格奖励与任务奖励按 **λ=0.5** 绑定，才能同时拿到
"像人""能听指令""省电"三件事。三个极端对照各缺一块，且每种缺失都能落到机制上解释：

| 配置 | 步态形状 | 能听指令吗 | 能耗 CoT | 一句话 |
|---|---|---|---|---|
| λ=1.0 纯 PPO | ❌ 高频小碎步，duty 秩相关 **+0.60** | ✅ 低速准，**2.0 m/s 以上崩溃** | ❌ **2.75** | 能走，但不像人、且费劲 |
| λ=0 纯风格 | ❌ duty 0.9，脚不离地 | ❌ 30 s 只挪 0.1–0.2 m | 1.20 | 像人，但**根本不走** |
| DeepMimic | ✅ **−1.00**（比 AMP 还单调） | ❌ **无速度指令接口**，速度由 clip 锁死 | ⚠️ 2.86（低速） | 会走得像人，但**叫不动** |
| **λ=0.5 AMP** | ✅ **−0.94** | ✅ 全程 0.5–3.0 m/s | ✅ **0.62** | 三项都占 |

最终交付模型：**amp_4000**（checkpoint 阶梯 1000/2000/3000/4000/4999 选出）。

> 三条必须同时说的注解，详见 `docs/技术报告_一页版.md` §2、§5：
> **① λ=0.5 是 legged_lab 的默认值，不是调出来的** —— 只测了 0 / 0.5 / 1.0 三档，
> 未扫描 0.3–0.7，因此不声称 0.5 最优。λ 是手设超参，对抗学习不参与它的确定。
> **② DeepMimic 能走**（关节跟踪误差仅 6.10°），缺的是"速度指令接口"而非速度；
> 实测只达参考速度 48%–77%，但那是**默认奖励权重下**的观测（姿态 0.50 / 根位置 0.15），
> 未做权重扫描，属**软结论**，不能算方法的固有缺陷。
> **③ 真正站得住的是源码级结构差异**：DeepMimic 观测组无 command 项 → 不能响应外部输入。

---

## 2. 为什么做这个

面试双足机器人运动控制岗时，面试官看了我基于 PPO 的人形机器人行走 demo，评价是"有点丑"。
根因：纯 PPO 从零探索出的步态只有"能走"，没有人类的步态风格（步频、摆臂、重心起伏、足部轨迹）。

我把"丑"翻译成了可测的数字 —— PPO 是 **1.5–2.1 Hz 步频 + 0.27–0.74 m 步长**，
人类同速度下是 0.85–1.0 Hz + 0.6–1.5 m，即**步频 2.2 倍、步长 48%**，能耗还是 AMP 的 4.4 倍。

AMP 的做法：不逐帧跟踪参考动作，而是训练一个 **discriminator** 去区分"生成的动作片段"和"参考动捕片段"，
把它的判别输出作为**风格奖励**叠加到任务奖励上。好处是不需要 phase 变量、不需要严格对齐时间轴，
还能保留按速度指令自由切换走/跑的能力。

```python
# 源码 rsl_rl/modules/amp.py:203
return λ * task_reward + (1 - λ) * style_reward   # λ 是「任务」权重，不是风格权重
```

## 3. 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| 机器人 | Unitree G1（29 DOF） | 宇树官方已把 LAFAN1 retarget 到 G1 骨架，跳过最耗时的 retarget 环节 |
| 算法 | AMP（PPO + discriminator 风格奖励） | 支持 velocity command，可做走→跑连续切换 |
| 参考数据 | LAFAN1（30 段 walk_and_run，总时长 156 s，含走跑过渡） | 官方 retarget 版，含过渡段才能训出连续步态族 |
| 对照方法 | DeepMimic（逐帧跟踪） | 用来回答"为什么是 AMP 而不是另一条路" |
| 框架 | `legged_lab` + `rsl_rl`（AMP fork） | AMP 任务带速度指令；DeepMimic 任务是框架原生 |
| 仿真 | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 | legged_lab 的硬要求；Sim 5.x 必须 Python 3.11 |

## 4. 环境（云端服务器）

| 项 | 值 |
|---|---|
| 机器 | 9gpu 租用实例，RTX 3090 24GB → 后期换 **RTX 4090**；Xeon 8350C 12 核 / 32GB 内存 |
| 系统 | Ubuntu 22.04.4 |
| 磁盘 | 系统盘 59G（紧张）+ **数据盘 200G → `/data`（所有环境都装这）** |
| conda env | `/data/envs/isaaclab`（Python 3.11.17，torch 2.7+cu128） |
| 网络 | GitHub / GitHub LFS / HuggingFace / pypi.nvidia.com 实测全通 |

## 5. 快速复现

```bash
# 1) 环境（约 60-90 min，后台跑）
bash setup/01_install_env.sh

# 2) legged_lab + rsl_rl(AMP fork) + G1 模型与参考动作
bash setup/02_install_legged_lab.sh

# 3) 参考动作体检（关节限位/骨盆高度/速度区间/突变帧）
python scripts/verify_motion.py --dir ./motions/lafan1

# 4) 训练 AMP（4096 envs × 5000 iter，约 2.9 s/iter）
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 4096

# 5) 6 速度档评估（三模型并排）
bash setup/11_run_all_evals.sh

# 6) 录终版视频
bash setup/14_single_video.sh
```

对照实验脚本：`setup/13_finish_ppo.sh`（λ=1.0 PPO 基线）、
`setup/16_train_deepmimic.sh` + `setup/18_eval_dm.sh`（DeepMimic 同源对照）。

## 6. 目录结构

```
g1-amp-locomotion/
├── setup/          服务器安装与训练/评估/录像脚本（01–19）
├── scripts/        远程操作、数据体检、进度监控、步态量化分析（本地纯 numpy，不需 Isaac Sim）
├── docs/           全部过程文档（11 份）+ 一页技术报告 + 面试 Q&A
├── data/           数据集下载脚本
├── backup/
│   ├── videos/     7 段演示视频（已入库）
│   ├── eval_out/   9 份 npz + 4 份 csv（全部量化原始数据，已入库）
│   └── server/     服务器快照（体积大，.gitignore 排除）
└── tmp/            临时命令输出（.gitignore 排除）
```

### 演示视频

在 GitHub 上点链接即可用页面自带播放器查看（仓库内 mp4 支持在线播放）：

| 视频 | 内容 |
|---|---|
| [**▶ 终版 λ=0.5（最终交付）**](backup/videos/amp_model_4000.mp4) | 6 个速度档并排，0.5 → 3.0 m/s |
| [**▶ 终版单机视角**](backup/videos/amp_4000_single.mp4) | 单个机器人，便于看步态细节 |
| [**▶ λ=1.0 纯 PPO 对照**](backup/videos/ppo_baseline.mp4) | 高频小碎步，2.0 m/s 以上崩溃 |
| [**▶ λ=0 纯风格对照**](backup/videos/style_only_lerp0.mp4) | 判别器给 0.98 分，但原地摆姿势 |
| [**▶ DeepMimic 对照**](backup/videos/deepmimic_debug.mp4) | 形状对，位置慢慢掉队 |
| [checkpoint 对照 amp_model_2000](backup/videos/amp_model_2000.mp4) | 跟踪最好、拟人最差 |
| [PPO 单机视角](backup/videos/ppo_baseline_single.mp4) | 便于看清小碎步 |

## 7. 文档索引

**对外交付（先看这两份）**
- [`docs/技术报告_一页版.md`](docs/%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A_%E4%B8%80%E9%A1%B5%E7%89%88.md) —— 一页讲完：方法、实验、结果、选型、边界
- [`docs/面试Q&A.md`](docs/%E9%9D%A2%E8%AF%95Q%26A.md) —— 20+ 道题的回答要点与追问预案，含 30/60 秒讲稿

**实验与推导**
- [`docs/步态量化对比与模型选型.md`](docs/%E6%AD%A5%E6%80%81%E9%87%8F%E5%8C%96%E5%AF%B9%E6%AF%94%E4%B8%8E%E6%A8%A1%E5%9E%8B%E9%80%89%E5%9E%8B.md) —— 五档 checkpoint 阶梯、三条趋势线、选型论证
- [`docs/AMP_vs_PPO对照实验.md`](docs/AMP_vs_PPO%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C.md) —— λ 三档消融，回答"AMP 比 PPO 好在哪"
- [`docs/AMP_vs_DeepMimic对照实验.md`](docs/AMP_vs_DeepMimic%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C.md) —— 同源对照 + 源码级论证
- [`docs/公平对比方法论与遗留问题.md`](docs/%E5%85%AC%E5%B9%B3%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%95%E8%AE%BA%E4%B8%8E%E9%81%97%E7%95%99%E9%97%AE%E9%A2%98.md) —— 能力边界矩阵、速度匹配对照、"丑"的可证伪化、遗留问题清单
- [`docs/训练曲线分析.md`](docs/%E8%AE%AD%E7%BB%83%E6%9B%B2%E7%BA%BF%E5%88%86%E6%9E%90.md) —— 训练过程的判读与一次结论更正
- [`docs/量化评估结果.md`](docs/%E9%87%8F%E5%8C%96%E8%AF%84%E4%BC%B0%E7%BB%93%E6%9E%9C.md) —— 早期评估结果

**过程记录**
- [`docs/PROGRESS.md`](docs/PROGRESS.md) —— 进度看板
- [`docs/开发日志.md`](docs/%E5%BC%80%E5%8F%91%E6%97%A5%E5%BF%97.md) —— 按时间追加的踩坑记录（现象→排查→真凶→规律）
- [`docs/决策清单与专业判断.md`](docs/%E5%86%B3%E7%AD%96%E6%B8%85%E5%8D%95%E4%B8%8E%E4%B8%93%E4%B8%9A%E5%88%A4%E6%96%AD.md) —— 哪些是自己的判断、AI 帮做了什么
- [`docs/技术路线与AMP原理.md`](docs/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF%E4%B8%8EAMP%E5%8E%9F%E7%90%86.md) —— 原理推导
- [`docs/今日工作报告_2026-10-19.md`](docs/%E4%BB%8A%E6%97%A5%E5%B7%A5%E4%BD%9C%E6%8A%A5%E5%91%8A_2026-10-19.md)、[`docs/建仓库与Git备份.md`](docs/%E5%BB%BA%E4%BB%93%E5%BA%93%E4%B8%8EGit%E5%A4%87%E4%BB%BD.md)

## 8. 诚实红线

1. **只做仿真，不声称 sim2real**，不涉及任何真机验证。
2. **AMP 核心代码不是我写的**：用的是 legged_lab + rsl_rl 的 AMP 实现，
   判别器/AMP 缓冲/奖励混合一行未动。我做的是对齐实验条件、设计对照、
   写量化评估脚本、对结果做机制归因。
3. **数字是单点观测**：单次评估、无多 seed 重复，差异不显著时只说"综合最优"，不说"显著优于"。
4. **训练步数不完全对等**（PPO 2500 / DeepMimic 3000 / AMP 4000–5000），主动承认，不做过度归因。
5. **人类参照值是典型量级参照**，不是精确引用值；秩相关、速度误差、CoT 这三个不依赖外部文献。
6. **保留错误与更正**：λ 语义写反、步长公式多乘 2，均在文档中保留原文 + 更正块，不偷偷改掉。
7. **λ=0.5 是 legged_lab 的默认值，不是调出来的**：只测了 0 / 0.5 / 1.0 三档，
   未扫描 0.3–0.7，不声称最优。λ 是手设超参，对抗学习不参与它的确定
   （对抗学习管"风格信号从哪来"，不管"两个信号各占多少"）。
8. **"任意速度"目前是推论，不是实测**：只测了 6 个离散档，
   未测中间值、未做过连续斜坡指令。

### 需要知道的两个结论强度分档

| 结论 | 强度 | 依据 |
|---|---|---|
| DeepMimic 观测组无速度指令 → 不能响应外部输入 | **硬** | 源码级确定，不受调参影响 |
| DeepMimic 只跑到参考速度 48%–77% | **软** | 默认奖励权重下的观测，权重未扫描 |
| λ=0 会被 exploit（风格分 0.98 却原地不动） | **硬** | 实测 + 判别器输入设计可解释 |
| λ=0.5 优于两个极端 | **硬** | 实测 |
| λ=0.5 是最优值 | **无证据** | 未扫描中间值 |

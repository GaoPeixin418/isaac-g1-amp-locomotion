# 进度看板

最后更新：2026-10-19 11:15

## 总览

| 阶段 | 内容 | 状态 | 时间 |
|---|---|---|---|
| P0 | 技术调研：AMP 实现 / 数据集 / 版本兼容 | ✅ 完成 | 10:19-10:45 |
| P1 | 服务器开通 + 环境安装 | 🔄 **进行中** | 10:45- |
| P2 | legged_lab + rsl_rl(AMP) + G1 模型/动作数据 | ⏳ 待 P1 | — |
| P3 | 参考动作体检 + motion viewer 目视 | ⏳ 待 P2 | — |
| P4 | AMP 行走训练（首个可看步态） | ⏳ 待 P3 | — |
| P5 | 走跑切换（多 clip discriminator + 速度指令） | ⏳ 待 P4 | — |
| P6 | 量化评估 + 录视频 + 技术报告 + 仓库 | ⏳ 待 P5 | — |

## 当前：P1 环境安装

| 子步骤 | 状态 | 说明 |
|---|---|---|
| 服务器开通 | ✅ | 3090 24G / Ubuntu 22.04 / 数据盘 200G→/data |
| 网络预检 | ✅ | GitHub、LFS、HF、pypi.nvidia.com 全通 |
| conda env (py3.11) | ✅ | `/data/envs/isaaclab`，Python 3.11.17 |
| torch 2.7.0+cu128 | 🔄 排队中 | |
| isaacsim 5.1.0 | 🔄 排队中 | ~20GB，最慢一步 |
| IsaacLab v2.3.2 | ⏳ | |
| headless 冒烟测试 | ⏳ | |

日志位置（服务器）：`/data/logs/install.log`
本地备份：`backup/server/<时间戳>/data/logs/install.log`

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

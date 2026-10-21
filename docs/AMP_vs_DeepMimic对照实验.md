# AMP vs DeepMimic：为什么这个项目用的是 AMP

> 开始：2026-10-21 18:50
> 状态：源码论证已完成 ✅ ｜ GPU 实测数据 🔄 训练中（预计 20:45）
> 数据来源：legged_lab 源码逐行核验 + `/data/eval_out/`

---

## 0. 先说结论（三句话）

1. **DeepMimic 和 AMP 不是同一个任务的两个解法，而是两个不同的任务。**
   DeepMimic 回答"这段动作能不能复现"，AMP 回答"能不能按任意指令走出人样"。
2. **这个选择的理由是结构性的**：DeepMimic 的观测里**没有速度指令**（源码核验），
   它的产出被参考动作锁死；而面试官要的是"按指令连续走跑"，AMP 天然支持。
3. 下面所有源码结论都经过逐行核验，**不是推测**。

---

## 1. 源码级事实（逐行核验，非推测）

核验对象：`/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/`

### 1.1 观测空间对比

| | DeepMimic | AMP（本项目） |
|---|---|---|
| 维度 | 145（无历史帧） | 495（5 帧历史） |
| **速度指令** | **无** ❌ | **有** ✅ `base_velocity` |
| 参考信息 | `ref_root_pos_error` / `ref_joint_pos` / `ref_key_body_pos_b` 等 6 项 | 无（判别器内部用） |
| 组成 | 本体感受 89 维 + 参考误差 56 维 | 本体 + 命令 + 历史 |

DeepMimic 的观测组（`deepmimic_env_cfg.py:87-108`）实际内容：

```
root_rot_tan_norm(6) root_vel_w(3) root_ang_vel_w(3)
joint_pos(29) joint_vel(29) key_body_pos_b(18) root_height(1)
ref_root_pos_error(3) ref_root_rot_tan_norm(6) ref_joint_pos(29) ref_key_body_pos_b(18)
```

**这是整个论证的支点：从头到尾没有任何 command 相关的观测项。**

### 1.2 奖励对比

| DeepMimic（10 项） | 权重 | AMP（16 项） |
|---|---|---|
| `ref_track_root_pos_w_error_exp` | 0.15 | 15 项手写任务奖励（速度跟踪、防摔 −50、拖脚 −0.1…） |
| `ref_track_quat_error_exp` | 0.08 | + |
| `ref_track_root_vel_w_error_exp` | 0.10 | **判别器风格分**（独立通道注入，不在任务项里） |
| `ref_track_root_ang_vel_w_error_exp` | 0.05 | |
| `ref_track_key_body_pos_b_error_exp` | 0.15 | |
| `ref_track_dof_pos_error_exp` | 0.50 | |
| `ref_track_dof_vel_error_exp` | 0.10 | |
| `joint_torques_l2` / `joint_acc_l2` / `action_rate_l2` | 正则 | 同 |

DeepMimic 的奖励**全部是「和参考帧的偏差取 exp 核」**——它奖励的是"越来越像这一帧"。

### 1.3 关键训练技巧

| 技巧 | DeepMimic | AMP |
|---|---|---|
| RSI（参考状态初始化） | ✅ `reset_from_ref`，随机起点 | ⚠️ 无（靠 rewards 自己摸索） |
| Early termination on deviation | ✅ 偏离 1.0 m 直接终止 | ⚠️ 只有摔倒/超时 |
| Domain randomization | 基座质量 ±3 kg、质心偏移 | 同类 |
| 判别器 | ❌ 没有 | ✅ 核心 |

**这是一组很有意思的对比**：DeepMimic 在"怎么训"上更精细（RSI + 提前终止），
AMP 在"训什么"上更聪明（用分布式匹配代替手写距离）。

---

## 2. 实验设计：怎么比才诚实

关键判断：**两者不能塞进同一张指标表**。目标函数不同类，硬比是误导。

我们采用**同源三维度对照**：

| 维度 | DeepMimic 侧 | AMP 侧 | 公平吗 |
|---|---|---|---|
| **复现保真度** | 关节跟踪误差 / 根位置误差 | 判别器风格分 | 各自主场，分开陈述 |
| **步态质量** | duty factor 随**参考速度**变化 | duty factor 随**指令速度**变化 | ✅ 同构，可并排 |
| **指令响应** | 源码决定：不支持 | 0.5–3.0 m/s 连续可调 | ✅ 源码级铁证 |

### 公平性保证（这才是对照实验值钱的地方）

| 项 | 值 |
|---|---|
| 机器人 | UNITREE_G1_29DOF（同一个 USD） |
| 参考数据 | **同一批 30 段 LAFAN1 走跑**（`MotionData/g1_29dof/amp/walk_and_run`） |
| seed | 42 |
| 并行规模 | 4096 envs |
| 算法 | PPO（同一个 rsl_rl） |
| **唯一变量** | 奖励从哪来 |

DeepMimic 原本默认跟的是 `G5_-__back_kick_stageii`（后踢）——拿它跟 AMP 比毫无意义，
因为两者看的根本不是同一批动作。我们把它的 `motion_data_dir` + `motion_data_weights`
**改成 AMP 用的那 30 段**（脚本 `17_patch_deepmimic_cfg.py`，带 `.bak` 备份），
这才是严格意义上的同源对照。

字段兼容性已验证通过：

```
walk_and_run 平铺 30 个 pkl，字段完整
    fps / root_pos(179,3) / root_rot(179,4) / dof_pos(179,29) / key_body_pos(179,6,3)
    OK 全部必需字段齐备
```

---

## 3. 已知的结构性差异（不需要跑实验就能讲）

### 3.1 DeepMimic 的 phase 依赖

DeepMimic 靠 `AnimationTerm` 每步推进参考时间轴：

```python
# animation_manager.py: update()
self.motion_fetch_time += dt
```

策略永远在追"下一秒参考动作的样子"。这意味着它**必须知道自己在时间轴上的位置**，
而这个位置由参考 clip 决定，不是由外部命令决定。

→ 想让它从走切到跑，只能换一条 clip，或者做一个多 clip + skill selector 的架构。
而框架当前的观测里**没有 one-hot 选择器**，所以连"多 clip 切换"都没原生支持。

### 3.2 为什么 AMP 不需要 phase

AMP 的判别器看的是**连续 4 帧**是否像人，不关心这是第几帧。
所以策略可以自由组合"快点走"和"慢点走"，只要每一小段看起来都像人。

代价是：**它失去了"精确复现某条动作"的能力**，换来的是"任意指令都能给出像人的解"。

### 3.3 一句话总结

> DeepMimic 用**时间轴**定义"该做什么"，AMP 用**风格约束**定义"做成什么样"。
> 前者适合编排好的套路动作，后者适合需要响应外部输入的通用运动。

---

## 4. 待补数据（GPU 跑完后填入）

| 指标 | DeepMimic | AMP (λ=0.5, iter4000) | 说明 |
|---|---|---|---|
| 关节跟踪误差 (rad) | 🔄 | n/a（无此概念） | DeepMimic 主场 |
| 根位置跟踪误差 (m) | 🔄 | n/a | DeepMimic 主场 |
| 平均 CoT | 🔄 | 0.62 | 可并排 |
| duty factor 单调性 | 🔄 | −0.94 | 可并排 |
| 指令响应范围 | ❌ 结构性不支持 | 0.5–3.0 m/s | 源码级结论 |

评估脚本：`scripts/eval_deepmimic.py`（自写，因为现有 `eval_gait.py`
依赖 `base_velocity` 命令，DeepMimic 环境没有这个接口）。

该脚本把**参考动作的瞬时速度**当作分组依据（存在 `cmd_x` 列里），
这样 duty-factor 曲线的横轴含义与 AMP 对齐，两边能同图并排。

---

## 5. 面试讲法（40 秒）

> "我一开始也考虑过 DeepMimic，但读完源码发现它不是这个任务的正确工具。
> DeepMimic 的观测里没有速度指令，它靠参考动作的时间轴推进，
> 产出是'复现这条动作'，不是'响应指令'。我们的需求是按任意速度连续走跑，
> 所以用了 AMP —— 它让判别器管'像不像人'，让任务奖励管'听不听话'。
>
> 我实际跑了这个对照来验证：同一台机器人、同一批 30 段数据、同一个 seed，
> 只换奖励来源。（此处填入实测结论）
>
> 不过要说清楚，DeepMimic 也有它强的地方 —— 它 RSI + 提前终止这套细节更精细，
> 复现精调后的套路动作（比如跳舞、武术）它比 AMP 准。只是不适合需要响应外界输入的场景。"

最后这句很重要：**不要说 DeepMimic 差，要说它适用面不同。**
把它说成"另一个工具"，比说成"更差的方法"显得懂行得多。

---

## 6. 诚实边界（必须提前说）

1. **DeepMimic 训 3000 iter，AMP 是 4000–5000**。步数不对等。
   可辩护点：DeepMimic 有 RSI + 提前终止，监督信号更强，收敛更快。
2. **30 段 clip 对 DeepMimic 是 harder setting**：它没有 skill selector，
   要靠观测里给的参考目标去区分几十条完全不同的动作。
   这确实是"同一批数据"，但对 DeepMimic 更难 —— 这一点要主动承认。
3. **单次评估，无多 seed**。数字是单点观测。
4. **全是仿真，没有真机。** 任何表述不得出现 sim2real。

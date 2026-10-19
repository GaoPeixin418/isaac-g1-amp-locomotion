# GitHub 建仓库 + SSH 认证 + 自动备份

> 目标：让项目在 GitHub 上有一份异地副本，防止云服务器释放后代码/日志/报告丢失。
> 本机的备份循环（每 10 分钟）负责把服务器资料拉回本地并 commit；有了 remote 之后会额外 push 一份到 GitHub。

---

## 状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 本仓库 git | ✅ | `D:\workbuddy\...\g1-amp-locomotion`，已有 3 条 commit |
| git 身份 | ✅ | `gpx / gpeixin0@gmail.com` |
| 本地 SSH 密钥 | ✅ 已生成 | `C:\Users\GPX\.ssh\id_ed25519`（ed25519） |
| GitHub 建仓库 | ⏳ **需要你做** | 见下方步骤 A |
| GitHub 添加 SSH 公钥 | ⏳ **需要你做** | 见下方步骤 B |
| remote 挂载 + 首次 push | ⏳ 等你给仓库名 | 我来做 |

---

## 步骤 A：在 GitHub 建空仓库（1 分钟）

1. 浏览器打开 <https://github.com/new>
2. **Repository name** 填：`isaac-g1-amp-locomotion`
   （想换名字也行，告诉我实际用的名字即可）
3. **Description**（可选）：`Unitree G1 拟人态走跑：Isaac Lab + AMP（LAFAN1 参考动作），含速度指令的走/跑切换`
4. 选 **Public**
   —— 面试官要看代码，公开仓库可以直接发链接；如果里面有敏感内容再改私有
5. **三个勾选项全部不要勾**：
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license

   > 因为本地已经有 README 和 .gitignore 了，勾了会产生冲突，还得手动 merge，浪费时间
6. 点 **Create repository**
7. 创建完成后，页面上会显示仓库地址，把 **SSH 地址** 发我，形如：
   ```
   git@github.com:<你的用户名>/isaac-g1-amp-locomotion.git
   ```
   （页面上点 SSH 按钮切换，默认是 HTTPS，别复制错了）

---

## 步骤 B：把本机公钥加到 GitHub（1 分钟）

1. 复制下面这一整行（这就是本机的 SSH 公钥，公开无风险）：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKY0vUEP8s4eONVSjfvm3HhqHp/N+DQfT6E7dcGAPpNk gpeixin0@gmail.com
```

2. 打开 <https://github.com/settings/ssh/new>
3. **Title** 填：`MagicBook-Pro-16`（随便，标识这台机器）
4. **Key type** 保持 `Authentication Key`
5. **Key** 粘贴上面那一行
6. 点 **Add SSH key**（可能会要求输入一次 GitHub 密码确认）

---

## 步骤 C：剩下的我来做

你发我仓库 SSH 地址后，我执行：

```bash
git remote add origin git@github.com:<用户名>/<仓库名>.git
git push -u origin master     # 首次推送 + 建立追踪
ssh -T git@github.com         # 验证认证是否通过
```

之后备份循环每次 commit 后会自动 push，你不用管。

---

## 原理补充：为什么用 SSH 而不是 HTTPS + Token

大白话讲一下，面试或以后自己搭环境都用得上：

- **HTTPS 方式**：GitHub 早在 2021 年就废掉了「用账号密码 push」，必须改用 Personal Access Token（PAT）。PAT 本质是一串长密码，有过期时间，还得手动存到凭据管理器里；本机没装 Git Credential Manager（`credential.helper` 是空的），配起来更麻烦。
- **SSH 方式**：本地生成一对密钥——私钥留在 `C:\Users\GPX\.ssh\id_ed25519` 永不外传，公钥上传到 GitHub。push 时 GitHub 用公钥挑战、本地用私钥应答，证明"你确实是这台机器的主人"。**没有过期时间、不用每次输密码、一台机器配一次终身可用**，以后 Poppy 项目也能直接复用同一把钥匙。

安全边界说明：私钥文件只存在本机，我没有上传到任何地方；本项目结束后如果这台机器不再用，删掉 GitHub 上那条 SSH key 即可吊销。

---

## 备份机制总览（三层）

| 层 | 位置 | 保护什么 | 频率 |
|---|---|---|---|
| 服务器 | `/data`（200G 数据盘） | 环境、代码、训练 checkpoint | 实时写完即在 |
| 本地 | `backup/server/<时间戳>/` | 日志、脚本、配置（>50MB 跳过） | 每 10 分钟自动拉 |
| Git | 本地仓库 + GitHub | 全部文档/脚本/报告 | 每次同步后自动 commit + push |

> 注意：模型 checkpoint（每个几百 MB）不进 git，靠服务器 `/data` + 本地 `backup/` 两层保住。真正重要的 final model 我会单独下载到本地。

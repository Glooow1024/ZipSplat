# Scene token 开发入口

本工作区分支为 `dev/scene_token`，从 `dev/multiview` 创建。

开始工作前完整阅读父目录 `../AGENTS.md`、`../PROJECT_CONTEXT.md`，然后阅读 `docs/SCENE_TOKEN_DEVELOPMENT_PLAN_ZH.md`。当前后续开发以该计划为准，并结合用户最新指令确定本次执行范围。

- 使用中文沟通；本地编辑，GPU 实验在 `ssh vllm1` 执行。
- 不从旧 `ZipSplat` 工作区复制 ToMe 改动，不合并 `dev/local_tome`。
- 本地与远端独立；修改前分别检查分支、HEAD、工作树和实际文件，不整体覆盖同步。
- 保留其他人的工作；不使用 `git add .`、强推、硬重置或自动清理旧工作区。
- 使用 ZipSplat 自己的服务器虚拟环境。独立远端 worktree 的相对路径、权重和依赖须单独核验。
- 完成阶段后更新计划状态与父目录交接，记录实验清单、代码版本、结果位置和未完成项。
- 若远端无法访问上述父目录文档，应先取得同步的交接内容，不能假设 Windows 文件在服务器存在。

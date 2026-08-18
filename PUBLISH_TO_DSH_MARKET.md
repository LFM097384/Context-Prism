# 发布到 DSH Plugin Market

DSH 插件市场的数据源是：

- 市场 App：https://github.com/dsh-market/dsh-market
- 插件目录（curated registry）：https://github.com/awesome-dsh-plugin/awesome-dsh-plugin
- 在线目录：https://awesome-dsh-plugin.com/plugins.json

**不要**向 `dsh-market/dsh-market` 提插件 PR；要向 `awesome-dsh-plugin/awesome-dsh-plugin` 提 PR。

## 步骤

1. 先确保 GitHub 仓库公开：
   - https://github.com/LFM097384/Context-Prism
   - 仓库根目录就是 DSH 插件包（`package.json` + `lib/` + `engine/`）

2. Fork 并克隆：

```bash
git clone https://github.com/awesome-dsh-plugin/awesome-dsh-plugin.git
cd awesome-dsh-plugin
```

3. 在插件列表中加入一条 ContextPrism 条目。

本仓库已准备好待提交内容：`dsh-market-entry.json`

```json
{
  "name": "ContextPrism",
  "owner": "LFM097384",
  "url": "https://github.com/LFM097384/Context-Prism",
  "category": "memory",
  "description": {
    "en": "Project-level Local Context Engine for DeepSeek Harness: retrieval/compression/prioritization over history, code, files, user preferences and agent trajectories, producing a dynamic context window for any LLM.",
    "zh": "面向 DeepSeek Harness 的项目级本地上下文引擎：对历史、代码、文件、用户偏好和 Agent 轨迹做检索/压缩/优先级排序，为任意 LLM 生成动态上下文窗口。"
  },
  "npm": null,
  "install": "dsh plugin --profile web add github:LFM097384/Context-Prism",
  "added": "2026-08-18"
}
```

4. 提交并推送，然后向 `awesome-dsh-plugin/awesome-dsh-plugin` 打开 PR。

5. PR 合并后，通常一天内 DSH 市场会自动更新。

## 说明

- 当前条目使用 `github:LFM097384/Context-Prism` 安装方式，不需要 npm 发布。
- 如果以后发布到 npm，可以把 `npm` 字段改成实际包名，市场会优先走 npm tarball 安装。

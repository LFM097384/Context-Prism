# ContextPrism 架构文档

> 版本：v0.7.0  
> 定位：DeepSeek Harness (DSH) 的本地上下文引擎 / 动态 context window 插件

---

## 1. 概述

ContextPrism 的目标是：**在调用任何 LLM API 之前，把本地项目里的无限历史、代码、文件、用户偏好、Agent trajectory 做 retrieval / compression / prioritization，输出一个受 token 预算约束的动态 context window。**

它不是一个独立的记忆数据库，也不是一个简单 RAG 工具，而是 DSH 与任意 LLM 之间的一层“上下文预处理层”。

---

## 2. 核心架构图

```text
┌────────────────────────────────────────────────────────────────────┐
│                         DSH Workspace / Project                     │
│                                                                      │
│  Session Events        Files & Code          User Preferences        │
│  (history/trajectory)  (md/ts/py/json...)   (preference chunks)     │
└───────────┬──────────────────┬───────────────────────┬──────────────┘
            │                  │                       │
            ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ContextPrism Host Plugin                     │
│                                                                      │
│  Auto Capture          Auto Index Files         File Watcher        │
│  (agent/pre-step)      (incremental scan)       (fs.watch)          │
│                                                                      │
│  LocalContextEngine                                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Ingest → Retrieve → Prioritize → Compress → Assemble        │    │
│  │                                                             │    │
│  │ BM25 / SQLite FTS5 + Local Semantic Embedding               │    │
│  │ Cache / Token precompute / Optional LLM Summarization       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Tools: build / ingest / status / dashboard / evaluate / summarize  │
│  HTTP Route: /context-prism/dashboard                               │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Any LLM API                                  │
│  DeepSeek / OpenAI / compatible gateways                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 设计目标

| 目标 | 说明 |
| --- | --- |
| 项目级隔离 | 每个 DSH workspace 独立索引，默认 `.lce/index.db` |
| 本地优先 | 不依赖云端存储，不额外消耗生产 API key |
| Provider-agnostic | 只负责“造窗口”，不绑定具体模型 |
| 自动上下文层 | 尽量让模型无感，自动捕获 + 自动注入 |
| 高性能 | SQLite FTS5 + 缓存 + 预计算 token + 增量 ingest |
| 可观测 | status / dashboard / evaluate 工具 |

---

## 4. 模块结构

```text
Context-Prism/
├── package.json               # DSH 插件包 + bundle + client 声明
├── cordis.patch.yml           # DSH bundle patch（激活插件）
├── lib/
│   ├── index.js               # Host 插件入口：工具、自动捕获/注入、文件监听、HTTP 路由
│   └── client.js              # DSH Web 客户端：侧边栏 ContextPrism 按钮
├── engine/                    # 核心 JS 引擎（进程内运行）
│   ├── models.js              # Chunk / ContextWindow / SourceKind
│   ├── text_utils.js          # tokenize / sentence split / TF
│   ├── cache.js               # TTL + LRU 缓存
│   ├── index.js               # 纯 JS BM25 索引
│   ├── sqlite_index.js        # Node 内置 SQLite FTS5 索引
│   ├── embedding.js           # 本地 hash n-gram embedding
│   ├── retrieval.js           # 混合检索打分
│   ├── prioritization.js      # 优先级排序
│   ├── compression.js         # 压缩 / 截断 / 省略提示
│   ├── summarizer.js          # 可选 LLM 摘要 + extractive fallback
│   ├── ingest.js              # 文件 / history / trajectory 导入
│   ├── engine.js              # LocalContextEngine 编排器
│   └── providers.js           # DeepSeek / OpenAI payload
├── test/                      # Node 测试
│   ├── engine.test.js
│   └── plugin.test.js
└── python/                    # Python 参考实现（可选）
```

---

## 5. 核心数据流

### 5.1 Ingest

- **Session 自动捕获**
  - 监听 `agent/pre-step`
  - 增量读取 `agent.session.events`
  - 将 `user/message`、`assistant/message` 作为 History chunk
  - 将 `tool/call`、`tool/result` 作为 Trajectory chunk
  - 通过 `seq` 记录已捕获位置，避免重复

- **文件自动索引**
  - 默认索引：`.md .txt .py .ts .tsx .js .jsx .json .jsonl .csv .html .css`
  - 默认忽略：`node_modules .git .lce dist build .venv __pycache__ .next coverage .turbo`
  - 使用 `mtime + size` 做增量状态，跳过未变化文件

- **实时文件监听**
  - `fs.watch(projectRoot, { recursive: true })`
  - 文件变化后防抖 `watchDebounceMs`（默认 2000ms）
  - 触发增量重索引

### 5.2 Retrieve

```text
score =
  relevanceWeight * BM25/FTS5 rank
+ semanticWeight  * cosine(queryEmbedding, chunkEmbedding)
+ recencyWeight   * exp(-age / halfLife)
+ sourceWeight    * sourceBoost
+ priorityWeight  * priority/10
```

- BM25：`engine/index.js` 或 `engine/sqlite_index.js`
- 语义：`engine/embedding.js` 的 hash n-gram embedding + cosine
- 来源权重：`preference > trajectory > history > code > file`

### 5.3 Prioritize

- 用户偏好永远优先
- 再按检索分、来源基础分、时间衰减、长度惩罚综合排序
- 支持 chunk 级 `priority` 字段

### 5.4 Compress

- 高优先级 chunk 完整保留
- 超预算 chunk：
  1. extractive sentence compression
  2. 可选 LLM 摘要（`llmSummarization: true`）
  3. 最后截断
- 被省略内容生成 `Omitted Memory` 提示

### 5.5 Assemble

最终 ContextWindow 按 section 组装：

```text
User Preferences → Relevant History → Agent Trajectory → Relevant Code → Relevant Files → Omitted Memory
```

输出形式：

- `window.text`：纯文本
- `window.asMessages()`：OpenAI 风格 messages
- `toDeepSeekPayload()`：DeepSeek API payload
- `window.report()`：JSON 可观测报告

---

## 6. DSH 集成

### 6.1 自动注入

在 `agent/pre-step` 中：

1. 解析当前 DSH workspace
2. 获取/创建项目级 engine
3. 自动捕获新 session 事件
4. 自动增量索引项目文件（带节流）
5. 构建 context window
6. 将结果作为 user message 注入到即将发送给 LLM 的 messages 中

因此模型不需要主动调用工具，ContextPrism 会自动出现在上下文里。

### 6.2 模型工具

| Tool | 作用 |
| --- | --- |
| `context_prism_build` | 手动构建动态 context window |
| `context_prism_ingest` | 手动灌入文件 / history / trajectory |
| `context_prism_status` | 查看索引状态 |
| `context_prism_dashboard` | 生成独立 HTML dashboard |
| `context_prism_evaluate` | A/B 检索评估 |
| `context_prism_summarize` | LLM / extractive 总结文本 |

### 6.3 HTTP 路由

- 路由：`GET /context-prism/dashboard`
- 返回当前项目的实时 HTML dashboard
- DSH Web 侧边栏底部按钮直接打开该路由

### 6.4 Client 插件

- `lib/client.js` 使用 DSH `__ModuleLoader__` 加载
- 注册 `sidebar.footer.action` 插槽
- 点击按钮打开 `/context-prism/dashboard`

---

## 7. 存储设计

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| 主索引 | `<workspace>/.lce/index.db` | SQLite FTS5 |
| 备用索引 | `<workspace>/.lce/index.json` | 纯 JS BM25 |
| 增量状态 | `<workspace>/.lce/index.db.state.json` | mtime + size |
| Dashboard | `<workspace>/.lce/dashboard.html` | 独立 HTML 面板 |
| 运行时缓存 | 内存 | TTL + LRU |

---

## 8. 配置参考

```yaml
context-prism:
  backend: "auto"              # auto | sqlite | json
  defaultIndex: ".lce/index.db"
  cacheSize: 128
  cacheTtl: 60

  autoInject: true             # 每次 LLM 请求前自动注入
  autoIndexFiles: true         # 自动增量索引项目文件
  autoMaxTokens: 4000
  autoReservedTokens: 800
  autoIndexIntervalMs: 60000

  fileWatch: true
  watchDebounceMs: 2000

  llmSummarization: false
  summarizationModel: "deepseek-chat"
  summaryMaxTokens: 200
```

---

## 9. 性能设计

- SQLite FTS5 全文检索：O(log N) 级别
- 检索 / 压缩结果缓存：TTL + LRU，索引 revision 变化自动失效
- 预计算 token：ingest 时算好，热路径不重复估算
- 增量 ingest：mtime + size 跳过未变文件
- 文件监听防抖：避免频繁全量扫描
- 本地语义 embedding：固定维度 hash，无外部模型调用

---

## 10. DeepSeek 兼容

- `toDeepSeekPayload(window, { model: "deepseek-chat" })`
- `deepseek-reasoner` 自动省略 `temperature`
- `callDeepSeek()` 使用标准库 `fetch` 发送请求
- 未配置 `DEEPSEEK_API_KEY` 时不会自动调用外部 API

---

## 11. 测试策略

```bash
cd Context-Prism
npm install
npm test
```

覆盖：

- Token 估算（中英文）
- BM25 / SQLite FTS5 检索
- 压缩预算
- 语义 embedding
- LLM summarizer fallback
- 端到端 context 构建
- DSH 工具注册与执行
- workspace 级默认索引
- 自动捕获 + 自动注入
- HTML dashboard 生成
- A/B 检索评估

当前：`15 tests / 15 pass`

---

## 12. 发布状态

- GitHub：https://github.com/LFM097384/Context-Prism
- DSH 市场 PR：#1772（等待仓库年龄满 1 天后通过 gate）
- 当前版本：v0.7.0

---

## 13. 后续方向

- 真正的向量数据库 / 本地 transformer embedding
- DSH 内嵌面板从“打开新标签”升级为真正嵌入式 UI
- 更多 provider adapter（Anthropic / Gemini / Ollama）
- 基于真实问答数据的离线评测集
- 自动记忆 consolidation / 分层记忆

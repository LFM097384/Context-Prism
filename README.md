# ContextPrism

**ContextPrism** 是一个面向 DeepSeek Harness (DSH) 的本地上下文引擎 / 动态 context window 插件。

它会在调用任何 LLM 之前，对以下本地数据做 **retrieval / compression / prioritization**：

- 无限历史（会话 / 对话）
- 代码
- 文件
- 用户偏好
- Agent trajectory

然后输出一个受 token 预算约束的 **动态 context window**，可适配任意 LLM（DeepSeek / OpenAI / 其他兼容接口）。

## 特性

- **DSH workspace / 项目级别**
  - 通过 `ctx.workspaceRegistry` 自动定位当前 DSH workspace
  - 每个项目独立存储：`.lce/index.db`
- **进程内 JS 实现**
  - 纯 JS BM25 + Node 内置 SQLite FTS5
  - 不依赖 Python / WSL
- **检索 / 压缩 / 优先级**
  - 混合检索：BM25 + 本地语义向量（hash n-gram embedding）+ 时间衰减 + 来源权重 + 用户优先级
  - 压缩：完整保留高优先级片段，超预算时 extractive compression / 可选 LLM 摘要 + 截断
  - 动态组装：`User Preferences → History → Trajectory → Code → Files`
- **LLM 摘要（可选）**
  - 配置 `llmSummarization: true` 后，压缩阶段会尝试用 DeepSeek 生成摘要
  - 未配置 API key 时自动回退到本地 extractive 压缩
- **DeepSeek / OpenAI 兼容**
  - 直接生成 `deepseek-chat` / `deepseek-reasoner` / OpenAI 兼容 payload
- **缓存与增量**
  - 检索 / 压缩结果 TTL+LRU 缓存
  - 文件增量 ingest，按 mtime + size 跳过未变更内容
- **自动捕获 + 自动注入**
  - 自动把当前会话历史、assistant 消息、tool call / tool result 写入项目索引
  - 自动增量索引项目文件（默认 60s 内最多扫描一次）
  - 在每次 LLM 请求前自动注入动态 context window，模型无需手动调用工具

## 目录

```text
Context-Prism/
├── package.json            # DSH 插件包
├── lib/index.js            # DSH 插件入口
├── engine/                 # JS 版 ContextPrism 核心
│   ├── models.js
│   ├── text_utils.js
│   ├── cache.js
│   ├── index.js
│   ├── sqlite_index.js
│   ├── retrieval.js
│   ├── prioritization.js
│   ├── compression.js
│   ├── ingest.js
│   ├── engine.js
│   └── providers.js
├── test/                   # Node 测试
│   ├── engine.test.js
│   └── plugin.test.js
└── python/                 # Python 参考实现（可选）
    ├── local_context_engine/
    ├── tests/
    └── pyproject.toml
```

## 安装到 DSH

```bash
# 从 GitHub 安装（需要仓库公开后）
dsh plugin --profile web add github:LFM097384/Context-Prism

# 或在 profile 的 cordis.patch.yml 中激活
```

```yaml
- insert:
    - id: context-prism
      name: 'context-prism-dsh-plugin'
      config:
        backend: 'auto'
        defaultIndex: '.lce/index.db'
        autoInject: true
        autoIndexFiles: true
        autoMaxTokens: 4000
        autoReservedTokens: 800
        autoIndexIntervalMs: 60000
        llmSummarization: false
        summarizationModel: 'deepseek-chat'
        summaryMaxTokens: 200
```

## 工具

| Tool | 作用 |
| --- | --- |
| `context_prism_build` | 在调用 LLM 前构建动态 context window |
| `context_prism_ingest` | 把文件/目录/history/trajectory 灌入本地索引 |
| `context_prism_status` | 查看当前项目索引状态（chunk 数量、类型、来源） |
| `context_prism_summarize` | 用 LLM 或本地 extractive 方式总结文本 |

## 测试

```bash
npm install
npm test
```

## Python 参考实现

```bash
cd python
python -m unittest discover -s tests -v
```

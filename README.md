# Rova

> 一个支持工具调用、持久会话、长期记忆与可扩展能力的本地通用 Agent。

Rova 面向本地单用户场景，可操作项目文件、执行命令、搜索网络，并通过 Session、Memory、Skills 与 Experience Evolution 保持连续的任务上下文和可复用经验。

Rova 使用统一的 Agent Runtime 组合 Workspace、Web、MCP、Extensions 等能力，并通过 CLI 或 Terminal UI 提供一致的使用体验。

```bash
# 在当前项目中启动
python -m rova --workspace .

# 开启联网能力
python -m rova --web

# 同时使用本地项目与网络工具
python -m rova --workspace . --web
```

---

## ✨ 核心能力

- 🤖 **统一 Agent Runtime**
  基于 Tool Calling 完成模型交互、工具调用与任务执行，CLI 与 TUI 共用同一套 Runtime。

- 🛠️ **项目与终端操作**
  支持目录浏览、文件读取、搜索、写入、编辑与 Shell 执行，并可选择 Local 或 Docker Terminal Backend。

- 🔒 **统一权限控制**
  Native、Extension 与 MCP Tool 统一经过 Policy 与 Approval；Workspace 文件操作受到路径边界约束。

- 🌐 **Web 与 MCP**
  支持 Web Search、网页获取，以及用户配置的 STDIO / Streamable HTTP MCP Server。

- 💬 **持久化 Session**  
  使用 JSONL 保存会话，支持恢复、分支与长上下文压缩。恢复时会补齐未闭合的 ToolCall；Rova 不会自动重试可能已有副作用的 Tool，并会提示先检查当前状态。

- 🧠 **Memory 与 Experience Evolution**
  保存长期信息；自动经验审查可更新 Memory，并生成待处理的 Skill Proposal，不会直接修改已安装的 Active Skill。

- 🧩 **Skills**  
  通过 Skill Catalog 发现能力，并在需要时按需读取完整 Skill 内容。

- 🧱 **Extensions**
  支持受信任的本地 Python 扩展注册 Tool、订阅 Agent Event 或提供动态 Context。

---

## 🏗️ 架构概览

```text
                         User
                          │
                     CLI / TUI
                          │
                          ▼
                     RovaRuntime
                    /           \
                   /             \
        Session / Memory /      Agent
          Skills                  │
                                  ▼
                            ToolRegistry
                       ┌──────────┼──────────┐
                       ▼          ▼          ▼
                    Native    Extension      MCP
                       └──────────┼──────────┘
                                  ▼
                         Policy / Approval
                                  │
                                  ▼
                              Execution
                         /                \
                  Workspace / Web       Terminal
                                      Local / Docker
```

---

## 🚀 快速开始

### 环境要求

| 环境         | 要求                  |
| ------------ | --------------------- |
| Python       | 3.10+                 |
| LLM Provider | OpenAI-compatible API |
| Node.js      | 仅 TUI 需要           |
| pnpm         | 仅 TUI 需要           |

### 1. 克隆项目

```bash
git clone https://github.com/Racy324/Rova.git
cd Rova
```

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3. 配置模型

在项目根目录创建 `.env`：

```dotenv
ROVA_PROVIDER=openai_compatible
ROVA_MODEL=<model-name>
ROVA_BASE_URL=<provider-base-url>
OPENAI_API_KEY=<api-key>
```

更多配置项见 `.env.example`。

### 4. 启动 Rova

进入交互模式：

```bash
python -m rova
```

执行单次任务：

```bash
python -m rova "分析一下当前项目结构"
```

操作当前项目：

```bash
python -m rova --workspace . "检查这个项目并总结核心模块"
```

联网查询：

```bash
python -m rova --web "调研最近的 Agent Memory 方案"
```

同时使用本地项目与 Web：

```bash
python -m rova --workspace . --web "调研相关实现，并结合当前项目给出改进建议"
```

启动 Terminal UI：

```bash
python -m rova --tui --workspace .
```

---

## ⚙️ 常用参数

| 参数                               | 作用                                   |
| ---------------------------------- | -------------------------------------- |
| `--workspace PATH`                 | 指定 Agent 可操作的 Workspace          |
| `--terminal-backend local\|docker` | 在 Workspace 中临时选择 Shell 执行后端 |
| `--docker-image IMAGE`             | 选择 Docker 后端时指定或覆盖 image     |
| `--mcp-config PATH`                | 临时指定 MCP 配置文件                  |
| `--web`                            | 开启 Web Search 与页面抓取             |
| `--context-path FILE`              | 加入 UTF-8 本地资料，可重复指定        |
| `--save`                           | 将最终结果保存为 Artifact              |
| `--data-dir PATH`                  | 指定 Rova 本地数据目录                 |
| `--permission ask\|full`           | 设置需要 Approval 的操作如何处理       |
| `--max-turns N`                    | 设置单次任务最大 Agent Loop 轮数       |
| `--tui`                            | 启动 Terminal UI                       |

---

## 🖥️ Terminal Backend

启用 `--workspace` 后，Rova 默认使用 **Local Terminal**，直接在宿主机环境中执行已批准的命令，可复用本地 Python、Conda、CUDA、Git 等开发环境。它不是文件系统沙箱。

也可以选择 **Docker Terminal**，在隔离的 Linux 容器环境中执行 Shell：

```bash
python -m rova --workspace . \
  --terminal-backend docker \
  --docker-image python:3.12-slim \
  "运行测试并分析失败原因"
```

Docker 模式会将当前 Workspace 映射到容器，因此容器内对 Workspace 的修改仍会反映到宿主项目。Docker 不替代 Rova 的 Policy 与 Approval，也不提供项目文件快照或自动回滚。

可以在 `.env` 中保存默认设置：

```dotenv
ROVA_TERMINAL_BACKEND=docker
ROVA_DOCKER_IMAGE=python:3.12-slim
```

CLI 参数只覆盖当前启动，不会写回持久配置。

---

## 🔌 MCP

Rova 可以连接用户配置的 MCP Server，并将发现到的能力作为普通 Agent Tools 使用。目前支持 **STDIO** 与 **Streamable HTTP**。

通过 `.env` 指定 MCP 配置文件：

```dotenv
ROVA_MCP_CONFIG=/path/to/mcp.toml
```

示例配置：

```toml
[mcp_servers.example]
enabled = true
transport = "stdio"
command = "uvx"
args = ["example-mcp"]
include_tools = ["search"]
```

`include_tools` 只控制哪些 MCP Tool 暴露给 Agent；实际调用仍与 Native 和 Extension Tool 共用同一套 Policy 与 Approval。

配置中的敏感 header 或 env 值必须通过环境变量引用，例如：

```toml
env = { SERVICE_TOKEN = "${SERVICE_TOKEN}" }
```

Rova 不会自动安装或启用未显式配置的 MCP Server。

---

## 🧱 Extensions

Rova 支持受信任的本地 Python Extension，可注册自定义 Tool、订阅 Agent Event 或提供动态 Context。

扩展可放在：

```text
~/.rova/extensions/               # 默认用户级路径
<workspace>/.rova/extensions/
```

使用 `--data-dir` 或 `ROVA_DATA_DIR` 时，用户级路径相应为该数据目录下的 `extensions/`。

示例见 [minimal_extension.py](examples/extensions/minimal_extension.py)。

Extension 会直接作为本地 Python 代码运行，因此应只加载可信来源。

---

## 💻 Terminal UI

Rova 提供基于 TypeScript、React 与 Ink 构建的 Terminal UI。它需要交互式终端；非交互环境会回退到滚动式 CLI 输出。

先构建前端：

```bash
cd ui-tui
pnpm install
pnpm build
cd ..
```

启动：

```bash
python -m rova --tui --workspace .
```

TUI 支持：

- 对话与流式输出
- Tool Activity
- Approval
- Session 新建、查看与恢复
- Markdown / Code 显示
- Runtime 状态展示

CLI 与 TUI 共用同一个 Python Runtime，不维护第二套 Agent、Session、Tool 或 Policy 实现。

---

## 📁 本地数据

Rova 默认将本地运行数据保存在：

```text
~/.rova/
├── sessions/
├── artifacts/
├── memory/
├── skills/
├── extensions/
└── experience/
```

数据根目录优先级：

```text
--data-dir
    >
ROVA_DATA_DIR
    >
~/.rova
```

其中：

- `sessions/`：持久化会话
- `artifacts/`：运行时产物
- `memory/`：长期 Memory
- `skills/`：本地 Skills
- `skill-proposals/`：Experience Evolution 生成的待处理 Skill Proposal
- `extensions/`：用户安装的本地 Extensions
- `experience/`：Experience Evolution 运行状态

Experience Evolution 默认启用；如需关闭自动经验审查，可在 `.env` 中设置：

```dotenv
ROVA_EXPERIENCE_REVIEW_ENABLED=false
```

其他可调参数见 `.env.example`。

---

## 🖼️ 可选 Vision

配置独立 Vision 模型后，Rova 可以在 Workspace 中按需分析 PNG、JPG/JPEG 或 WebP 图片。相关配置见 `.env.example`。

---

## 📂 项目结构

```text
rova/
├── agent_core/       # Agent Loop、Tool Registry、Events
├── agent_session/    # Session、Restore、Branch、Compaction
├── ai/               # Message、Model、Context、Provider
├── app/              # Product Runtime 与应用层能力
│   ├── context/
│   ├── web/
│   └── workspace/
├── artifacts/        # Runtime Artifacts
├── eval/             # Evaluation
├── mcp/              # MCP 集成
└── trace/            # Runtime Trace

ui-tui/               # TypeScript + React + Ink Terminal UI
tests/                # Regression Tests
requirements.txt
```

---

## 🧪 开发与测试

Python：

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python -m compileall -q rova
git diff --check
```

TUI：

```bash
cd ui-tui
pnpm typecheck
pnpm build
```

---

## 隔离 Sandbox 工作流

Rova 有两种不同的执行环境：

- **Local（可信 Host）**：文件工具和 Shell 直接操作 Host Workspace。Policy 与逐次 Approval 不等同于文件系统隔离。
- **Sandbox（隔离）**：内置编码工具只操作 Rova 管理的 Sandbox；Shell 只在挂载该 Sandbox 的 Docker 容器中运行。Host 项目在显式 Apply 前保持不变。

```bash
python -m rova --workspace . --environment sandbox --sandbox-image python:3.12-slim
```

也可以在 `.env` 设置默认选择：

```dotenv
ROVA_EXECUTION_ENVIRONMENT=sandbox
ROVA_SANDBOX_IMAGE=python:3.12-slim
```

```text
Host Workspace → Sandbox baseline B0 → Agent 编码/测试 → ChangedSet
                                                        ├─ Apply → 修改 Host
                                                        └─ Discard → Host 不变
```

CLI 使用 `/sandbox status`、`/sandbox diff`、`/sandbox apply`、`/sandbox discard`、`/sandbox restore` 和 `/sandbox new`。Apply、Discard 与 Apply preimage restore 都需要独立确认；`--permission full` 不会跳过它们。TUI 状态栏持续显示环境，并提供相同的控制。

Sandbox 创建时捕获 Host 当前文件树为不可变 B0；退出后 Sandbox 文件保留，同一 Session 恢复时继续使用原 B0。容器会重新创建，因此容器内安装的包、进程、`/tmp` 和可写层状态不保证保留。Sandbox 的私有 Git 仅用于基线和 diff，不包含 Host Git 历史或远程仓库。

Sandbox 默认关闭网络，不自动继承 Host 环境变量、挂载 Host secrets 或暴露 Docker socket。这是 Host Workspace 隔离边界，不是对受信任 Extension、MCP 外部副作用或 Docker 本身的绝对安全承诺。

旧的 `--terminal-backend docker` / `ROVA_TERMINAL_BACKEND=docker` Host 直接挂载入口不再是公开隔离模式；请使用 `--environment sandbox` / `ROVA_EXECUTION_ENVIRONMENT=sandbox`。

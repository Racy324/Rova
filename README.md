# Rova

> 一个支持工具调用、长期记忆与 Skills 的本地通用 Agent。

Rova 是一个面向本地单用户的 AI Agent。它可以操作本地项目、执行命令、搜索网络，并通过持久化 Session、长期 Memory 和 Skills 保持连续的工作上下文。

Rova 使用统一的 Agent Runtime，根据启动时启用的能力组合 Workspace、Web、Skills 等工具，可通过 CLI 或 Terminal UI 使用。

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

- 🤖 **通用 Agent Runtime**  
  基于 Tool Calling 实现完整 Agent Loop，由统一 Runtime 负责模型交互、工具调用与任务执行。

- 🛠️ **本地项目操作**  
  支持目录浏览、文件读取、搜索、写入、编辑与 Shell 执行，可直接用于代码仓库和本地项目。

- 🔒 **Workspace 安全控制**  
  文件操作限制在指定 Workspace 内；Shell 在 Workspace 中启动（作为 cwd），但属于需审批的本机宿主命令，不是文件系统 sandbox。写入、编辑和命令执行经过 Policy 与 Approval 控制。

- 🐳 **可配置 Terminal Backend**
  默认使用 Local Terminal；也可显式选择 Docker，在 bind mount 的 `/workspace` 中执行命令。Docker 不替代 Policy/Approval，且对 Workspace 的写入会直接反映到宿主项目。

- 🌐 **联网搜索与网页获取**  
  支持 Web Search 与页面抓取，并保留来源身份与引用关系。

- 💬 **持久化 Session**  
  使用 JSONL 保存会话，支持恢复、分支与上下文压缩。

- 🧠 **长期 Memory**  
  使用本地 Markdown 文件保存长期用户信息；Agent 可通过 `memory_manage` 显式保存稳定事实或偏好。
- 🔄 **Experience Review**
  默认在正常完成任务后累计最小证据，并在任务数或合格 Tool 调用数达到阈值时审查可沉淀的 Memory/Skill 改进。Reviewer 只读 Skill，写入由 Runtime 服务验证后执行；当前会话的冻结上下文不会被审查结果中途刷新。

- 🧩 **Skills**  
  通过 Skill Catalog 发现能力，需要时按需读取完整 `SKILL.md`，避免一次性加载全部内容。

- 🧱 **本地 Extensions**
  可从用户或 Workspace 目录加载受信任的本地 Python 扩展，以注册工具、订阅 Agent 生命周期事件或提供小型动态上下文。

- 📦 **Artifacts 与 Trace**  
  支持保存运行产物，并记录 Agent、Tool 与 Session 的执行过程，方便调试和回溯。

- 💻 **CLI + TUI**  
  提供命令行与 Terminal UI，两者共用同一个 Rova Runtime。

---

## 🏗️ 系统架构

```text
                         User
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                   CLI           TUI
                    │             │
                    └──────┬──────┘
                           ▼
                    Product Runtime
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
           Agent         Context       Session
             │
             ▼
            Tools
      ┌──────┼────────┐
      ▼      ▼        ▼
 Workspace   Web    Skills
      │      │        │
      └──────┴────────┘
             │
             ▼
          Provider
```

Rova 始终使用一个统一 Runtime。启用不同参数，只是为当前 Agent 增加相应的工具与运行上下文，不会创建另一套 Agent 实现。

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
git clone <repository-url>
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

更多配置项可参考项目中的 `.env.example`。

### 可选：辅助 Vision

配置独立的 Vision 模型后，Rova 会在指定 Workspace 中提供 `vision_analyze`，按需将本地 PNG、JPG/JPEG 或 WebP 图片交给辅助模型观察；主 Agent 仍保持文本推理与工具调用。

```dotenv
ROVA_VISION_MODEL=qwen3-vl-flash
ROVA_VISION_BASE_URL=<vision-provider-base-url>
ROVA_VISION_API_KEY=<vision-api-key>
```

启动 `python -m rova --workspace .` 后，可请求 Agent 分析 Workspace 内的图片，例如 `./test-data/architecture.png`。未配置 Vision 时不会注册该工具。

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

---

## ⚙️ 常用参数

| 参数                     | 作用                             |
| ------------------------ | -------------------------------- |
| `--workspace PATH`       | 指定 Agent 可操作的 Workspace    |
| `--terminal-backend local\|docker` | 选择 Shell 执行后端，默认 `local` |
| `--docker-image IMAGE`   | Docker 后端必填的 Linux image     |
| `--web`                  | 开启 Web Search 与页面抓取       |
| `--context-path FILE`    | 加入 UTF-8 本地资料，可重复指定  |
| `--save`                 | 将最终结果保存为 Artifact        |
| `--data-dir PATH`        | 指定 Rova 本地数据目录           |
| `--permission ask\|full` | 设置需要 Approval 的操作如何处理 |
| `--max-turns N`          | 设置单次任务最大 Agent Loop 轮数 |
| `--tui`                  | 启动 Terminal UI                 |

### Terminal Backend

默认 Local 后端直接以 Workspace 为 cwd 在宿主机执行已批准的命令，不是 OS 级文件系统 sandbox。

Docker 后端必须同时显式指定 Workspace 和 Linux image：

```bash
python -m rova --workspace . \
  --terminal-backend docker \
  --docker-image rova-python:latest \
  "运行测试并分析失败原因"
```

Docker 后端在首次 Shell Tool Call 时使用 `docker run -d --rm` 创建一个由当前 Rova Runtime 独占的容器；同一 Runtime 后续调用使用 `docker exec` 复用它。Host Workspace 被读写挂载为容器 `/workspace`，因此 Host 文件工具与容器命令观察同一份文件，写入也会影响宿主项目。容器内文件系统状态（例如安装的依赖）可跨 ToolCall 保留；`cd`、`export` 等 shell process state 不保证保留。已安装 Skill 的根目录只读挂载在 `/opt/rova/skills`；Memory、Session、完整 `~/.rova` 和 Provider 凭据不会被挂载。timeout、取消和 Runtime 正常关闭会移除该容器。容器内命令使用 `/bin/sh -lc`，不支持 Windows `cmd.exe`/PowerShell 特有语法；不包含 SSH、Copy-in、GPU、网络开关或远程 Workspace。

也可在现有 `.env` 持久配置中设置默认值（CLI 仅覆盖当前启动，不会写回配置）：

```dotenv
ROVA_TERMINAL_BACKEND=docker
ROVA_DOCKER_IMAGE=python:3.12-slim
```

优先级为：显式 CLI > `.env` 配置 > `local` 程序默认值。若最终选择 Docker 但没有 image，启动会明确失败。

---

## 🔌 本地 Extensions

Rova 在启动时按以下顺序加载顶层 Python 文件；后加载的扩展不能注册同名工具：

```text
~/.rova/extensions/
<workspace>/.rova/extensions/
```

扩展的 `setup(api)` 可调用 `register_tool(...)`、`on(event_type, handler)` 和 `register_context_provider(...)`。示例见 [minimal_extension.py](examples/extensions/minimal_extension.py)。Extension 是用户主动安装的受信任本地代码；当前不提供依赖隔离、热重载或沙箱。

---

## 💻 Terminal UI

Rova 提供基于 TypeScript、React 与 Ink 构建的 Terminal UI。

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

同时启用 Web：

```bash
python -m rova --tui --workspace . --web
```

TUI 支持：

- 对话与流式输出
- Tool Activity
- Approval
- Session 新建、查看与恢复
- Markdown / Code 显示
- Runtime 状态展示

运行中的当前 Turn 可通过 `Esc` 或 `Ctrl+C` 取消；空闲时 `Ctrl+C` 退出 TUI。审批框中 `Y` 仅允许当前调用、`N` 仅拒绝当前调用，`Esc` 取消整个当前 Turn。

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
└── extensions/
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
- `memory/`：`USER.md` 与 `MEMORY.md`
- `skills/`：本地 Skills
- `extensions/`：用户安装的本地 Extension
- `experience/`：待审任务证据、计数器和脱敏审查日志

Experience Review 默认开启；如需完全关闭自动审查，同时保留显式 `memory_manage`，可在 `.env` 中设置：

```dotenv
ROVA_EXPERIENCE_REVIEW_ENABLED=false
# 默认值：Tool=10，已完成任务=5；任一阈值达到即审查。
# ROVA_EXPERIENCE_REVIEW_TOOL_THRESHOLD=10
# ROVA_EXPERIENCE_REVIEW_TASK_THRESHOLD=5
```

---

## 📂 项目结构

```text
rova/
├── agent_core/       # Agent Loop、Tool Registry、Events
├── agent_session/    # Session、Restore、Branch、Compaction
├── ai/               # Message、Model、Stream、Provider
├── app/              # Product Runtime 与产品能力组合
│   ├── context/
│   ├── web/
│   └── workspace/
├── artifacts/        # Runtime Artifacts
├── eval/             # Evaluation
├── mcp/              # MCP Client Adapter
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

### MCP 工具

MCP 仅在显式指定 `ROVA_MCP_CONFIG` 或 `--mcp-config PATH` 时启用。配置中的 `include_tools` 只决定哪些远端工具可被发现；每次执行仍会经过同一套 Policy 和 Approval。STDIO Server 在宿主机启动，并只获得安全基础环境加配置中显式引用的环境变量；不会默认继承 Provider 凭据或完整宿主环境。MCP discovery 在后台执行，不阻塞 CLI、TUI 或首个请求；新发现的工具从后续 Provider Turn 可见。

```toml
[mcp_servers.example]
enabled = true
transport = "stdio"
command = "uvx"
args = ["example-mcp"]
include_tools = ["search"]
```

敏感 header/env 值只能写为 `${ENV_NAME}` 引用，例如 `env = { SERVICE_TOKEN = "${SERVICE_TOKEN}" }`。MCP 连接是受信任的本地/远端扩展连接，不是 sandbox；未实现自动重连、OAuth、热重载或 MCP 管理命令。

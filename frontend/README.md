# 反欺诈 RAG 系统 — 前端

基于 React + TailwindCSS 构建的现代化前端页面，为 `antifraud-rag` Python 库提供可视化操作界面。

## 功能页面

| 页面 | 路由 | 说明 |
|------|------|------|
| 仪表盘 | `/dashboard` | 系统状态、架构概览、功能导航 |
| 文本分析 | `/analyze` | 输入可疑文本进行欺诈风险评估（核心功能） |
| 案例库 | `/cases` | 向向量知识库添加历史诈骗案例 |
| 知识库 | `/tips` | 添加反诈防骗知识，辅助 RAG 分析 |
| 混合搜索 | `/search` | BM25 + 向量 + RRF 混合检索调试工具 |

## 快速启动

### 1. 启动后端 API

```bash
# 在项目根目录 (antichet_RAG/)
uv sync --all-extras

# 确保已配置 .env 文件（参考 .env.example），并已初始化数据库表
uv run uvicorn api.main:app --reload --port 8000
```

### 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开 http://localhost:5173 即可访问。

## 技术栈

- **React 18** — UI 框架
- **React Router v6** — 前端路由
- **TailwindCSS 3** — 原子化 CSS 样式
- **Lucide React** — 图标库
- **Axios** — HTTP 请求
- **Vite 5** — 构建工具（开发时代理 /api → localhost:8000）

## 开发说明

Vite 配置了 API 代理，开发时所有 `/api/*` 请求将自动转发到 `http://localhost:8000`。

如需连接不同 API 地址，可设置 `VITE_API_BASE_URL`，例如：

```bash
VITE_API_BASE_URL=http://localhost:8000/api npm run dev
```

生产部署使用 React Router 的 history 模式，静态服务器需要将 `/dashboard`、`/analyze` 等前端路由回退到 `index.html`，并确保 `/api` 代理到后端服务，或通过 `VITE_API_BASE_URL` 指向后端 API。

生产构建：

```bash
npm run build
# 产物在 dist/ 目录，可部署到任何静态文件服务器
```

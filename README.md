# antifraud-rag

**反欺诈 RAG 系统** - 基于 BM25 + 向量搜索 + RRF 融合的 Python 库

> 这是一个**纯 Python 库**，不包含 FastAPI/HTTP 服务；请通过导入 `AntiFraudRAG` / `FraudAnalyzer` 在代码中使用。

## 简介

`antifraud-rag` 是一个专注于欺诈信息识别的 RAG（Retrieval-Augmented Generation）Python 库。通过混合检索策略（BM25 精确匹配 + 向量语义搜索 + RRF 融合排序），结合历史案例库和反诈知识库，输出风险评估分析。

核心特性：
- **混合检索**: BM25 + 向量搜索 + RRF 融合（k=60）
- **双库设计**: 案例库（历史诈骗案件）+ 知识库（反诈技巧文档）
- **PostgreSQL/pgvector**: 高效向量存储和全文搜索
- **异步支持**: SQLAlchemy asyncpg 驱动
- **阈值判定**: RRF score > 0.85 直接返回高危告警

## 安装

```bash
uv sync
```

开发安装（含 dev 依赖）:

```bash
git clone https://github.com/your-org/antifraud-rag.git
cd antifraud-rag
uv sync --all-extras
```

## 配置

通过 `Settings` 类显式传入配置（不依赖环境变量）：

```python
from antifraud_rag import Settings

settings = Settings(
    EMBEDDING_MODEL_URL="https://your-embedding-api.com/v1/embeddings",  # 必需
    EMBEDDING_MODEL_API_KEY="your-api-key",                              # 必需
    DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/dbname",      # 可选
    EMBEDDING_MODEL_NAME="text-embedding-ada-002",                       # 可选，默认 "text-embedding-ada-002"
    EMBEDDING_DIMENSION=1536,                                              # 可选，默认 1536
    HIGH_RISK_THRESHOLD=0.85                                             # 可选，默认 0.85
)
```

`EMBEDDING_DIMENSION` 在**单个库实例 / 单个部署**中应保持固定，并与数据库建表时使用的维度一致；当前库不支持同一套表混用多个向量维度。

## 快速开始

### 1. 初始化数据库

```bash
docker-compose up -d db
uv run python scripts/init_db.py \
  --db-url postgresql+asyncpg://user:pass@localhost:5432/antifraud \
  --embedding-dimension 1536
```

`docker-compose.yml` 仅用于启动本地 PostgreSQL/pgvector 开发数据库。

### 2. 使用 AntiFraudRAG

```python
from antifraud_rag import AntiFraudRAG, Settings
from antifraud_rag.db.session import get_session, init_engine

# 初始化配置（必需）
settings = Settings(
    EMBEDDING_MODEL_URL="https://api.example.com/v1/embeddings",
    EMBEDDING_MODEL_API_KEY="your-key",
    DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/antifraud"
)

# 初始化数据库引擎
init_engine(settings)

# 创建 RAG 实例（settings 为必需参数）
async with get_session() as db:
    rag = AntiFraudRAG(db, settings=settings)
    
    # 分析可疑文本
    result = await rag.analyze("对方声称是公安局，说我涉嫌洗钱...")
    
    # 添加案例到知识库
    case = await rag.add_case(
        description="冒充客服诈骗：对方声称支付宝客服...",
        fraud_type="冒充客服",
        amount=50000,
        keywords=["客服", "退款", "银行卡"]
    )
    
    # 添加反诈知识
    tip = await rag.add_tip(
        title="如何识别冒充公检法诈骗",
        content="公检法机关不会通过电话办案...",
        category="防骗指南",
        keywords=["公检法", "电话办案"]
    )
```

### 3. 检索示例

```python
# 向量搜索
similar_cases = await rag.search_similar_cases(
    query="对方让我转账到安全账户",
    limit=5
)

# 混合搜索（BM25 + 向量 + RRF）
results = await rag.hybrid_search(
    query="可疑的电话诈骗",
    limit=10
)
```

## 库接口

### `AntiFraudRAG` 类

#### `analyze(text: str) -> AnalysisResponse`

分析文本的欺诈风险。

返回类型:
- **Direct_Hit**: 高风险（RRF score ≥ 0.85），直接命中已知案例
- **RAG_Prompt**: 中低风险，返回 RAG prompt 供 LLM 分析

```python
result = await rag.analyze("可疑文本...")
# result.result_type: "Direct_Hit" 或 "RAG_Prompt"
# result.data: DirectHitData 或 RAGPromptData
```

#### `add_case(description, fraud_type=None, amount=None, keywords=None) -> Case`

添加案例到知识库。

```python
case = await rag.add_case(
    description="案例描述",
    fraud_type="电信诈骗",
    amount=10000,
    keywords=["关键词1", "关键词2"]
)
```

#### `add_tip(title, content, category=None, keywords=None) -> Tip`

添加反诈知识。

```python
tip = await rag.add_tip(
    title="标题",
    content="知识内容",
    category="分类",
    keywords=["关键词"]
)
```

#### `search_similar_cases(query, limit=5) -> List[Dict]`

向量搜索相似案例。

#### `hybrid_search(query, limit=10) -> List[Dict]`

混合搜索（BM25 + 向量 + RRF 融合）。

## 架构说明

### 系统架构图

```mermaid
graph TB
    subgraph Client["客户端"]
        A[用户代码]
    end

    subgraph Core["核心层"]
        B[AntiFraudRAG]
    end

    subgraph Services["服务层"]
        C[EmbeddingService]
        D[RetrievalService]
        E[PromptsBuilder]
    end

    subgraph Database["数据层"]
        F[(PostgreSQL<br/>+ pgvector)]
        G[cases_table]
        H[tips_table]
    end

    subgraph External["外部服务"]
        I[Embedding API]
    end

    A -->|analyze/add_case/add_tip| B
    B -->|get_embeddings| C
    C -->|HTTP请求| I
    I -->|向量| C
    B -->|search_bm25/search_vector/rrf_fusion| D
    D -->|SQL查询| F
    F --> G
    F --> H
    B -->|build_prompt| E

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style I fill:#ffebee
```

### 核心流程：analyze 方法

```mermaid
flowchart TD
    Start([用户调用 analyze]) --> A[获取查询文本 embedding]
    A --> B[并行执行 BM25 + 向量搜索]
    
    B --> C1[BM25 搜索<br/>PostgreSQL tsvector]
    B --> C2[向量搜索<br/>pgvector 余弦相似度]
    
    C1 --> D[RRF 融合排序<br/>k=60]
    C2 --> D
    
    D --> E{是否有结果?}
    
    E -->|无结果| F[搜索 Tips 知识库]
    F --> G[返回 RAG_Prompt<br/>risk_level=LOW]
    
    E -->|有结果| H[获取 Top1 分数]
    H --> I{score >= 0.85?}
    
    I -->|是| J[返回 Direct_Hit<br/>risk_level=HIGH]
    I -->|否| K[搜索 Tips 知识库]
    K --> L[返回 RAG_Prompt<br/>risk_level=MEDIUM/LOW]
    
    G --> End([返回 AnalysisResponse])
    J --> End
    L --> End

    style Start fill:#e1f5fe
    style End fill:#e8f5e9
    style J fill:#ffcdd2
    style G fill:#fff9c4
    style L fill:#fff9c4
```

### 混合检索机制

1. **BM25 检索**: PostgreSQL `tsvector` 全文搜索，精确关键词匹配
2. **向量检索**: pgvector 余弦相似度搜索，语义理解
3. **RRF 融合**: `score = Σ(1/(k+rank))`, k=60

### 数据库设计

**cases_table** (案例库):
- `description`: 案例描述全文
- `fraud_type`: 诈骗类型
- `amount`: 涉案金额
- `embedding`: 向量列，维度由 `EMBEDDING_DIMENSION` 决定（默认 1536）
- `content_tsv`: 全文搜索索引

**tips_table** (知识库):
- `title`: 知识标题
- `content`: 知识内容
- `embedding`: 向量列，维度由 `EMBEDDING_DIMENSION` 决定（默认 1536，基于 `title + content`）
- `content_tsv`: 全文搜索索引

### 关键实现

- **Tip embedding**: 使用 `title + " " + content`（与 Case 不同）
- **索引**: IVFFlat (lists=100 for cases, 50 for tips) + GIN
- **分词**: 英文 tokenizer (`to_tsvector('english', ...)`)

## 开发

### 测试

```bash
make test          # 运行 pytest
make test-cov      # 覆盖率报告
make lint          # ruff 检查
make fmt           # 格式化代码
make ci            # lint + test
```

### 项目结构

```
antifraud_rag/
├── main.py              # AntiFraudRAG 核心类
├── schemas.py           # Pydantic 响应模型
├── core/config.py       # Settings 配置
├── db/
│   ├── models.py        # Case, Tip SQLAlchemy 模型
│   └── session.py       # 异步会话管理
└── services/
    ├── retrieval.py     # BM25, 向量搜索, RRF 融合
    ├── embedding.py     # Embedding API 客户端
    └── prompts.py       # Prompt 构建辅助
```

## 依赖

核心依赖:
- `sqlalchemy[asyncio]>=2.0.35`
- `asyncpg>=0.30.0`
- `pgvector==0.2.5`
- `pydantic>=2.0`
- `httpx==0.27.0`

可选依赖:
- 开发工具: `uv sync --all-extras`

## 许可证

MIT License

## 参考

- [pgvector](https://github.com/pgvector/pgvector)
- [RRF 融合算法](https://dl.acm.org/doi/epdf/10.1145/1571941.1572114)

import { useState } from 'react'
import { SearchCode, Loader2, Zap, Database, AlertTriangle, Tag } from 'lucide-react'
import { searchCases, hybridSearch } from '../api/client'

const EXAMPLE_QUERIES = [
  '冒充客服退款诈骗',
  '公安局安全账户',
  '网络贷款先缴费',
  '投资理财高回报',
]

export default function Search() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [limit, setLimit] = useState(10)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const data =
        mode === 'hybrid'
          ? await hybridSearch(query, limit)
          : await searchCases(query, limit)
      setResults(data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || '搜索失败，请检查服务状态')
    } finally {
      setLoading(false)
    }
  }

  const getScore = (r) => (mode === 'hybrid' ? r.rrf_score : r.score)

  const getScoreWidth = (r) => {
    const score = getScore(r)
    if (score == null) return '0%'
    if (mode === 'hybrid') {
      return `${Math.min(score * 120, 100)}%`
    }
    return `${Math.min(score * 100, 100)}%`
  }

  return (
    <div className="p-8 max-w-3xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">混合搜索</h1>
        <p className="text-slate-400 mt-1 text-sm">
          在案例库中进行向量相似性搜索或 BM25 + 向量 + RRF 混合检索
        </p>
      </div>

      {/* Search controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-5">
        {/* Mode toggle */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setMode('hybrid')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'hybrid'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            <Zap size={14} />
            混合搜索（BM25 + 向量 + RRF）
          </button>
          <button
            onClick={() => setMode('vector')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'vector'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            <Database size={14} />
            向量搜索
          </button>
        </div>

        {/* Search input row */}
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="输入搜索关键词..."
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <select
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value))}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          >
            <option value={5}>前 5 条</option>
            <option value={10}>前 10 条</option>
            <option value={20}>前 20 条</option>
          </select>
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <SearchCode size={16} />
            )}
            搜索
          </button>
        </div>

        {/* Example queries */}
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map((q, i) => (
            <button
              key={i}
              onClick={() => setQuery(q)}
              className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 px-3 py-1 rounded-full border border-slate-700 transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Mode description */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 mb-5 text-xs text-slate-500 leading-relaxed">
        {mode === 'hybrid' ? (
          <>
            <span className="text-blue-400 font-medium">混合搜索</span>：BM25 精确关键词匹配 +
            pgvector 余弦相似度搜索，通过 RRF 融合排序（score = Σ 1/(k+rank)，k=60）提升召回精度
          </>
        ) : (
          <>
            <span className="text-blue-400 font-medium">向量搜索</span>：使用 pgvector 计算查询文本与案例库的余弦相似度，返回语义最相近的案例
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-4 flex items-start gap-3 animate-fade-in-up">
          <AlertTriangle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {results !== null && (
        <div className="animate-fade-in-up">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
              搜索结果
            </div>
            <span className="text-xs text-slate-600">{results.length} 条</span>
          </div>

          {results.length === 0 ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-10 text-center">
              <SearchCode size={32} className="text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">未找到相关案例</p>
              <p className="text-slate-600 text-xs mt-1">请确认知识库中已录入案例数据</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((r, i) => {
                const score = getScore(r)
                return (
                  <div
                    key={i}
                    className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl p-5 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs text-slate-600 font-mono w-5">
                          #{i + 1}
                        </span>
                        {r.fraud_type && (
                          <span className="flex items-center gap-1 bg-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded font-medium">
                            <Tag size={10} />
                            {r.fraud_type}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500">
                          {mode === 'hybrid' ? 'RRF' : 'Score'}
                        </span>
                        <span className="text-xs font-mono text-blue-400 font-semibold">
                          {score?.toFixed(4) ?? '—'}
                        </span>
                      </div>
                    </div>

                    {/* Score bar */}
                    <div className="w-full bg-slate-800 rounded-full h-1 mb-3">
                      <div
                        className="bg-blue-500 h-1 rounded-full transition-all"
                        style={{ width: getScoreWidth(r) }}
                      />
                    </div>

                    <p className="text-slate-300 text-sm leading-relaxed">{r.description}</p>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

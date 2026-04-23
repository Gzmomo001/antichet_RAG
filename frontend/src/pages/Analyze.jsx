import { useState } from 'react'
import {
  ScanText,
  AlertTriangle,
  Info,
  Loader2,
  ChevronDown,
  Copy,
  Check,
  Lightbulb,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react'
import { analyzeText } from '../api/client'
import RiskBadge from '../components/RiskBadge'

const EXAMPLES = [
  '对方声称是公安局，说我涉嫌洗钱，要求我把钱转到安全账户配合调查',
  '某网站声称我中奖了，但需要先缴纳手续税款5000元才能领取50万奖金',
  '有人让我帮忙用我的银行卡转账，说给我一定好处费，我不知道这是否违法',
  '贷款平台让我先缴纳保证金1000元才能放款，说会一起返还',
]

export default function Analyze() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [promptExpanded, setPromptExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleAnalyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setPromptExpanded(false)
    try {
      const data = await analyzeText(text)
      setResult(data)
    } catch (e) {
      setError(
        e.response?.data?.detail || e.message || '分析失败，请检查 API 服务是否正常运行'
      )
    } finally {
      setLoading(false)
    }
  }

  const handleCopyPrompt = (prompt) => {
    navigator.clipboard.writeText(prompt)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="p-8 max-w-3xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">文本分析</h1>
        <p className="text-slate-400 mt-1 text-sm">
          输入可疑文本，通过 BM25 + 向量搜索 + RRF 融合进行欺诈风险评估
        </p>
      </div>

      {/* Input card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-5">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
          待分析文本
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="请输入需要分析的可疑文本内容..."
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm leading-relaxed"
          rows={5}
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key === 'Enter') handleAnalyze()
          }}
        />

        {/* Example texts */}
        <div className="mt-3 mb-4">
          <p className="text-xs text-slate-600 mb-2">示例文本（点击填入）：</p>
          <div className="flex flex-col gap-1">
            {EXAMPLES.map((t, i) => (
              <button
                key={i}
                onClick={() => setText(t)}
                className="text-left text-xs text-slate-600 hover:text-blue-400 transition-colors line-clamp-1"
              >
                · {t}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <ScanText size={16} />
            )}
            {loading ? '分析中...' : '开始分析'}
          </button>
          <span className="text-xs text-slate-600">Ctrl + Enter 快捷分析</span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-5 flex items-start gap-3 animate-fade-in-up">
          <AlertTriangle size={18} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="animate-fade-in-up">
          {result.result_type === 'Direct_Hit' ? (
            <DirectHitResult data={result.data} />
          ) : (
            <RAGPromptResult
              data={result.data}
              promptExpanded={promptExpanded}
              setPromptExpanded={setPromptExpanded}
              onCopyPrompt={handleCopyPrompt}
              copied={copied}
            />
          )}
        </div>
      )}
    </div>
  )
}

function DirectHitResult({ data }) {
  return (
    <div className="space-y-4">
      {/* High risk alert banner */}
      <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 animate-pulse-border">
        <div className="flex items-start gap-3">
          <ShieldAlert size={22} className="text-red-400 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-red-400 font-bold text-base">⚠️ 高风险告警 — 直接命中</h3>
              <RiskBadge level="HIGH" size="md" />
            </div>
            <p className="text-red-300 text-sm font-medium">{data.recommended_action}</p>
            <p className="text-slate-500 text-xs mt-1">
              已在历史案例库中匹配到高度相似案例（RRF score ≥ 0.85）
            </p>
          </div>
        </div>
      </div>

      {/* Matched cases */}
      {data.matched_cases?.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
            命中案例 ({data.matched_cases.length})
          </div>
          <div className="space-y-4">
            {data.matched_cases.map((c, i) => (
              <div key={i} className="border-l-2 border-red-500/40 pl-4">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  {c.fraud_type && (
                    <span className="bg-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded font-medium">
                      {c.fraud_type}
                    </span>
                  )}
                  <span className="text-xs text-slate-500">
                    置信度 {(c.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {/* Confidence bar */}
                <div className="w-full bg-slate-800 rounded-full h-1.5 mb-2">
                  <div
                    className="bg-red-500 h-1.5 rounded-full transition-all"
                    style={{ width: `${Math.min(c.confidence * 100, 100)}%` }}
                  />
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">{c.description}</p>
                {c.key_indicators?.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {c.key_indicators.map((k, j) => (
                      <span
                        key={j}
                        className="bg-red-500/10 text-red-400 text-xs px-2 py-0.5 rounded border border-red-500/20"
                      >
                        {k}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function RAGPromptResult({ data, promptExpanded, setPromptExpanded, onCopyPrompt, copied }) {
  const riskColorMap = {
    HIGH: 'text-red-400',
    MEDIUM: 'text-amber-400',
    LOW: 'text-green-400',
  }
  const riskColor = riskColorMap[data.risk_level] || 'text-green-400'
  const riskIcon = data.risk_level === 'LOW' ? ShieldCheck : Info

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {data.risk_level === 'LOW' ? (
              <ShieldCheck size={20} className="text-green-400" />
            ) : (
              <Info size={20} className="text-amber-400" />
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-white font-semibold text-sm">RAG 分析结果</span>
                <RiskBadge level={data.risk_level} size="md" />
              </div>
              <p className="text-slate-500 text-xs mt-0.5">
                RRF 融合分数：
                <span className="font-mono text-slate-300">{data.rrf_score?.toFixed(4) ?? '—'}</span>
                <span className="ml-2">
                  (阈值 0.85 — 未直接命中，建议进一步 LLM 分析)
                </span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Relevant cases */}
      {data.context?.relevant_cases?.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
            相关历史案例 ({data.context.relevant_cases.length})
          </div>
          <div className="space-y-3">
            {data.context.relevant_cases.map((c, i) => (
              <div key={i} className="border-l-2 border-amber-500/40 pl-4">
                {c.fraud_type && (
                  <span className="inline-block bg-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded mb-1.5">
                    {c.fraud_type}
                  </span>
                )}
                <p className="text-slate-300 text-sm leading-relaxed">{c.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Anti-fraud tips */}
      {data.context?.anti_fraud_tips?.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
            <span className="flex items-center gap-2">
              <Lightbulb size={13} className="text-green-400" />
              反诈知识提示
            </span>
          </div>
          <div className="space-y-3">
            {data.context.anti_fraud_tips.map((t, i) => (
              <div key={i} className="border-l-2 border-green-500/40 pl-4">
                <p className="text-green-400 text-sm font-medium">{t.title}</p>
                <p className="text-slate-400 text-xs mt-0.5 leading-relaxed">{t.content}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* RAG Prompt */}
      {data.prompt && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
          <button
            onClick={() => setPromptExpanded(!promptExpanded)}
            className="w-full flex items-center justify-between px-5 py-4 hover:bg-slate-800 transition-colors"
          >
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
              RAG Prompt（供 LLM 使用）
            </span>
            <ChevronDown
              size={16}
              className={`text-slate-500 transition-transform duration-200 ${promptExpanded ? 'rotate-180' : ''}`}
            />
          </button>
          {promptExpanded && (
            <div className="px-5 pb-5">
              <div className="relative">
                <pre className="bg-slate-800 border border-slate-700 rounded-lg p-4 text-xs text-slate-300 whitespace-pre-wrap overflow-x-auto leading-relaxed font-mono">
                  {data.prompt}
                </pre>
                <button
                  onClick={() => onCopyPrompt(data.prompt)}
                  className="absolute top-2 right-2 flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 px-2.5 py-1 rounded text-xs transition-colors"
                >
                  {copied ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                  {copied ? '已复制' : '复制'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

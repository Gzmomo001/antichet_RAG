import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  ScanText,
  FileStack,
  BookMarked,
  SearchCode,
  CheckCircle2,
  XCircle,
  AlertCircle,
  ArrowRight,
  Cpu,
  Database,
  Layers,
  GitMerge,
} from 'lucide-react'
import { getHealth } from '../api/client'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [healthLoading, setHealthLoading] = useState(true)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))
      .finally(() => setHealthLoading(false))
  }, [])

  const features = [
    {
      icon: ScanText,
      title: '文本分析',
      description: '输入可疑文本，使用混合 RAG 检索进行欺诈风险评估，输出高/中/低风险分级结果',
      to: '/analyze',
      color: 'text-blue-400',
      border: 'border-blue-500/20',
      bg: 'hover:bg-blue-500/5',
      badge: '核心功能',
      badgeColor: 'bg-blue-500/15 text-blue-400',
    },
    {
      icon: FileStack,
      title: '案例库管理',
      description: '添加历史诈骗案例，系统自动生成向量嵌入并存入 PostgreSQL/pgvector',
      to: '/cases',
      color: 'text-purple-400',
      border: 'border-purple-500/20',
      bg: 'hover:bg-purple-500/5',
      badge: '知识入库',
      badgeColor: 'bg-purple-500/15 text-purple-400',
    },
    {
      icon: BookMarked,
      title: '反诈知识库',
      description: '管理防骗技巧文档，检索时自动匹配并加入 RAG Prompt 上下文',
      to: '/tips',
      color: 'text-green-400',
      border: 'border-green-500/20',
      bg: 'hover:bg-green-500/5',
      badge: '知识入库',
      badgeColor: 'bg-green-500/15 text-green-400',
    },
    {
      icon: SearchCode,
      title: '混合搜索',
      description: 'BM25 精确匹配 + 向量语义搜索，通过 RRF 融合排序提升检索精度',
      to: '/search',
      color: 'text-amber-400',
      border: 'border-amber-500/20',
      bg: 'hover:bg-amber-500/5',
      badge: '调试工具',
      badgeColor: 'bg-amber-500/15 text-amber-400',
    },
  ]

  const pipeline = [
    { icon: ScanText, label: '文本输入', color: 'text-blue-400', bg: 'bg-blue-500/10' },
    { icon: Cpu, label: 'Embedding', color: 'text-purple-400', bg: 'bg-purple-500/10' },
    { icon: Database, label: 'BM25 + 向量检索', color: 'text-green-400', bg: 'bg-green-500/10' },
    { icon: Layers, label: 'RRF 融合 (k=60)', color: 'text-amber-400', bg: 'bg-amber-500/10' },
    { icon: GitMerge, label: '风险评估输出', color: 'text-red-400', bg: 'bg-red-500/10' },
  ]

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">仪表盘</h1>
        <p className="text-slate-400 mt-1 text-sm">
          反欺诈 RAG 系统 — 基于混合检索的欺诈信息智能识别平台
        </p>
      </div>

      {/* System status */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
          系统状态
        </div>
        {healthLoading ? (
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <span className="w-2 h-2 rounded-full bg-slate-600 animate-pulse" />
            正在连接 API...
          </div>
        ) : health?.status === 'ok' ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-sm">
              <CheckCircle2 size={16} className="text-green-400" />
              <span className="text-green-400 font-medium">API 服务正常</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              {health?.engine_initialized ? (
                <>
                  <CheckCircle2 size={16} className="text-green-400" />
                  <span className="text-slate-300">数据库引擎已初始化</span>
                </>
              ) : (
                <>
                  <AlertCircle size={16} className="text-amber-400" />
                  <span className="text-amber-400">
                    数据库未连接 — 请配置 .env 并重启 API 服务
                  </span>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm">
            <XCircle size={16} className="text-red-400" />
            <span className="text-red-400 font-medium">
              无法连接 API — 请启动后端服务（uvicorn api.main:app --reload）
            </span>
          </div>
        )}
      </div>

      {/* Pipeline visualization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
          分析流水线
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          {pipeline.map(({ icon: Icon, label, color, bg }, i) => (
            <div key={i} className="flex items-center gap-1">
              <div className={`flex items-center gap-2 ${bg} rounded-lg px-3 py-1.5`}>
                <Icon size={14} className={color} />
                <span className={`text-xs font-medium ${color}`}>{label}</span>
              </div>
              {i < pipeline.length - 1 && (
                <ArrowRight size={12} className="text-slate-600 shrink-0" />
              )}
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-slate-600">
          RRF score ≥ 0.85 → Direct Hit（高危告警）&nbsp;|&nbsp; RRF score &lt; 0.85 → RAG Prompt（供 LLM 分析）
        </div>
      </div>

      {/* Feature cards */}
      <div className="grid grid-cols-2 gap-4">
        {features.map(({ icon: Icon, title, description, to, color, border, bg, badge, badgeColor }) => (
          <Link
            key={to}
            to={to}
            className={`bg-slate-900 border ${border} rounded-xl p-5 ${bg} transition-colors group`}
          >
            <div className="flex items-start justify-between mb-3">
              <Icon size={22} className={color} />
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeColor}`}>
                {badge}
              </span>
            </div>
            <h3 className="text-white font-semibold text-sm mb-1.5 group-hover:text-blue-300 transition-colors">
              {title}
            </h3>
            <p className="text-slate-500 text-xs leading-relaxed">{description}</p>
            <div className="flex items-center gap-1 mt-3 text-xs text-slate-600 group-hover:text-blue-400 transition-colors">
              <span>进入</span>
              <ArrowRight size={12} />
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

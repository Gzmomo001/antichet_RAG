import { useState } from 'react'
import { Plus, BookOpen, Tag, Loader2, CheckCircle2, AlertTriangle, Hash, FileText } from 'lucide-react'
import { addTip } from '../api/client'

const CATEGORIES = [
  '防骗指南',
  '识别技巧',
  '应急措施',
  '典型手法',
  '维权途径',
  '案例解析',
  '官方提示',
]

const EXAMPLE_TIPS = [
  {
    title: '如何识别冒充公检法诈骗',
    content:
      '公检法机关不会通过电话办案，不会要求当事人将资金转入"安全账户"。接到此类电话请立即挂断并拨打110核实。',
    category: '防骗指南',
    keywords: '公检法 电话办案 安全账户',
  },
  {
    title: '网络贷款诈骗识别要点',
    content:
      '正规贷款机构不会要求借款人先缴纳保证金、手续费或解冻金。要求先付费才能放款的均为诈骗。',
    category: '识别技巧',
    keywords: '贷款 保证金 手续费 先付费',
  },
  {
    title: '遭遇诈骗的紧急处置',
    content:
      '发现被骗后应立即联系银行冻结账户，同时拨打110报警，保留所有聊天记录、转账凭证作为证据。',
    category: '应急措施',
    keywords: '冻结账户 报警 证据',
  },
]

const INITIAL_FORM = { title: '', content: '', category: '', keywords: '' }

export default function Tips() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.title.trim() || !form.content.trim()) return
    setLoading(true)
    setError(null)
    setSuccess(null)
    try {
      const keywords = form.keywords
        ? form.keywords.split(/[,，\s]+/).filter(Boolean)
        : undefined
      const result = await addTip({
        title: form.title,
        content: form.content,
        category: form.category || undefined,
        keywords,
      })
      setSuccess(result)
      setForm(INITIAL_FORM)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || '添加失败，请检查服务状态')
    } finally {
      setLoading(false)
    }
  }

  const fillExample = (ex) => {
    setForm({
      title: ex.title,
      content: ex.content,
      category: ex.category,
      keywords: ex.keywords,
    })
    setSuccess(null)
    setError(null)
  }

  return (
    <div className="p-8 max-w-2xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">知识库</h1>
        <p className="text-slate-400 mt-1 text-sm">
          添加反诈防骗知识，在 RAG 分析时自动匹配并加入 LLM 上下文
        </p>
      </div>

      {/* Example quick-fill */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 mb-5">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
          示例知识（点击快速填入）
        </div>
        <div className="flex flex-col gap-2">
          {EXAMPLE_TIPS.map((ex, i) => (
            <button
              key={i}
              onClick={() => fillExample(ex)}
              className="text-left text-xs text-slate-500 hover:text-blue-400 border border-slate-800 hover:border-blue-500/30 rounded-lg px-3 py-2 transition-all"
            >
              <span className="text-green-400 font-medium">[{ex.category}]</span>{' '}
              {ex.title}
            </button>
          ))}
        </div>
      </div>

      {/* Form */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-5">
          添加新知识
        </div>
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Title */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <FileText size={14} className="text-slate-500" />
              知识标题
              <span className="text-red-400 ml-0.5">*</span>
            </label>
            <input
              type="text"
              value={form.title}
              onChange={(e) => setForm({ ...form, title: e.target.value })}
              placeholder="如：如何识别冒充公检法诈骗"
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              required
            />
          </div>

          {/* Content */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <BookOpen size={14} className="text-slate-500" />
              知识内容
              <span className="text-red-400 ml-0.5">*</span>
            </label>
            <textarea
              value={form.content}
              onChange={(e) => setForm({ ...form, content: e.target.value })}
              placeholder="详细描述防骗知识，将作为 RAG 上下文提供给 LLM..."
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm leading-relaxed"
              rows={5}
              required
            />
          </div>

          {/* Category */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <Tag size={14} className="text-slate-500" />
              知识分类
            </label>
            <select
              value={form.category}
              onChange={(e) => setForm({ ...form, category: e.target.value })}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            >
              <option value="">— 请选择 —</option>
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          {/* Keywords */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <Hash size={14} className="text-slate-500" />
              关键词
              <span className="text-slate-500 text-xs font-normal ml-1">（逗号或空格分隔）</span>
            </label>
            <input
              type="text"
              value={form.keywords}
              onChange={(e) => setForm({ ...form, keywords: e.target.value })}
              placeholder="如：公检法 电话办案 安全账户"
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            />
          </div>

          <button
            type="submit"
            disabled={loading || !form.title.trim() || !form.content.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
            {loading ? '添加中...' : '添加知识'}
          </button>
        </form>
      </div>

      {/* Success */}
      {success && (
        <div className="mt-4 bg-green-500/10 border border-green-500/30 rounded-xl p-4 flex items-start gap-3 animate-fade-in-up">
          <CheckCircle2 size={18} className="text-green-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-green-400 font-medium text-sm">知识添加成功</p>
            <p className="text-slate-300 text-xs font-medium mt-1">{success.title}</p>
            <p className="text-slate-400 text-xs font-mono">ID: {success.id}</p>
            {success.category && (
              <p className="text-slate-400 text-xs">分类: {success.category}</p>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3 animate-fade-in-up">
          <AlertTriangle size={18} className="text-red-400 shrink-0 mt-0.5" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}
    </div>
  )
}

import { useState } from 'react'
import { Plus, Tag, DollarSign, FileText, Loader2, CheckCircle2, AlertTriangle, Hash } from 'lucide-react'
import { addCase } from '../api/client'

const FRAUD_TYPES = [
  '冒充客服',
  '冒充公检法',
  '电信诈骗',
  '网络贷款诈骗',
  '投资理财诈骗',
  '网络购物诈骗',
  '网络刷单诈骗',
  '情感诈骗',
  '中奖诈骗',
  '杀猪盘诈骗',
  '虚假兼职诈骗',
  '其他诈骗',
]

const EXAMPLE_CASES = [
  {
    description: '对方自称支付宝客服，以账号异常需要退款为由，要求提供银行卡号和验证码，随后盗刷账户资金5万元',
    fraud_type: '冒充客服',
    amount: 50000,
    keywords: '客服 退款 银行卡 验证码',
  },
  {
    description: '接到"公安局"电话，对方称涉及洗钱案件，要求配合调查将资金转入"安全账户"，实为诈骗',
    fraud_type: '冒充公检法',
    amount: 200000,
    keywords: '公安局 洗钱 安全账户 配合调查',
  },
]

const INITIAL_FORM = { description: '', fraud_type: '', amount: '', keywords: '' }

export default function Cases() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.description.trim()) return
    setLoading(true)
    setError(null)
    setSuccess(null)
    try {
      const keywords = form.keywords
        ? form.keywords.split(/[,，\s]+/).filter(Boolean)
        : undefined
      const result = await addCase({
        description: form.description,
        fraud_type: form.fraud_type || undefined,
        amount: form.amount ? parseFloat(form.amount) : undefined,
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
      description: ex.description,
      fraud_type: ex.fraud_type,
      amount: String(ex.amount),
      keywords: ex.keywords,
    })
    setSuccess(null)
    setError(null)
  }

  return (
    <div className="p-8 max-w-2xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">案例库</h1>
        <p className="text-slate-400 mt-1 text-sm">
          向向量知识库中添加历史诈骗案例，系统自动生成 Embedding 并建立索引
        </p>
      </div>

      {/* Example quick-fill */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 mb-5">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
          示例案例（点击快速填入）
        </div>
        <div className="flex flex-col gap-2">
          {EXAMPLE_CASES.map((ex, i) => (
            <button
              key={i}
              onClick={() => fillExample(ex)}
              className="text-left text-xs text-slate-500 hover:text-blue-400 border border-slate-800 hover:border-blue-500/30 rounded-lg px-3 py-2 transition-all"
            >
              <span className="text-slate-400 font-medium">[{ex.fraud_type}]</span>{' '}
              {ex.description.slice(0, 60)}...
            </button>
          ))}
        </div>
      </div>

      {/* Form */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-5">
          添加新案例
        </div>
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Description */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <FileText size={14} className="text-slate-500" />
              案例描述
              <span className="text-red-400 ml-0.5">*</span>
            </label>
            <textarea
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
              placeholder="详细描述诈骗经过：对方身份、手段、涉及金额等..."
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm leading-relaxed"
              rows={4}
              required
            />
          </div>

          {/* Fraud type */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <Tag size={14} className="text-slate-500" />
              诈骗类型
            </label>
            <select
              value={form.fraud_type}
              onChange={(e) => setForm({ ...form, fraud_type: e.target.value })}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            >
              <option value="">— 请选择 —</option>
              {FRAUD_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          {/* Amount */}
          <div>
            <label className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-2">
              <DollarSign size={14} className="text-slate-500" />
              涉案金额（元）
            </label>
            <input
              type="number"
              min="0"
              value={form.amount}
              onChange={(e) => setForm({ ...form, amount: e.target.value })}
              placeholder="如：50000"
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            />
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
              placeholder="如：客服 退款 银行卡"
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            />
          </div>

          <button
            type="submit"
            disabled={loading || !form.description.trim()}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
            {loading ? '添加中...' : '添加案例'}
          </button>
        </form>
      </div>

      {/* Success */}
      {success && (
        <div className="mt-4 bg-green-500/10 border border-green-500/30 rounded-xl p-4 flex items-start gap-3 animate-fade-in-up">
          <CheckCircle2 size={18} className="text-green-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-green-400 font-medium text-sm">案例添加成功</p>
            <p className="text-slate-400 text-xs mt-1 font-mono">ID: {success.id}</p>
            {success.fraud_type && (
              <p className="text-slate-400 text-xs">类型: {success.fraud_type}</p>
            )}
            <p className="text-slate-500 text-xs mt-1 line-clamp-2">{success.description}</p>
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

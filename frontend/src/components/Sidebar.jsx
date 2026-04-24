import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  ScanText,
  FileStack,
  BookMarked,
  SearchCode,
  Shield,
} from 'lucide-react'

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: '仪表盘', desc: 'Dashboard' },
  { to: '/analyze', icon: ScanText, label: '文本分析', desc: 'Analyze' },
  { to: '/cases', icon: FileStack, label: '案例库', desc: 'Cases' },
  { to: '/tips', icon: BookMarked, label: '知识库', desc: 'Knowledge' },
  { to: '/search', icon: SearchCode, label: '混合搜索', desc: 'Hybrid Search' },
]

export default function Sidebar() {
  return (
    <aside className="w-60 shrink-0 bg-slate-900 flex flex-col border-r border-slate-800">
      {/* Logo */}
      <div className="px-5 py-6 flex items-center gap-3 border-b border-slate-800">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center shrink-0">
          <Shield size={20} className="text-white" />
        </div>
        <div className="min-w-0">
          <div className="text-white font-bold text-base leading-tight truncate">反欺诈 RAG</div>
          <div className="text-slate-500 text-xs mt-0.5">AntiCheat RAG System</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={17} className={isActive ? 'text-white' : 'text-slate-500 group-hover:text-slate-300'} />
                <span>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-800">
        <div className="text-xs text-slate-600 leading-relaxed">
          <div className="font-medium text-slate-500 mb-1">检索策略</div>
          <div>BM25 + 向量搜索</div>
          <div>RRF 融合排序 (k=60)</div>
        </div>
      </div>
    </aside>
  )
}

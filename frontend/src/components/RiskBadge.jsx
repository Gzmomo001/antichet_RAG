const RISK_CONFIG = {
  HIGH: {
    label: '高风险',
    className: 'bg-red-500/15 text-red-400 border border-red-500/30',
    dot: 'bg-red-400',
  },
  MEDIUM: {
    label: '中风险',
    className: 'bg-amber-500/15 text-amber-400 border border-amber-500/30',
    dot: 'bg-amber-400',
  },
  LOW: {
    label: '低风险',
    className: 'bg-green-500/15 text-green-400 border border-green-500/30',
    dot: 'bg-green-400',
  },
}

export default function RiskBadge({ level, size = 'md' }) {
  const config = RISK_CONFIG[level] || RISK_CONFIG.LOW
  const sizeClass =
    size === 'lg'
      ? 'px-3 py-1.5 text-sm font-semibold gap-2'
      : 'px-2.5 py-1 text-xs font-medium gap-1.5'

  return (
    <span className={`inline-flex items-center rounded-full ${sizeClass} ${config.className}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot} shrink-0`} />
      {config.label}
    </span>
  )
}

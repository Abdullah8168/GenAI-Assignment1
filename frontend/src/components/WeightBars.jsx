// Horizontal bars for classifier probabilities (hard routing) or gate weights
// (soft MoE). Bars are plain divs whose width is the percentage.
import { EXPERTS, pct } from '../utils.js';

export function WeightBars({ values, highlight, labels }) {
  if (!values) return null;
  return (
    <div className="space-y-2.5">
      {EXPERTS.map((e) => {
        const v = values[e.key] ?? 0;
        const isTop = e.key === highlight;
        return (
          <div key={e.key}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className={`flex items-center gap-1.5 ${isTop ? 'font-semibold text-slate-900' : 'text-slate-600'}`}>
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: e.color }} />
                {labels?.[e.key] || e.label}
                {isTop && <span className="rounded bg-indigo-100 px-1.5 text-[10px] text-indigo-700">TOP</span>}
              </span>
              <span className="font-mono text-slate-700">{pct(v)}</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%`, background: e.color, opacity: isTop ? 1 : 0.55 }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** A single 100 % bar split into coloured segments proportional to each weight. */
export function StackedBar({ values }) {
  if (!values) return null;
  const total = EXPERTS.reduce((s, e) => s + (values[e.key] ?? 0), 0) || 1;
  return (
    <div>
      <div className="flex h-6 w-full overflow-hidden rounded-lg ring-1 ring-slate-200">
        {EXPERTS.map((e) => {
          const share = (values[e.key] ?? 0) / total;
          return (
            <div
              key={e.key}
              title={`${e.label}: ${pct(share)}`}
              className="flex items-center justify-center text-[10px] font-semibold text-white transition-all duration-500"
              style={{ width: `${share * 100}%`, background: e.color }}
            >
              {share > 0.08 ? pct(share) : ''}
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-600">
        {EXPERTS.map((e) => (
          <span key={e.key} className="flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: e.color }} />
            {e.short}
          </span>
        ))}
      </div>
    </div>
  );
}

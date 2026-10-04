// Left sidebar on desktop (md and up); on phones it becomes a top bar with a
// horizontally scrollable navigation row. The footer shows a live health dot.
import { FlaskConical } from 'lucide-react';

/** Coloured dot + text describing backend health. */
function HealthIndicator({ health, healthError, compact = false }) {
  let color = 'bg-slate-400';
  let text = 'Checking…';
  if (healthError) {
    color = 'bg-red-500';
    text = 'Backend offline';
  } else if (health?.status === 'ok') {
    color = 'bg-emerald-500';
    text = 'Backend healthy';
  } else if (health) {
    color = 'bg-amber-500';
    text = `Backend ${health.status || 'degraded'}`;
  }
  return (
    <div className="flex items-center gap-2 text-xs text-slate-300" title={healthError || text}>
      <span className="relative flex h-2.5 w-2.5">
        {health?.status === 'ok' && (
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${color} opacity-60`} />
        )}
        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${color}`} />
      </span>
      {!compact && <span>{text}</span>}
    </div>
  );
}

export default function Sidebar({ views, current, onSelect, health, healthError }) {
  return (
    <>
      {/* ---------- Desktop sidebar ---------- */}
      <aside className="fixed inset-y-0 left-0 hidden w-72 flex-col bg-slate-900 text-slate-100 md:flex">
        <div className="flex items-center gap-3 px-6 py-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600">
            <FlaskConical className="h-5 w-5" />
          </div>
          <div>
            <div className="text-lg font-semibold leading-tight">RestoreLab</div>
            <div className="text-xs text-slate-400">Generative AI Studio</div>
          </div>
        </div>

        <nav className="flex-1 space-y-1 px-3">
          {views.map((v) => {
            const Icon = v.icon;
            const active = v.id === current;
            return (
              <button
                key={v.id}
                onClick={() => onSelect(v.id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${
                  active ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className="flex-1 leading-snug">{v.label}</span>
                <span className={`text-[10px] uppercase ${active ? 'text-indigo-200' : 'text-slate-500'}`}>
                  {v.task}
                </span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-slate-800 px-6 py-4">
          <HealthIndicator health={health} healthError={healthError} />
          <div className="mt-1 text-[11px] text-slate-500">Polled every 15 s</div>
        </div>
      </aside>

      {/* ---------- Mobile top bar ---------- */}
      <header className="sticky top-0 z-20 bg-slate-900 text-slate-100 md:hidden">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600">
              <FlaskConical className="h-4 w-4" />
            </div>
            <span className="font-semibold">RestoreLab</span>
          </div>
          <HealthIndicator health={health} healthError={healthError} />
        </div>
        <nav className="flex gap-1 overflow-x-auto px-2 pb-2">
          {views.map((v) => {
            const Icon = v.icon;
            const active = v.id === current;
            return (
              <button
                key={v.id}
                onClick={() => onSelect(v.id)}
                className={`flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs ${
                  active ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {v.label}
              </button>
            );
          })}
        </nav>
      </header>
    </>
  );
}

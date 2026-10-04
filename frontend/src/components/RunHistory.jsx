// Last few runs of this session (kept in memory only). Clicking an entry shows it again.
import { History } from 'lucide-react';

export default function RunHistory({ items, onSelect, activeId }) {
  return (
    <div className="card">
      <h3 className="card-title mb-3 flex items-center gap-2">
        <History className="h-4 w-4" /> Run history
      </h3>
      {items.length === 0 ? (
        <p className="text-xs text-slate-500">No runs yet in this session.</p>
      ) : (
        <ul className="space-y-2">
          {items.map((h) => (
            <li key={h.id}>
              <button
                onClick={() => onSelect?.(h)}
                className={`flex w-full items-center gap-3 rounded-xl p-2 text-left transition hover:bg-slate-50 ${
                  h.id === activeId ? 'bg-indigo-50 ring-1 ring-indigo-200' : ''
                }`}
              >
                <img src={h.thumb} alt="" className="h-10 w-10 rounded-lg object-cover ring-1 ring-slate-200" />
                <div className="min-w-0 flex-1 text-xs">
                  <div className="truncate font-medium text-slate-700">{h.summary}</div>
                  <div className="text-slate-400">
                    {h.time} · {h.label}
                  </div>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

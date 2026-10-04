// System status: renders GET /api/health and links to the MLflow tracking UI.
import { CheckCircle2, XCircle, ExternalLink, RefreshCw, Cpu } from 'lucide-react';
import { ErrorAlert, PageHeader } from '../components/ui.jsx';

const MLFLOW_URL = 'http://localhost:5000';

function formatUptime(s) {
  if (s == null) return '—';
  const sec = Math.floor(s);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h ? `${h} h ${m} min` : m ? `${m} min ${sec % 60} s` : `${sec} s`;
}

function Stat({ label, value }) {
  return (
    <div className="rounded-xl bg-slate-50 p-4 ring-1 ring-slate-200">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 break-words text-lg font-semibold text-slate-900">{value}</div>
    </div>
  );
}

export default function SystemView({ health, error, checkedAt, onRefresh }) {
  const models = Object.entries(health?.models || {});
  const status = error ? 'offline' : health?.status || 'unknown';
  const statusColor =
    status === 'ok' ? 'bg-emerald-100 text-emerald-800' : status === 'offline' ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800';

  return (
    <div>
      <PageHeader
        task="Status"
        title="System"
        right={
          <div className="flex flex-wrap gap-2">
            <button className="btn-secondary" onClick={onRefresh}>
              <RefreshCw className="h-4 w-4" /> Refresh
            </button>
            <a className="btn-primary" href={MLFLOW_URL} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4" /> Open experiment tracking (MLflow)
            </a>
          </div>
        }
      >
        Live status of the inference backend (GET /api/health), refreshed every 15 seconds.
      </PageHeader>

      <div className="space-y-6">
        <ErrorAlert message={error} />

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <div className="rounded-xl bg-slate-50 p-4 ring-1 ring-slate-200">
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Status</div>
            <span className={`mt-2 inline-block rounded-full px-3 py-1 text-sm font-semibold ${statusColor}`}>{status}</span>
          </div>
          <Stat label="onnxruntime" value={health?.onnxruntime ?? '—'} />
          <Stat label="Image size" value={health?.image_size ? `${health.image_size} × ${health.image_size}` : '—'} />
          <Stat label="Uptime" value={formatUptime(health?.uptime_s)} />
          <Stat label="Last check" value={checkedAt ? checkedAt.toLocaleTimeString() : '—'} />
        </div>

        <section className="card">
          <h2 className="card-title mb-3 flex items-center gap-2">
            <Cpu className="h-4 w-4" /> Execution providers
          </h2>
          <div className="flex flex-wrap gap-2">
            {(health?.providers || []).map((p) => (
              <span key={p} className="chip font-mono">
                {p}
              </span>
            ))}
            {!health?.providers?.length && <span className="text-sm text-slate-500">—</span>}
          </div>
        </section>

        <section className="card overflow-x-auto">
          <h2 className="card-title mb-3">ONNX models</h2>
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 pr-4">Loaded</th>
                <th className="py-2 pr-4">File</th>
                <th className="py-2 pr-4 text-right">Size</th>
                <th className="py-2 pr-4">Inputs</th>
                <th className="py-2">Outputs</th>
              </tr>
            </thead>
            <tbody>
              {models.map(([name, m]) => (
                <tr key={name} className="border-b border-slate-100 last:border-0">
                  <td className="py-2 pr-4 font-medium text-slate-800">{name}</td>
                  <td className="py-2 pr-4">
                    {m.loaded ? (
                      <span className="inline-flex items-center gap-1 text-emerald-700">
                        <CheckCircle2 className="h-4 w-4" /> yes
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-red-700" title={m.error || ''}>
                        <XCircle className="h-4 w-4" /> no
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-4 font-mono text-xs">{m.file || '—'}</td>
                  <td className="py-2 pr-4 text-right font-mono text-xs">
                    {m.size_mb != null ? `${Number(m.size_mb).toFixed(2)} MB` : '—'}
                  </td>
                  <td className="py-2 pr-4 font-mono text-xs">{(m.inputs || []).join(', ') || '—'}</td>
                  <td className="py-2 font-mono text-xs">{(m.outputs || []).join(', ') || '—'}</td>
                </tr>
              ))}
              {models.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-slate-500">
                    {error ? 'Backend unreachable.' : 'No model information yet.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}

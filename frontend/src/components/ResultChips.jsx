// Chip rows for metrics, corruption settings and inference timings.
import { ArrowRight, Clock, TrendingDown, TrendingUp } from 'lucide-react';
import { formatValue } from '../utils.js';

/** One metric: input -> output with a coloured delta (green if improved). */
function MetricChip({ name, input, output, unit = '', digits = 2 }) {
  if (input == null || output == null) return null;
  const delta = output - input;
  const improved = delta > 0;
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm ring-1 ${
        improved ? 'bg-emerald-50 text-emerald-800 ring-emerald-200' : 'bg-red-50 text-red-800 ring-red-200'
      }`}
    >
      <span className="font-semibold">{name}</span>
      <span className="font-mono">{input.toFixed(digits)}</span>
      <ArrowRight className="h-3.5 w-3.5" />
      <span className="font-mono font-semibold">
        {output.toFixed(digits)}
        {unit}
      </span>
      <span className="inline-flex items-center gap-0.5 font-mono text-xs">
        {improved ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
        {delta >= 0 ? '+' : ''}
        {delta.toFixed(digits)}
      </span>
    </div>
  );
}

export function MetricsRow({ metrics }) {
  if (!metrics) {
    return <p className="text-xs text-slate-500">No metrics — a clean reference is only available when the corruption is applied at runtime.</p>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      <MetricChip name="PSNR" input={metrics.psnr_input} output={metrics.psnr_output} unit=" dB" />
      <MetricChip name="SSIM" input={metrics.ssim_input} output={metrics.ssim_output} digits={3} />
    </div>
  );
}

/** Key/value chips: type, severity and every parameter (including rect coordinates). */
export function CorruptionChips({ corruption }) {
  if (!corruption) return <span className="chip">Input used as-is (no runtime corruption)</span>;
  const entries = [
    ['type', corruption.type],
    ['severity', corruption.severity],
    ...Object.entries(corruption.params || {}),
  ];
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([k, v]) => (
        <span key={k} className="chip">
          <span className="text-slate-500">{k}</span>
          <span className="font-mono text-slate-800">{formatValue(v)}</span>
        </span>
      ))}
    </div>
  );
}

/** One chip per key of inference_ms. */
export function TimingChips({ timings }) {
  if (!timings) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {Object.entries(timings).map(([k, v]) => (
        <span key={k} className={`chip ${k === 'total' ? 'bg-indigo-100 text-indigo-800' : ''}`}>
          <Clock className="h-3 w-3" />
          {k}: <span className="font-mono">{Number(v).toFixed(1)} ms</span>
        </span>
      ))}
    </div>
  );
}

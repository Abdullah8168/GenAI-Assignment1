// "Runtime corruption" controls: type, severity, custom parameter sliders, seed.
// The settings object is owned by the parent and updated via onChange(patch).
import { CORRUPTION_LABELS } from '../utils.js';

export const DEFAULT_CORRUPTION = {
  corruption: 'none', // none | clean | salt | blur | occlusion
  severity: 'medium', // low | medium | high | custom
  salt_p: 0.08,
  blur_k: 5,
  blur_sigma: 1.5,
  occ_coverage: 0.2,
  occ_rects: 2,
  seed: '',
};

/** Labelled range slider showing its current value. */
function Slider({ label, value, min, max, step, onChange }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-slate-600">
        <span>{label}</span>
        <span className="font-mono text-indigo-700">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-indigo-600"
      />
    </div>
  );
}

export default function CorruptionControls({ value, onChange, allowNone = true }) {
  const set = (patch) => onChange({ ...value, ...patch });
  const types = allowNone ? ['none', 'clean', 'salt', 'blur', 'occlusion'] : ['clean', 'salt', 'blur', 'occlusion'];
  // Severity only matters for an actual corruption.
  const hasSeverity = ['salt', 'blur', 'occlusion'].includes(value.corruption);
  const custom = hasSeverity && value.severity === 'custom';

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className="field-label">Corruption</label>
          <select className="select" value={value.corruption} onChange={(e) => set({ corruption: e.target.value })}>
            {types.map((t) => (
              <option key={t} value={t}>
                {CORRUPTION_LABELS[t]}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="field-label">Severity</label>
          <select
            className="select"
            value={value.severity}
            disabled={!hasSeverity}
            onChange={(e) => set({ severity: e.target.value })}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="custom">Custom</option>
          </select>
        </div>
      </div>

      {/* Custom parameters: only the ones for the selected corruption type */}
      {custom && (
        <div className="space-y-3 rounded-xl bg-indigo-50/60 p-3 ring-1 ring-indigo-100">
          {value.corruption === 'salt' && (
            <Slider label="salt_p (noise fraction)" value={value.salt_p} min={0.02} max={0.15} step={0.01}
              onChange={(v) => set({ salt_p: v })} />
          )}
          {value.corruption === 'blur' && (
            <>
              <div>
                <label className="field-label">blur_k (kernel size)</label>
                <select className="select" value={value.blur_k} onChange={(e) => set({ blur_k: Number(e.target.value) })}>
                  {[3, 5, 7].map((k) => (
                    <option key={k} value={k}>
                      {k} × {k}
                    </option>
                  ))}
                </select>
              </div>
              <Slider label="blur_sigma" value={value.blur_sigma} min={0.5} max={2.5} step={0.1}
                onChange={(v) => set({ blur_sigma: v })} />
            </>
          )}
          {value.corruption === 'occlusion' && (
            <>
              <Slider label="occ_coverage (area fraction)" value={value.occ_coverage} min={0.1} max={0.35} step={0.01}
                onChange={(v) => set({ occ_coverage: v })} />
              <Slider label="occ_rects (rectangles)" value={value.occ_rects} min={1} max={3} step={1}
                onChange={(v) => set({ occ_rects: v })} />
            </>
          )}
        </div>
      )}

      <div>
        <label className="field-label">Seed (optional, for reproducible corruption)</label>
        <input
          type="number"
          className="select"
          placeholder="random"
          value={value.seed}
          onChange={(e) => set({ seed: e.target.value })}
        />
      </div>
    </div>
  );
}

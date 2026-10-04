// Task 2 — a classifier predicts the corruption type and the image is routed to
// exactly ONE specialist autoencoder (or bypassed unchanged if it looks clean).
import { useState } from 'react';
import { ArrowRight, Brain, CornerDownRight } from 'lucide-react';
import RestorationWorkspace, { defaultSummary } from '../components/RestorationWorkspace.jsx';
import { WeightBars } from '../components/WeightBars.jsx';
import { Segmented } from '../components/ui.jsx';
import { EXPERTS, pct } from '../utils.js';

// classifier class -> expert key used in the pipeline diagram
const EXPERT_FOR_CLASS = { clean: 'identity', salt: 'salt', blur: 'blur', occlusion: 'occlusion' };
const BRANCHES = [
  { key: 'identity', label: 'Identity', color: '#10b981' },
  { key: 'salt', label: 'Salt AE', color: '#f59e0b' },
  { key: 'blur', label: 'Blur AE', color: '#0ea5e9' },
  { key: 'occlusion', label: 'Occlusion AE', color: '#f43f5e' },
];

/** Badge describing which expert handled the image; identity bypass looks different. */
function ExpertBadge({ expert }) {
  if (expert === 'identity') {
    return (
      <span className="inline-flex items-center gap-2 rounded-xl border-2 border-dashed border-emerald-400 bg-emerald-50 px-3 py-1.5 text-sm font-semibold text-emerald-800">
        Selected expert: Identity bypass (image judged clean — returned unchanged)
      </span>
    );
  }
  const b = BRANCHES.find((x) => x.key === expert);
  return (
    <span
      className="inline-flex items-center gap-2 rounded-xl px-3 py-1.5 text-sm font-semibold text-white shadow-sm"
      style={{ background: b?.color || '#4f46e5' }}
    >
      Selected expert: {b?.label || expert}
    </span>
  );
}

/** Input → Classifier → [4 branches] → Output, with the chosen branch highlighted. */
function PipelineDiagram({ expert, mode }) {
  const Node = ({ children, active = true }) => (
    <div
      className={`rounded-xl px-3 py-2 text-center text-xs font-semibold ${
        active ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-400'
      }`}
    >
      {children}
    </div>
  );
  return (
    <div className="flex flex-col items-center gap-3 overflow-x-auto md:flex-row md:justify-center">
      <Node>Input</Node>
      <ArrowRight className="h-4 w-4 rotate-90 text-slate-400 md:rotate-0" />
      <Node>
        <span className="flex items-center gap-1">
          <Brain className="h-3.5 w-3.5" /> Classifier
        </span>
        {mode === 'oracle' && <span className="block text-[10px] font-normal text-amber-300">bypassed (oracle)</span>}
      </Node>
      <ArrowRight className="h-4 w-4 rotate-90 text-slate-400 md:rotate-0" />
      <div className="flex flex-col gap-1.5">
        {BRANCHES.map((b) => {
          const chosen = b.key === expert;
          return (
            <div
              key={b.key}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                chosen ? 'text-white shadow' : 'bg-slate-50 text-slate-400 ring-1 ring-slate-200'
              } ${b.key === 'identity' ? 'border border-dashed border-emerald-500' : ''}`}
              style={chosen ? { background: b.color } : undefined}
            >
              {chosen && <CornerDownRight className="h-3.5 w-3.5" />}
              {b.label}
            </div>
          );
        })}
      </div>
      <ArrowRight className="h-4 w-4 rotate-90 text-slate-400 md:rotate-0" />
      <Node>Output</Node>
    </div>
  );
}

export default function HardView(props) {
  const [mode, setMode] = useState('predicted');

  return (
    <RestorationWorkspace
      {...props}
      kind="hard"
      task="Task 2"
      title="Hard-Routed Restoration"
      description="A CNN classifier recognises the corruption type and sends the image to a single specialist autoencoder. Oracle mode skips the classifier and uses the known (runtime) corruption instead."
      // Oracle routing needs a known corruption, so it is disabled for "none".
      renderControls={(settings) => {
        const oracleAllowed = settings.corruption !== 'none';
        const effective = oracleAllowed ? mode : 'predicted';
        return (
          <div>
            <label className="field-label">Routing mode</label>
            <Segmented
              value={effective}
              onChange={setMode}
              options={[
                { value: 'predicted', label: 'Predicted (classifier)' },
                {
                  value: 'oracle',
                  label: 'Oracle (true type)',
                  disabled: !oracleAllowed,
                  title: oracleAllowed ? '' : 'Needs a runtime corruption (not "none")',
                },
              ]}
            />
          </div>
        );
      }}
      getExtraFields={(settings) => ({ mode: settings.corruption === 'none' ? 'predicted' : mode })}
      summarize={(d) => `${d.expert === 'identity' ? 'Identity bypass' : `Expert: ${d.expert}`} · ${defaultSummary(d)}`}
      renderExtras={(r) => (
        <section className="card space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="card-title">Routing decision</h2>
            <span className="chip">mode: {r.mode}</span>
          </div>

          <ExpertBadge expert={r.expert} />

          {r.probabilities && (
            <div>
              <div className="mb-2 text-xs text-slate-500">
                Classifier probabilities — predicted <b className="text-slate-800">{r.predicted}</b> (
                {pct(r.probabilities[r.predicted])})
              </div>
              <WeightBars values={r.probabilities} highlight={r.predicted}
                labels={Object.fromEntries(EXPERTS.map((e) => [e.key, e.key === 'clean' ? 'Clean' : e.label]))} />
            </div>
          )}

          <div className="rounded-xl bg-slate-50 p-4">
            <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Pipeline</div>
            <PipelineDiagram expert={r.expert || EXPERT_FOR_CLASS[r.predicted]} mode={r.mode} />
          </div>
        </section>
      )}
    />
  );
}

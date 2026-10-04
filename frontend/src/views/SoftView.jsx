// Task 3 — a gating network produces softmax weights and the final output is the
// weighted sum of ALL four branches (identity + three specialist autoencoders).
import RestorationWorkspace, { defaultSummary } from '../components/RestorationWorkspace.jsx';
import { StackedBar, WeightBars } from '../components/WeightBars.jsx';
import ImagePanel from '../components/ImagePanel.jsx';
import { EXPERTS, makeFilename, pct } from '../utils.js';

export default function SoftView(props) {
  return (
    <RestorationWorkspace
      {...props}
      kind="soft"
      task="Task 3"
      title="Soft Mixture-of-Experts Restoration"
      description="A gating network assigns a softmax weight to every expert; the restored image is the weighted blend of the identity branch and the salt, blur and occlusion autoencoders."
      summarize={(d) => `Dominant: ${d.dominant} (${pct(d.weights?.[d.dominant])}) · ${defaultSummary(d)}`}
      renderExtras={(r) => (
        <section className="card space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="card-title">Gating weights</h2>
            <div className="flex gap-2">
              <span className="chip">dominant: {r.dominant}</span>
              <span className="chip">temperature: {r.temperature}</span>
            </div>
          </div>

          <WeightBars values={r.weights} highlight={r.dominant} />

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Contribution</div>
            <StackedBar values={r.weights} />
          </div>

          {r.expert_images && (
            <div>
              <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Branch outputs (border ∝ weight)
              </div>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                {EXPERTS.map((e) => {
                  const w = r.weights?.[e.key] ?? 0;
                  return (
                    <div key={e.key} className="flex justify-center" style={{ opacity: 0.35 + 0.65 * w }}>
                      <ImagePanel
                        src={r.expert_images[e.key]}
                        size={150}
                        caption={`${e.short} · ${pct(w)}`}
                        filename={makeFilename('soft', 'expert', e.key)}
                        // border thickness grows with the weight (1–8 px)
                        style={{ outline: `${1 + Math.round(w * 7)}px solid ${e.color}`, outlineOffset: 0 }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}
    />
  );
}

// Shared layout + logic for the three restoration workspaces (Tasks 1–3).
// Each view passes in what makes it different:
//   kind            -> 'universal' | 'hard' | 'soft'  (selects POST /api/restore/{kind})
//   renderControls  -> extra input controls (e.g. routing mode for the hard router)
//   getExtraFields  -> extra form fields sent to the endpoint (e.g. { mode })
//   renderExtras    -> view-specific result visualisations (bars, pipeline, experts)
//   summarize       -> one-line text for the run-history list
import { useState } from 'react';
import { Eye, Play } from 'lucide-react';
import { corruptImage, restoreImage } from '../api.js';
import { useImageSource } from '../hooks.js';
import { makeFilename } from '../utils.js';
import ImageSourcePicker from './ImageSourcePicker.jsx';
import CorruptionControls, { DEFAULT_CORRUPTION } from './CorruptionControls.jsx';
import ImagePanel from './ImagePanel.jsx';
import RunHistory from './RunHistory.jsx';
import { CorruptionChips, MetricsRow, TimingChips } from './ResultChips.jsx';
import { ErrorAlert, PageHeader, PixelToggle, Spinner } from './ui.jsx';

export default function RestorationWorkspace({
  kind,
  task,
  title,
  description,
  samples,
  samplesError,
  reloadSamples,
  renderControls,
  getExtraFields,
  renderExtras,
  summarize,
}) {
  const [source, setFile] = useImageSource();
  const [settings, setSettings] = useState(DEFAULT_CORRUPTION);
  const [preview, setPreview] = useState(null); // response of /api/corrupt
  const [result, setResult] = useState(null); // response of /api/restore/{kind}
  const [activeId, setActiveId] = useState(null);
  const [busy, setBusy] = useState(null); // 'preview' | 'run' | null
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);

  // When a new image is chosen, old previews no longer apply.
  function handleFile(file) {
    setFile(file);
    setPreview(null);
    setError('');
  }

  async function handlePreview() {
    setBusy('preview');
    setError('');
    try {
      setPreview(await corruptImage(source.file, settings));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function handleRun() {
    setBusy('run');
    setError('');
    try {
      const extra = getExtraFields ? getExtraFields(settings) : {};
      const data = await restoreImage(kind, source.file, settings, extra);
      const entry = {
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        thumb: data.output_image,
        label: data.corruption ? `${data.corruption.type}/${data.corruption.severity}` : 'as-is',
        summary: summarize ? summarize(data) : defaultSummary(data),
        result: data,
      };
      setResult(data);
      setActiveId(entry.id);
      setHistory((h) => [entry, ...h].slice(0, 5)); // keep the last 5 runs
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  const canPreview = source && settings.corruption !== 'none' && !busy;
  const canRun = source && !busy;

  return (
    <div>
      <PageHeader task={task} title={title} right={<PixelToggle />}>
        {description}
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        {/* ---------------- left column: inputs ---------------- */}
        <div className="space-y-6">
          <section className="card space-y-4">
            <h2 className="card-title">1 · Input image</h2>
            <ImageSourcePicker
              source={source}
              onFile={handleFile}
              samples={samples.pets}
              samplesError={samplesError}
              reloadSamples={reloadSamples}
            />
          </section>

          <section className="card space-y-4">
            <h2 className="card-title">2 · Runtime corruption</h2>
            <CorruptionControls value={settings} onChange={setSettings} />
            {renderControls && renderControls(settings)}

            <div className="flex flex-wrap gap-2 pt-1">
              <button className="btn-secondary" onClick={handlePreview} disabled={!canPreview}
                title={settings.corruption === 'none' ? 'Choose a corruption type to preview it' : ''}>
                {busy === 'preview' ? <Spinner /> : <Eye className="h-4 w-4" />} Preview corruption
              </button>
              <button className="btn-primary" onClick={handleRun} disabled={!canRun}>
                {busy === 'run' ? <Spinner /> : <Play className="h-4 w-4" />} Run model
              </button>
            </div>
            {!source && <p className="text-xs text-slate-500">Select an input image first.</p>}
            <ErrorAlert message={error} onClose={() => setError('')} />
          </section>

          <RunHistory
            items={history}
            activeId={activeId}
            onSelect={(h) => {
              setResult(h.result);
              setActiveId(h.id);
            }}
          />
        </div>

        {/* ---------------- right column: outputs ---------------- */}
        <div className="min-w-0 space-y-6">
          {preview && (
            <section className="card">
              <h2 className="card-title mb-4">Corruption preview</h2>
              <div className="flex flex-wrap justify-center gap-6">
                <ImagePanel src={preview.clean_image} caption="Clean" filename={makeFilename(kind, 'clean')} />
                <ImagePanel src={preview.corrupted_image} caption="Corrupted"
                  filename={makeFilename(kind, 'corrupted', preview.corruption?.type)} />
              </div>
              <div className="mt-4">
                <CorruptionChips corruption={preview.corruption} />
              </div>
            </section>
          )}

          <section className="card relative">
            <h2 className="card-title mb-4">Results</h2>
            {busy === 'run' && (
              <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-white/70">
                <div className="flex items-center gap-2 text-sm text-indigo-700">
                  <Spinner className="h-5 w-5" /> Running model…
                </div>
              </div>
            )}
            {!result ? (
              <p className="py-12 text-center text-sm text-slate-500">
                Choose an image and press <b>Run model</b> to see the restoration.
              </p>
            ) : (
              <div className="space-y-5">
                <div className="flex flex-wrap justify-center gap-6">
                  <ImagePanel src={result.input_image} caption="Input (what the model saw)"
                    filename={makeFilename(kind, 'input')} />
                  <ImagePanel src={result.output_image} caption="Restored output"
                    filename={makeFilename(kind, 'output')} />
                  {result.reference_image && (
                    <ImagePanel src={result.reference_image} caption="Clean reference"
                      filename={makeFilename(kind, 'reference')} />
                  )}
                  {result.error_map && (
                    <ImagePanel src={result.error_map} caption="Error map |output − reference|"
                      filename={makeFilename(kind, 'errormap')} />
                  )}
                </div>

                <div className="space-y-3 border-t border-slate-100 pt-4">
                  <Row label="Quality">
                    <MetricsRow metrics={result.metrics} />
                  </Row>
                  <Row label="Corruption">
                    <CorruptionChips corruption={result.corruption} />
                  </Row>
                  <Row label="Inference">
                    <TimingChips timings={result.inference_ms} />
                  </Row>
                  {result.model && (
                    <Row label="Model">
                      <span className="chip font-mono">{result.model}</span>
                    </Row>
                  )}
                </div>
              </div>
            )}
          </section>

          {result && renderExtras && renderExtras(result)}
        </div>
      </div>
    </div>
  );
}

/** Label on the left, content on the right (stacks on small screens). */
function Row({ label, children }) {
  return (
    <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:gap-4">
      <div className="w-24 shrink-0 pt-1 text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

export function defaultSummary(data) {
  if (data.metrics) {
    const d = data.metrics.psnr_output - data.metrics.psnr_input;
    return `PSNR ${data.metrics.psnr_output.toFixed(2)} dB (${d >= 0 ? '+' : ''}${d.toFixed(2)})`;
  }
  return `Restored in ${Number(data.inference_ms?.total ?? 0).toFixed(1)} ms`;
}

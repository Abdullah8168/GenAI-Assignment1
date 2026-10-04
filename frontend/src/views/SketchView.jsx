// Task 4 — conditional generator that turns a face photo into a pencil sketch
// in one of three styles. Input: upload, webcam capture or a sample face.
import { useState } from 'react';
import { Sparkles, PenTool, Layers } from 'lucide-react';
import { generateSketch } from '../api.js';
import { useImageSource } from '../hooks.js';
import { makeFilename } from '../utils.js';
import ImageSourcePicker from '../components/ImageSourcePicker.jsx';
import ImagePanel from '../components/ImagePanel.jsx';
import { TimingChips } from '../components/ResultChips.jsx';
import { ErrorAlert, PageHeader, PixelToggle, Spinner } from '../components/ui.jsx';

const STYLES = [
  { id: 1, name: 'Style 1', hint: 'Sketch style #1 of the training set' },
  { id: 2, name: 'Style 2', hint: 'Sketch style #2 of the training set' },
  { id: 3, name: 'Style 3', hint: 'Sketch style #3 of the training set' },
];

export default function SketchView({ samples, samplesError, reloadSamples, active }) {
  const [source, setFile] = useImageSource();
  const [style, setStyle] = useState(1);
  const [single, setSingle] = useState(null); // one /api/sketch response
  const [all, setAll] = useState(null); // array of 3 responses ("Generate all 3 styles")
  const [busy, setBusy] = useState(null); // 'one' | 'all' | null
  const [error, setError] = useState('');

  function handleFile(file) {
    setFile(file);
    setError('');
  }

  async function handleGenerate() {
    setBusy('one');
    setError('');
    try {
      setSingle(await generateSketch(source.file, style));
      setAll(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateAll() {
    setBusy('all');
    setError('');
    try {
      // Three sequential calls, one per style.
      const results = [];
      for (const s of STYLES) results.push(await generateSketch(source.file, s.id));
      setAll(results);
      setSingle(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div>
      <PageHeader task="Task 4" title="Face-to-Sketch Generator" right={<PixelToggle />}>
        A conditional image-to-image generator converts a facial photograph into a pencil sketch. The style
        code selects one of three sketch styles.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        {/* ---------------- inputs ---------------- */}
        <div className="space-y-6">
          <section className="card space-y-4">
            <h2 className="card-title">1 · Facial photo</h2>
            <ImageSourcePicker
              source={source}
              onFile={handleFile}
              samples={samples.faces}
              samplesError={samplesError}
              reloadSamples={reloadSamples}
              allowWebcam
              webcamActive={active} // camera is released when this view is hidden
            />
          </section>

          <section className="card space-y-4">
            <h2 className="card-title">2 · Sketch style</h2>
            <div className="grid grid-cols-3 gap-2">
              {STYLES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setStyle(s.id)}
                  title={s.hint}
                  className={`flex flex-col items-center gap-1 rounded-xl p-3 text-sm font-medium ring-2 transition ${
                    style === s.id
                      ? 'bg-indigo-50 text-indigo-700 ring-indigo-500'
                      : 'bg-white text-slate-600 ring-slate-200 hover:ring-indigo-300'
                  }`}
                >
                  <PenTool className="h-5 w-5" />
                  {s.name}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="btn-primary" onClick={handleGenerate} disabled={!source || !!busy}>
                {busy === 'one' ? <Spinner /> : <Sparkles className="h-4 w-4" />} Generate sketch
              </button>
              <button className="btn-secondary" onClick={handleGenerateAll} disabled={!source || !!busy}>
                {busy === 'all' ? <Spinner /> : <Layers className="h-4 w-4" />} Generate all 3 styles
              </button>
            </div>
            {!source && <p className="text-xs text-slate-500">Upload, capture or pick a face first.</p>}
            <ErrorAlert message={error} onClose={() => setError('')} />
          </section>
        </div>

        {/* ---------------- outputs ---------------- */}
        <section className="card relative min-w-0">
          <h2 className="card-title mb-4">Result</h2>
          {busy && (
            <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-white/70">
              <div className="flex items-center gap-2 text-sm text-indigo-700">
                <Spinner className="h-5 w-5" /> Generating…
              </div>
            </div>
          )}

          {!single && !all && (
            <p className="py-12 text-center text-sm text-slate-500">
              Choose a face and press <b>Generate sketch</b>.
            </p>
          )}

          {single && (
            <div className="space-y-4">
              <div className="flex flex-wrap justify-center gap-6">
                <ImagePanel src={single.input_image} caption="Original photo" filename={makeFilename('sketch', 'photo')} />
                <ImagePanel src={single.output_image} caption={`Generated sketch · Style ${single.style}`}
                  filename={makeFilename('sketch', `style${single.style}`)} />
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                <TimingChips timings={single.inference_ms} />
                {single.model && <span className="chip font-mono">{single.model}</span>}
              </div>
            </div>
          )}

          {all && (
            <div className="space-y-4">
              <div className="flex flex-wrap justify-center gap-6">
                <ImagePanel src={all[0].input_image} caption="Original photo" size={200}
                  filename={makeFilename('sketch', 'photo')} />
                {all.map((r) => (
                  <ImagePanel
                    key={r.style}
                    src={r.output_image}
                    size={200}
                    caption={`Style ${r.style} · ${Number(r.inference_ms?.total ?? 0).toFixed(1)} ms`}
                    filename={makeFilename('sketch', `style${r.style}`)}
                  />
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

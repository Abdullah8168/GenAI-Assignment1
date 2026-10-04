// Input source selector: Upload (drag & drop / file picker), Samples gallery,
// and optionally Webcam. Calls onFile(File) whenever a new input image is chosen.
import { useRef, useState } from 'react';
import { Upload, Images, Camera, RefreshCw } from 'lucide-react';
import { fetchSampleFile } from '../api.js';
import { validateImageFile, ACCEPTED_TYPES } from '../utils.js';
import { Segmented, Spinner, ErrorAlert } from './ui.jsx';
import WebcamCapture from './WebcamCapture.jsx';

export default function ImageSourcePicker({
  source, // { file, url } or null
  onFile,
  samples = [],
  samplesError,
  reloadSamples,
  allowWebcam = false,
  webcamActive = true, // parent can force the webcam off (view not visible)
}) {
  const [tab, setTab] = useState('upload');
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  const [loadingSample, setLoadingSample] = useState(null);
  const inputRef = useRef(null);

  function acceptFile(file) {
    const problem = validateImageFile(file);
    if (problem) {
      setError(problem);
      return;
    }
    setError('');
    onFile(file);
  }

  async function pickSample(sample) {
    setLoadingSample(sample.name);
    setError('');
    try {
      onFile(await fetchSampleFile(sample));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingSample(null);
    }
  }

  const tabs = [
    { value: 'upload', label: 'Upload' },
    { value: 'samples', label: 'Samples' },
  ];
  if (allowWebcam) tabs.push({ value: 'webcam', label: 'Webcam' });

  return (
    <div className="space-y-3">
      <Segmented value={tab} onChange={setTab} options={tabs} />

      {tab === 'upload' && (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            acceptFile(e.dataTransfer.files?.[0]);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center text-sm transition ${
            dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'
          }`}
        >
          <Upload className="h-6 w-6 text-indigo-500" />
          <span className="font-medium text-slate-700">Drop an image here or click to browse</span>
          <span className="text-xs text-slate-500">PNG, JPG or WEBP · max 10 MB</span>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(',')}
            className="hidden"
            onChange={(e) => {
              acceptFile(e.target.files?.[0]);
              e.target.value = ''; // allow re-selecting the same file
            }}
          />
        </div>
      )}

      {tab === 'samples' && (
        <div>
          {samplesError ? (
            <div className="space-y-2">
              <ErrorAlert message={`Could not load samples: ${samplesError}`} />
              <button className="btn-secondary text-xs" onClick={reloadSamples}>
                <RefreshCw className="h-3.5 w-3.5" /> Retry
              </button>
            </div>
          ) : samples.length === 0 ? (
            <p className="flex items-center gap-2 text-sm text-slate-500">
              <Images className="h-4 w-4" /> No samples available.
            </p>
          ) : (
            <div className="grid max-h-56 grid-cols-4 gap-2 overflow-y-auto pr-1 sm:grid-cols-5">
              {samples.map((s) => (
                <button
                  key={s.name}
                  title={s.name}
                  onClick={() => pickSample(s)}
                  disabled={!!loadingSample}
                  className={`relative aspect-square overflow-hidden rounded-lg ring-2 transition hover:ring-indigo-400 ${
                    source?.file?.name === s.name ? 'ring-indigo-600' : 'ring-transparent'
                  }`}
                >
                  <img src={s.url} alt={s.name} className="h-full w-full object-cover" loading="lazy" />
                  {loadingSample === s.name && (
                    <span className="absolute inset-0 flex items-center justify-center bg-white/60">
                      <Spinner />
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Only mounted while visible, so the camera is released otherwise */}
      {tab === 'webcam' && allowWebcam && webcamActive && <WebcamCapture onCapture={acceptFile} />}
      {tab === 'webcam' && allowWebcam && !webcamActive && (
        <p className="flex items-center gap-2 text-sm text-slate-500">
          <Camera className="h-4 w-4" /> Camera paused.
        </p>
      )}

      <ErrorAlert message={error} onClose={() => setError('')} />

      {source && (
        <div className="flex items-center gap-3 rounded-xl bg-slate-50 p-2 ring-1 ring-slate-200">
          <img src={source.url} alt="Selected input" className="h-14 w-14 rounded-lg object-cover" />
          <div className="min-w-0 text-xs">
            <div className="font-medium text-slate-700">Selected input</div>
            <div className="truncate text-slate-500">{source.file.name}</div>
            <div className="text-slate-400">{(source.file.size / 1024).toFixed(1)} KB</div>
          </div>
        </div>
      )}
    </div>
  );
}

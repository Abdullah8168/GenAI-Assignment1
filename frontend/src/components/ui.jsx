// Small presentational building blocks used across the app.
import { AlertTriangle, Loader2 } from 'lucide-react';
import { useDisplay } from '../display.jsx';

export function Spinner({ className = 'h-4 w-4' }) {
  return <Loader2 className={`${className} animate-spin`} />;
}

/** Red alert box for error messages (e.g. the backend's `detail`). */
export function ErrorAlert({ message, onClose }) {
  if (!message) return null;
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1 break-words">{message}</div>
      {onClose && (
        <button onClick={onClose} className="text-red-500 hover:text-red-700" aria-label="Dismiss error">
          ×
        </button>
      )}
    </div>
  );
}

/** Page heading: task tag + exact workspace name + description. */
export function PageHeader({ task, title, children, right }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        {task && (
          <span className="mb-2 inline-block rounded-full bg-indigo-100 px-2.5 py-0.5 text-xs font-semibold text-indigo-700">
            {task}
          </span>
        )}
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">{title}</h1>
        {children && <p className="mt-2 text-sm text-slate-600">{children}</p>}
      </div>
      {right}
    </div>
  );
}

/** Switch between smooth and pixelated (nearest-neighbour) upscaling of 128×128 images. */
export function PixelToggle() {
  const { pixelated, setPixelated } = useDisplay();
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-xs text-slate-600">
      <input
        type="checkbox"
        className="h-4 w-4 rounded accent-indigo-600"
        checked={pixelated}
        onChange={(e) => setPixelated(e.target.checked)}
      />
      Pixelated upscaling
    </label>
  );
}

/** Segmented control: options = [{value, label, disabled?}] */
export function Segmented({ value, onChange, options }) {
  return (
    <div className="inline-flex rounded-xl bg-slate-100 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${
            value === o.value ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

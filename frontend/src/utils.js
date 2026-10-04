// Small helpers shared by several components.

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB
export const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

/** Client-side upload check. Returns an error message, or '' if the file is fine. */
export function validateImageFile(file) {
  if (!file) return 'No file selected.';
  if (!ACCEPTED_TYPES.includes(file.type)) return 'Only PNG, JPG or WEBP images are accepted.';
  if (file.size > MAX_UPLOAD_BYTES) return 'File is larger than 10 MB.';
  return '';
}

/** Save a data URL (e.g. "data:image/png;base64,...") as a file. */
export function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Builds names like "restorelab_hard_output_20261004-142233.png". */
export function makeFilename(...parts) {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const stamp =
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return ['restorelab', ...parts, stamp].join('_') + '.png';
}

/** Readable text for any parameter value (numbers, rectangle arrays, objects). */
export function formatValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(+v.toFixed(4));
  if (Array.isArray(v)) {
    // a list of rectangles like [[x,y,w,h], ...] -> "(x,y,w,h) (x,y,w,h)"
    if (v.length && Array.isArray(v[0])) return v.map((r) => `(${r.map(formatValue).join(',')})`).join(' ');
    if (v.length && typeof v[0] === 'object') return v.map(formatValue).join(' ');
    return `[${v.map(formatValue).join(', ')}]`;
  }
  if (typeof v === 'object') {
    return `{${Object.entries(v)
      .map(([k, x]) => `${k}:${formatValue(x)}`)
      .join(', ')}}`;
  }
  return String(v);
}

/** 0.973 -> "97.3%" */
export const pct = (x) => `${((x ?? 0) * 100).toFixed(1)}%`;

/** Classes / experts in a fixed order, with display labels and colours (hex, used inline). */
export const EXPERTS = [
  { key: 'clean', label: 'Clean / Identity', short: 'Identity', color: '#10b981' },
  { key: 'salt', label: 'Salt & pepper', short: 'Salt AE', color: '#f59e0b' },
  { key: 'blur', label: 'Gaussian blur', short: 'Blur AE', color: '#0ea5e9' },
  { key: 'occlusion', label: 'Occlusion', short: 'Occlusion AE', color: '#f43f5e' },
];

export const CORRUPTION_LABELS = {
  none: 'None (already corrupted — use as-is)',
  clean: 'Clean (no corruption)',
  salt: 'Salt & pepper',
  blur: 'Gaussian blur',
  occlusion: 'Rectangular occlusion',
};

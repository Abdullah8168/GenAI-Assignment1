// ---------------------------------------------------------------------------
// api.js — every call to the FastAPI backend lives in this one module.
// All URLs are relative ("/api/..."): Vite's dev proxy (vite.config.js) or
// nginx (nginx.conf) forwards them to the backend.
// ---------------------------------------------------------------------------

const BASE = '/api';

/** Turn FastAPI's `detail` (string OR validation-error array) into readable text. */
export function formatDetail(detail) {
  if (!detail) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const where = (d.loc || []).filter((p) => p !== 'body').join('.');
        return where ? `${where}: ${d.msg}` : d.msg;
      })
      .join('; ');
  }
  return JSON.stringify(detail);
}

/** fetch() wrapper: parses JSON and throws an Error with a readable message on failure. */
async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error('Cannot reach the backend. Is the API server running on port 8000?');
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* response had no JSON body */
  }
  if (!res.ok) {
    const msg = formatDetail(data?.detail);
    throw new Error(msg || `Request failed: HTTP ${res.status} ${res.statusText}`);
  }
  return data;
}

/**
 * Append the runtime-corruption settings to a FormData object.
 * Custom parameters are only sent when severity === 'custom', and only the
 * ones that belong to the selected corruption type.
 */
function appendCorruption(form, s) {
  form.append('corruption', s.corruption);
  form.append('severity', s.severity);
  if (s.severity === 'custom') {
    if (s.corruption === 'salt') {
      form.append('salt_p', String(s.salt_p));
    } else if (s.corruption === 'blur') {
      form.append('blur_k', String(s.blur_k));
      form.append('blur_sigma', String(s.blur_sigma));
    } else if (s.corruption === 'occlusion') {
      form.append('occ_coverage', String(s.occ_coverage));
      form.append('occ_rects', String(s.occ_rects));
    }
  }
  if (s.seed !== '' && s.seed !== null && s.seed !== undefined) {
    const seed = parseInt(s.seed, 10);
    if (!Number.isNaN(seed)) form.append('seed', String(seed));
  }
}

// ---------------------------- endpoints ------------------------------------

/** GET /api/health — model status, onnxruntime info, uptime. */
export function getHealth() {
  return request('/health');
}

/** GET /api/samples — { pets: [{name,url}], faces: [{name,url}] } */
export function getSamples() {
  return request('/samples');
}

/** Download a sample image (its url already starts with /api) and wrap it as a File. */
export async function fetchSampleFile(sample) {
  let res;
  try {
    res = await fetch(sample.url);
  } catch {
    throw new Error('Cannot reach the backend to download the sample.');
  }
  if (!res.ok) throw new Error(`Could not load sample ${sample.name} (HTTP ${res.status})`);
  const blob = await res.blob();
  return new File([blob], sample.name, { type: blob.type || 'image/png' });
}

/** POST /api/corrupt — returns clean + corrupted preview images. */
export function corruptImage(file, settings) {
  const form = new FormData();
  form.append('file', file);
  appendCorruption(form, settings);
  return request('/corrupt', { method: 'POST', body: form });
}

/**
 * POST /api/restore/{kind} where kind = 'universal' | 'hard' | 'soft'.
 * `extra` holds view-specific fields, e.g. { mode: 'oracle' } for the hard router.
 */
export function restoreImage(kind, file, settings, extra = {}) {
  const form = new FormData();
  form.append('file', file);
  appendCorruption(form, settings);
  Object.entries(extra).forEach(([k, v]) => form.append(k, String(v)));
  return request(`/restore/${kind}`, { method: 'POST', body: form });
}

/** POST /api/sketch — face photo -> pencil sketch in style 1, 2 or 3. */
export function generateSketch(file, style) {
  const form = new FormData();
  form.append('file', file);
  form.append('style', String(style));
  return request('/sketch', { method: 'POST', body: form });
}

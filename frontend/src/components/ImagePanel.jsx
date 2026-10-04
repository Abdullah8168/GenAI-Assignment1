// Shows one (usually 128×128) image upscaled to ~256 px with a caption and an
// optional Download button. Scaling is smooth unless the global "pixelated" toggle is on.
import { Download } from 'lucide-react';
import { useDisplay } from '../display.jsx';
import { downloadDataUrl } from '../utils.js';

export default function ImagePanel({ src, caption, filename, size = 256, downloadable = true, style, footer }) {
  const { pixelated } = useDisplay();
  return (
    <figure className="flex flex-col items-center gap-2">
      <div
        className="overflow-hidden rounded-xl bg-slate-100 ring-1 ring-slate-200"
        style={{ width: size, height: size, maxWidth: '100%', ...style }}
      >
        {src ? (
          <img
            src={src}
            alt={caption}
            className="h-full w-full object-contain"
            style={{ imageRendering: pixelated ? 'pixelated' : 'auto' }}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-400">No image</div>
        )}
      </div>
      <figcaption className="text-center text-xs font-medium text-slate-600">{caption}</figcaption>
      {footer}
      {downloadable && src && (
        <button className="btn-ghost text-xs" onClick={() => downloadDataUrl(src, filename || 'image.png')}>
          <Download className="h-3.5 w-3.5" /> Download
        </button>
      )}
    </figure>
  );
}

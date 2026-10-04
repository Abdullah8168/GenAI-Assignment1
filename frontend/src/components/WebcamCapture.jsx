// Live webcam preview + "Capture" button. The camera stream is started when this
// component mounts and STOPPED when it unmounts (e.g. switching tab or leaving the view).
import { useEffect, useRef, useState } from 'react';
import { Camera } from 'lucide-react';
import { Spinner } from './ui.jsx';

export default function WebcamCapture({ onCapture }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('Webcam is not supported in this browser (it needs HTTPS or localhost).');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setReady(true);
      } catch (e) {
        setError(`Could not open the webcam: ${e.message}`);
      }
    }
    start();
    // Cleanup: release the camera.
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  /** Grab a centred square frame (mirrored like the preview) and hand it back as a PNG File. */
  function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const side = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    const canvas = document.createElement('canvas');
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext('2d');
    ctx.translate(side, 0); // mirror horizontally so the capture matches the preview
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, side, side, 0, 0, side, side);
    canvas.toBlob((blob) => {
      if (blob) onCapture(new File([blob], `webcam_${Date.now()}.png`, { type: 'image/png' }));
    }, 'image/png');
  }

  if (error) return <p className="rounded-xl bg-amber-50 p-3 text-sm text-amber-800">{error}</p>;

  return (
    <div className="space-y-3">
      <div className="relative mx-auto aspect-square w-full max-w-xs overflow-hidden rounded-xl bg-slate-900">
        <video ref={videoRef} playsInline muted className="h-full w-full scale-x-[-1] object-cover" />
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-300">
            <Spinner className="h-6 w-6" />
          </div>
        )}
      </div>
      <button className="btn-primary w-full" onClick={capture} disabled={!ready}>
        <Camera className="h-4 w-4" /> Capture
      </button>
    </div>
  );
}

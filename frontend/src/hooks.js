// Custom React hooks.
import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Holds the currently selected input image as { file, url } where `url` is an
 * object URL for previewing. Old object URLs are revoked to avoid memory leaks.
 */
export function useImageSource() {
  const [source, setSource] = useState(null);
  const urlRef = useRef(null);

  const setFile = useCallback((file) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    if (!file) {
      urlRef.current = null;
      setSource(null);
      return;
    }
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    setSource({ file, url });
  }, []);

  // revoke the last URL when the component using the hook unmounts
  useEffect(() => () => urlRef.current && URL.revokeObjectURL(urlRef.current), []);

  return [source, setFile];
}

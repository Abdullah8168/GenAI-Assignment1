// A tiny React context for display preferences shared by every image panel
// (currently only: nearest-neighbour "pixelated" upscaling on/off).
import { createContext, useContext, useState } from 'react';

const DisplayContext = createContext({ pixelated: false, setPixelated: () => {} });

export function DisplayProvider({ children }) {
  const [pixelated, setPixelated] = useState(false); // default: smooth scaling
  return <DisplayContext.Provider value={{ pixelated, setPixelated }}>{children}</DisplayContext.Provider>;
}

export const useDisplay = () => useContext(DisplayContext);

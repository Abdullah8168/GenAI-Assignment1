// Root component: owns navigation state, polls /api/health and loads /api/samples once.
// Navigation is plain React state (no router). All views stay mounted (hidden with CSS)
// so each workspace keeps its inputs, results and run history when you switch tabs.
import { useCallback, useEffect, useState } from 'react';
import { Wand2, Route, Layers, PenTool, Server } from 'lucide-react';
import Sidebar from './components/Sidebar.jsx';
import UniversalView from './views/UniversalView.jsx';
import HardView from './views/HardView.jsx';
import SoftView from './views/SoftView.jsx';
import SketchView from './views/SketchView.jsx';
import SystemView from './views/SystemView.jsx';
import { getHealth, getSamples } from './api.js';

// The five workspaces shown in the sidebar.
export const VIEWS = [
  { id: 'universal', label: 'Universal Restoration', task: 'Task 1', icon: Wand2 },
  { id: 'hard', label: 'Hard-Routed Restoration', task: 'Task 2', icon: Route },
  { id: 'soft', label: 'Soft Mixture-of-Experts Restoration', task: 'Task 3', icon: Layers },
  { id: 'sketch', label: 'Face-to-Sketch Generator', task: 'Task 4', icon: PenTool },
  { id: 'system', label: 'System', task: 'Status', icon: Server },
];

const HEALTH_POLL_MS = 15000;

export default function App() {
  const [view, setView] = useState('universal');

  // ---- health polling (sidebar dot + System view) ----
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState('');
  const [healthCheckedAt, setHealthCheckedAt] = useState(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await getHealth());
      setHealthError('');
    } catch (e) {
      setHealth(null);
      setHealthError(e.message);
    } finally {
      setHealthCheckedAt(new Date());
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const id = setInterval(refreshHealth, HEALTH_POLL_MS);
    return () => clearInterval(id); // stop polling on unmount
  }, [refreshHealth]);

  // ---- sample gallery (shared by all views) ----
  const [samples, setSamples] = useState({ pets: [], faces: [] });
  const [samplesError, setSamplesError] = useState('');

  const loadSamples = useCallback(async () => {
    try {
      const data = await getSamples();
      setSamples({ pets: data?.pets || [], faces: data?.faces || [] });
      setSamplesError('');
    } catch (e) {
      setSamplesError(e.message);
    }
  }, []);

  useEffect(() => {
    loadSamples();
  }, [loadSamples]);

  const sampleProps = { samples, samplesError, reloadSamples: loadSamples };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800 md:flex">
      <Sidebar views={VIEWS} current={view} onSelect={setView} health={health} healthError={healthError} />

      <main className="min-w-0 flex-1 px-4 py-6 md:ml-72 md:px-8 md:py-8">
        <div className={view === 'universal' ? '' : 'hidden'}>
          <UniversalView {...sampleProps} />
        </div>
        <div className={view === 'hard' ? '' : 'hidden'}>
          <HardView {...sampleProps} />
        </div>
        <div className={view === 'soft' ? '' : 'hidden'}>
          <SoftView {...sampleProps} />
        </div>
        <div className={view === 'sketch' ? '' : 'hidden'}>
          {/* `active` lets the sketch view switch the webcam off when you leave it */}
          <SketchView {...sampleProps} active={view === 'sketch'} />
        </div>
        <div className={view === 'system' ? '' : 'hidden'}>
          <SystemView
            health={health}
            error={healthError}
            checkedAt={healthCheckedAt}
            onRefresh={refreshHealth}
          />
        </div>
      </main>
    </div>
  );
}

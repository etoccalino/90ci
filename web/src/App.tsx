import { useState } from 'react';
import type { Model, SimResult } from './model';
import { useNinetyCi } from './hooks/useNinetyCi';
import { TopBar } from './components/TopBar';
import { ModelEditor } from './components/ModelEditor';
import { ResultPanel } from './components/ResultPanel';

const INITIAL_MODEL: Model = {
  name: 'Exchange exposure',
  equation: '200 * EXCHANGE_RATE + BASE_FEE',
  variables: [
    { id: 'v1', name: 'EXCHANGE_RATE', shape: 'uniform', p5: 1000, p95: 1200 },
    { id: 'v2', name: 'BASE_FEE', shape: 'normal', p5: 0, p95: 50 },
  ],
};

export default function App() {
  const [model, setModel] = useState<Model>(INITIAL_MODEL);
  const [result, setResult] = useState<SimResult | null>(null);
  const { run, running, error, engineReady } = useNinetyCi();

  const onRun = async () => {
    try {
      const simResult = await run(model);
      setResult(simResult);
    } catch {
      // E-10: clear any stale result so the error banner is never shown beside
      // an old chart. The error message is surfaced through the hook's `error` state.
      setResult(null);
    }
  };

  // E-09: Run is disabled only when the engine failed to load (not for validation errors).
  const runDisabled = running || engineReady === false;

  return (
    <div className="app">
      <TopBar modelName={model.name} />
      <div className="page">
        <ModelEditor
          model={model}
          onChange={setModel}
          onRun={onRun}
          running={running}
          runDisabled={runDisabled}
          validationError={error}
        />
        <ResultPanel result={result} error={error} />
      </div>
    </div>
  );
}

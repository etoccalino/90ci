import type { SimResult } from '../model';
import { OutputChart } from './OutputChart';
import { CIHero } from './CIHero';

interface Props {
  result: SimResult | null;
  error: string | null;
}

export function ResultPanel({ result, error }: Props) {
  return (
    <div className="col right">
      <div className="sec-lbl">
        <span className="ico">◉</span> Output distribution
      </div>

      {error && <div className="error">⚠ {error}</div>}
      {!result && !error && (
        <div className="empty">
          Press <b>Run</b> to simulate the model.
        </div>
      )}

      {result && (
        <>
          <div className="resultcard">
            <div className="rc-head">
              <span className="t">Output value</span>
              <span className="s">{result.samples.toLocaleString('en-US')} samples</span>
            </div>
            <OutputChart result={result} />
          </div>
          <CIHero result={result} />
        </>
      )}
    </div>
  );
}

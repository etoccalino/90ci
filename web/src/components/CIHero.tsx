import type { SimResult } from '../model';

const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

export function CIHero({ result }: { result: SimResult }) {
  const width = Math.max(0, result.ciHigh - result.ciLow);
  return (
    <div className="ci-hero">
      <div className="ci-lbl">
        <span className="ico">[ ]</span> 90% Confidence interval
      </div>
      <div className="ci-range">
        <span>{fmt(result.ciLow)}</span>
        <span className="dash">–</span>
        <span>{fmt(result.ciHigh)}</span>
      </div>
      <div className="ci-sub">
        Width <b>{fmt(width)}</b> · the middle 90% of outcomes land in this range.
      </div>
    </div>
  );
}

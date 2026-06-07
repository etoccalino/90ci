import type { SimResult } from '../model';

const W = 520;
const H = 188;
const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

// Draws the output histogram as an area/line chart, with the 90% interval band
// shaded and dashed markers at the bounds. Adapted from the mock's `renderChart`,
// but driven by the real (buckets, counts) instead of a hardcoded curve.
export function OutputChart({ result }: { result: SimResult }) {
  const { buckets, counts, ciLow, ciHigh } = result;
  if (buckets.length < 2) {
    return <svg className="chart" viewBox={`0 0 ${W} ${H}`} />;
  }

  const lo = buckets[0];
  const hi = buckets[buckets.length - 1];
  const span = hi - lo || 1;
  const maxCount = Math.max(...counts, 1);
  const base = H - 4;
  const top = 12;

  const x = (val: number) => ((val - lo) / span) * W;
  const y = (c: number) => base - (c / maxCount) * (base - top);

  const pts = buckets.map((b, i) => [x(b), y(counts[i])] as [number, number]);
  const line = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const area = `M 0 ${base} ${pts.map((p) => `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ')} L ${W} ${base} Z`;

  const x5 = x(ciLow);
  const x95 = x(ciHigh);
  const ciPts = pts.filter((p) => p[0] >= x5 && p[0] <= x95);
  const ci = `M ${x5.toFixed(1)} ${base} ${ciPts
    .map((p) => `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`)
    .join(' ')} L ${x95.toFixed(1)} ${base} Z`;

  const accent = 'var(--blue)';
  return (
    <div className="chartwrap">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={area} fill="#eef2f6" />
        <path d={ci} fill={accent} fillOpacity={0.18} />
        <line x1={x5} y1={top} x2={x5} y2={base} stroke={accent} strokeWidth={1} strokeDasharray="3 3" />
        <line x1={x95} y1={top} x2={x95} y2={base} stroke={accent} strokeWidth={1} strokeDasharray="3 3" />
        <path d={line} fill="none" stroke={accent} strokeWidth={2} strokeLinejoin="round" />
      </svg>
      <div className="axis">
        <span>{fmt(lo)}</span>
        <span>{fmt((lo + hi) / 2)}</span>
        <span>{fmt(hi)}</span>
      </div>
    </div>
  );
}

import type { SimResult } from '../model';

const W = 520;
const H = 188;
const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

// Minimum pixel separation between the two CI marker lines. When x5 and x95
// collapse below this threshold both labels are rendered as a single combined
// marker to avoid illegible overlapping text.
const MIN_MARKER_SEP = 2;

// Draws the output histogram as an area/line chart, with the 90% interval band
// shaded and dashed markers at the bounds. Adapted from the mock's `renderChart`,
// but driven by the real (buckets, counts) instead of a hardcoded curve.
export function OutputChart({ result }: { result: SimResult }) {
  const { buckets, counts, ciLow, ciHigh } = result;

  // Degenerate: single bucket (fixed-value model or zero-width spread).
  if (buckets.length < 2) {
    const val = buckets[0] ?? ciLow;
    const mid = W / 2;
    const base = H - 4;
    const top = 12;
    return (
      <div className="chartwrap">
        <svg
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          aria-label={`Fixed value: ${fmt(val)}`}
        >
          {/* Spike line */}
          <line
            x1={mid}
            y1={top}
            x2={mid}
            y2={base}
            stroke="var(--blue)"
            strokeWidth={2}
            strokeDasharray="3 3"
            data-testid="degenerate-spike"
          />
          {/* Value label centered above the spike */}
          <text
            x={mid}
            y={top - 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--ink)"
            data-testid="degenerate-label"
          >
            {fmt(val)}
          </text>
        </svg>
        <div className="axis">
          <span>{fmt(val)}</span>
          <span>{fmt(val)}</span>
          <span>{fmt(val)}</span>
        </div>
      </div>
    );
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

  const x5Raw = x(ciLow);
  const x95Raw = x(ciHigh);

  // Enforce minimum pixel separation so both markers stay visible on narrow spread.
  const sep = x95Raw - x5Raw;
  const x5 = sep < MIN_MARKER_SEP ? x5Raw - MIN_MARKER_SEP / 2 : x5Raw;
  const x95 = sep < MIN_MARKER_SEP ? x5Raw + MIN_MARKER_SEP / 2 : x95Raw;
  const narrowSpread = sep < MIN_MARKER_SEP;

  const ciPts = pts.filter((p) => p[0] >= x5Raw && p[0] <= x95Raw);
  const ci = `M ${x5.toFixed(1)} ${base} ${ciPts
    .map((p) => `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`)
    .join(' ')} L ${x95.toFixed(1)} ${base} Z`;

  const accent = 'var(--blue)';

  // Label y position: just above the top of the chart.
  const lblY = top - 2;

  return (
    <div className="chartwrap">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={area} fill="#eef2f6" />
        <path d={ci} fill={accent} fillOpacity={0.18} />
        <line x1={x5} y1={top} x2={x5} y2={base} stroke={accent} strokeWidth={1} strokeDasharray="3 3" data-testid="marker-low" />
        <line x1={x95} y1={top} x2={x95} y2={base} stroke={accent} strokeWidth={1} strokeDasharray="3 3" data-testid="marker-high" />
        {/* Marker labels: when spread is too narrow, collapse into one combined label */}
        {narrowSpread ? (
          <text
            x={(x5 + x95) / 2}
            y={lblY}
            textAnchor="middle"
            fontSize={10}
            fill={accent}
            data-testid="marker-label-low"
          >
            {fmt(ciLow)}
          </text>
        ) : (
          <>
            <text
              x={x5}
              y={lblY}
              textAnchor="start"
              fontSize={10}
              fill={accent}
              data-testid="marker-label-low"
            >
              {fmt(ciLow)}
            </text>
            <text
              x={x95}
              y={lblY}
              textAnchor="end"
              fontSize={10}
              fill={accent}
              data-testid="marker-label-high"
            >
              {fmt(ciHigh)}
            </text>
          </>
        )}
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

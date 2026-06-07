import type { Shape } from '../model';

// Ported from the design mock's `sparkPath`: a plateau for uniform/range, a bell
// for normal.
function sparkPath(shape: Shape, w: number, h: number): string {
  const pts: [number, number][] = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const y = shape === 'normal' ? Math.exp(-Math.pow((t - 0.5) / 0.16, 2) / 2) : t > 0.18 && t < 0.82 ? 1 : 0.04;
    pts.push([4 + t * (w - 8), h - 3 - y * (h - 7)]);
  }
  return pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
}

export function Sparkline({ shape }: { shape: Shape }) {
  const w = 54;
  const h = 24;
  const color = shape === 'normal' ? 'var(--grn)' : 'var(--blue)';
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`}>
      <path d={sparkPath(shape, w, h)} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

export type Shape = 'uniform' | 'normal' | 'range';

export interface Variable {
  id: string;
  name: string;
  shape: Shape;
  p5: number; // 5th percentile — maps to the engine's `lower`
  p95: number; // 95th percentile — maps to the engine's `upper`
}

export interface Model {
  name: string;
  equation: string;
  variables: Variable[];
  samples: number;
}

export interface SimResult {
  ciLow: number;
  ciHigh: number;
  buckets: number[];
  counts: number[];
  samples: number;
}

export const SHAPES: Shape[] = ['uniform', 'normal', 'range'];
export const SAMPLE_OPTIONS = [1_000, 10_000, 100_000] as const;

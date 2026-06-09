export type Shape = 'uniform' | 'normal' | 'range';

export interface Variable {
  id: string;
  name: string;
  shape: Shape;
  p5: number | null; // 5th percentile — maps to the engine's `lower`; null = blank/unset
  p95: number | null; // 95th percentile — maps to the engine's `upper`; null = blank/unset
}

export interface Model {
  name: string;
  equation: string;
  variables: Variable[];
}

export interface SimResult {
  ciLow: number;
  ciHigh: number;
  buckets: number[];
  counts: number[];
  samples: number;
}

export const SHAPES: Shape[] = ['uniform', 'normal', 'range'];

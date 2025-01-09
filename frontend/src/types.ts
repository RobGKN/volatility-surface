export interface SurfaceData {
    x: number[];
    y: number[];
    z: number[][];
    modelParams?: {
      alpha: number;
      beta: number;
      rho: number;
      nu: number;
    };
  }
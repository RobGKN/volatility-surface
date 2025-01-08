export interface SABRParameters {
    alpha: number;
    beta: number;
    rho: number;
    nu: number;
  }
  
  export interface SurfaceData {
    strikes: number[];
    maturities: number[];
    volatilities: number[][];
    modelParams?: SABRParameters;
  }
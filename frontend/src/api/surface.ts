export interface SABRParameters {
    alpha: number;
    beta: number;
    rho: number;
    nu: number;
  }
  
export interface SurfaceData {
    x: number[];  // strikes
    y: number[];  // maturities
    z: number[][]; // volatilities
    atm_line: {
        maturities: number[];
        vols: number[];
    };
    market_params: {
        spot: number;
        rate: number;
        min_strike: number;
        max_strike: number;
        min_maturity: number;
        max_maturity: number;
    };
    metrics: {
        atm_vol: number;
        skew_1m: number;
        term_structure_slope: number;
    };
    }

    export interface SurfaceResponse {
    success: boolean;
    data: SurfaceData;
    metadata: {
        spot: number;
        rate: number;
        sabr_params: SABRParameters;
    };
    }

    export interface SurfaceRequestParams {
    strikes: number[];
    maturities: number[];
    spot: number;
    rate: number;
    sabr_params: SABRParameters;
    }

    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    export async function fetchSurfaceData(params: SurfaceRequestParams): Promise<SurfaceResponse> {
    try {
        const response = await fetch(`${API_BASE_URL}/api/surface`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
        });

        if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch surface data');
        }

        return await response.json();
    } catch (error) {
        console.error('Error fetching surface data:', error);
        throw error;
    }
}
import React, { useState, useEffect } from 'react';
import VolatilitySurface from './components/VolatilitySurface';
import ParameterControls from './components/ParameterControls';
import MarketControls from './components/MarketControls';
import { fetchSurfaceData } from './api/surface';
import type { SABRParameters, SurfaceData } from './api/surface';

function App() {
  // States remain the same as before...
  const [parameters, setParameters] = useState<SABRParameters>({
    alpha: 0.2,
    beta: 0.5,
    rho: -0.3,
    nu: 0.4
  });

  const [marketParams, setMarketParams] = useState({
    spot: 100,
    rate: 0.02,
    strikeRange: [80, 120] as [number, number],
    maturityRange: [0.1, 1.0] as [number, number],
    gridDensity: {
      strikes: 21,
      maturities: 10
    }
  });

  // Other state and functions remain the same...
  const [surfaceData, setSurfaceData] = useState<SurfaceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // The existing functions (generateGrid, updateSurface, etc.) remain the same...
  const generateGrid = () => {
    const { strikeRange, maturityRange, gridDensity } = marketParams;
    
    const strikes = Array.from(
      { length: gridDensity.strikes }, 
      (_, i) => strikeRange[0] + (strikeRange[1] - strikeRange[0]) * (i / (gridDensity.strikes - 1))
    );
    
    const maturities = Array.from(
      { length: gridDensity.maturities },
      (_, i) => maturityRange[0] + (maturityRange[1] - maturityRange[0]) * (i / (gridDensity.maturities - 1))
    );

    return { strikes, maturities };
  };

  const updateSurface = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const grid = generateGrid();
      const response = await fetchSurfaceData({
        strikes: grid.strikes,
        maturities: grid.maturities,
        spot: marketParams.spot,
        rate: marketParams.rate,
        sabr_params: parameters
      });

      setSurfaceData(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch surface data');
      console.error('Error updating surface:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    updateSurface();
  }, [parameters, marketParams]);

  const handleParameterChange = (param: string, value: number) => {
    setParameters(prev => ({ ...prev, [param]: value }));
  };

  const handleMarketParamsUpdate = (updates: Partial<typeof marketParams>) => {
    setMarketParams(prev => ({ ...prev, ...updates }));
  };

  const transformedData = surfaceData ? {
    strikes: surfaceData.x,
    maturities: surfaceData.y,
    volatilities: surfaceData.z,
    modelParams: parameters
  } : null;

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Title Bar */}
      <div className="w-full bg-white shadow-sm px-4 py-3">
        <h1 className="text-2xl font-bold text-gray-900">Volatility Surface Explorer</h1>
        {surfaceData && (
          <div className="mt-1 text-sm text-gray-600">
            ATM Vol: {(surfaceData.metrics.atm_vol * 100).toFixed(2)}% | 
            Skew (1m): {(surfaceData.metrics.skew_1m * 100).toFixed(2)}% | 
            Term Structure Slope: {(surfaceData.metrics.term_structure_slope * 100).toFixed(4)}%
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex w-full min-h-[calc(100vh-80px)]">
        {/* Surface Plot Container - Fixed width */}
        <div className="w-[800px] p-4">
          <div className="bg-white rounded shadow-lg p-4 h-full">
            <div className="h-[600px]">
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <span className="text-lg text-gray-500">Loading...</span>
                </div>
              ) : transformedData && (
                <VolatilitySurface data={transformedData} />
              )}
            </div>
          </div>
        </div>

        {/* Parameters Container - Takes remaining width */}
        <div className="flex-1 p-4 space-y-4">
          <div className="bg-white rounded shadow-lg">
            <ParameterControls 
              parameters={parameters}
              onParameterChange={handleParameterChange}
            />
          </div>
          <div className="bg-white rounded shadow-lg">
            <MarketControls 
              {...marketParams}
              onUpdate={handleMarketParamsUpdate}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
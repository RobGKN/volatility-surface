import { useState, useEffect } from 'react';
import VolatilitySurface from './components/VolatilitySurface';
import { Card, CardTitle, CardContent } from './components/Card';
import { fetchSurfaceData, type SurfaceRequestParams } from './api/surface';
import './App.css';
import MarketControls from './components/MarketControls';
import { SurfaceData } from './types';

interface SABRParameters {
  alpha: number;
  beta: number;
  rho: number;
  nu: number;
}

function App() {
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

  const [surfaceData, setSurfaceData] = useState<SurfaceData | null>(null);
  const [loading, setLoading] = useState(true);

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

  const handleParameterChange = (param: keyof SABRParameters, value: number) => {
    setParameters(prev => ({ ...prev, [param]: value }));
  };

  useEffect(() => {
    const updateSurface = async () => {
      setLoading(true);
      try {
        const grid = generateGrid();
        const requestParams: SurfaceRequestParams = {
          strikes: grid.strikes,
          maturities: grid.maturities,
          spot: marketParams.spot,
          rate: marketParams.rate,
          sabr_params: parameters
        };
        
        const response = await fetchSurfaceData(requestParams);
        setSurfaceData({
          x: response.data.x,           // Changed from strikes
          y: response.data.y,           // Changed from maturities
          z: response.data.z,           // Changed from volatilities
          modelParams: parameters       // Add this if you want to pass SABR parameters
        });
      } catch (error) {
        console.error('Error fetching surface data:', error);
      } finally {
        setLoading(false);
      }
    };

    updateSurface();
  }, [parameters, marketParams]);

  return (
    <>
      <div className="header">
        <h1 className="text-2xl font-bold">Volatility Surface Explorer</h1>
      </div>

      <div className="main">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-lg">Loading...</div>
          </div>
        ) : surfaceData && (
          <VolatilitySurface data={surfaceData} />
        )}
      </div>

      <div className="sidebar">
        <Card>
          <CardTitle>SABR Parameters</CardTitle>
          <CardContent>
            {Object.entries(parameters).map(([param, value]) => (
              <div key={param} className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-300">{param.toUpperCase()}</span>
                  <span className="font-mono text-gray-400">{value.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={-1}
                  max={1}
                  step={0.01}
                  value={value}
                  onChange={(e) => handleParameterChange(param as keyof SABRParameters, parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            ))}
          </CardContent>
        </Card>

        <MarketControls
          {...marketParams}
          onUpdate={(updates) => setMarketParams(prev => ({ ...prev, ...updates }))}
        />
      </div>
    </>
  );
}

export default App;
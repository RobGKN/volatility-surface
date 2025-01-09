import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

interface SurfaceData {
  strikes: number[];
  maturities: number[];
  volatilities: number[][];
  modelParams?: {
    alpha: number;
    beta: number;
    rho: number;
    nu: number;
  };
}

interface VolatilitySurfaceProps {
  data: SurfaceData;
}

const VolatilitySurface: React.FC<VolatilitySurfaceProps> = ({ data }) => {
  const [height, setHeight] = useState('70vh');

  useEffect(() => {
    const updateHeight = () => {
      const availableHeight = window.innerHeight - 160;
      setHeight(`${Math.max(600, availableHeight)}px`);
    };

    window.addEventListener('resize', updateHeight);
    updateHeight();

    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  return (
    <div className="w-full h-full flex flex-col">
      <div 
        className="w-full flex-grow rounded-lg border border-gray-700"
        style={{ height }}
      >
        <Plot
          data={[
            {
              type: 'surface',
              x: data.strikes,
              y: data.maturities,
              z: data.volatilities,
              colorscale: 'Viridis',
              showscale: true,
              hoverongaps: false,
              hoverlabel: {
                bgcolor: '#1a1b1e',
                font: { color: '#ffffff' }
              }
            }
          ]}
          layout={{
            title: {
              text: 'Implied Volatility Surface',
              font: {
                family: 'Inter, system-ui, sans-serif',
                size: 24,
                color: '#ffffff'
              }
            },
            paper_bgcolor: '#1a1b1e',
            plot_bgcolor: '#1a1b1e',
            autosize: true,
            scene: {
              xaxis: {
                title: 'Strike',
                gridcolor: '#333333',
                showgrid: true,
                zeroline: false,
                showline: true,
                linewidth: 2,
                linecolor: '#333333',
                titlefont: { color: '#ffffff' },
                tickfont: { color: '#a0a0a0' },
                backgroundcolor: '#1a1b1e'
              },
              yaxis: {
                title: 'Time to Maturity',
                gridcolor: '#333333',
                showgrid: true,
                zeroline: false,
                showline: true,
                linewidth: 2,
                linecolor: '#333333',
                titlefont: { color: '#ffffff' },
                tickfont: { color: '#a0a0a0' },
                backgroundcolor: '#1a1b1e'
              },
              zaxis: {
                title: 'Implied Volatility',
                gridcolor: '#333333',
                showgrid: true,
                zeroline: false,
                showline: true,
                linewidth: 2,
                linecolor: '#333333',
                titlefont: { color: '#ffffff' },
                tickfont: { color: '#a0a0a0' },
                backgroundcolor: '#1a1b1e'
              },
              camera: {
                eye: { x: 1.8, y: 1.8, z: 1.4 }
              },
              aspectratio: { x: 1.2, y: 1, z: 0.8 }
            },
            margin: {
              t: 50,
              b: 30,
              l: 30,
              r: 50
            }
          }}
          config={{
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
          }}
          style={{
            width: '100%',
            height: '100%'
          }}
        />
      </div>
    </div>
  );
};

export default VolatilitySurface;
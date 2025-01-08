import React from 'react';
import Plot from 'react-plotly.js';
import { SurfaceData } from '../types/surface';

interface VolatilitySurfaceProps {
  data: SurfaceData;
}

const VolatilitySurface: React.FC<VolatilitySurfaceProps> = ({ data }) => {
  return (
    <div className="w-full h-full min-h-[600px]">
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
              bgcolor: '#FFF',
              font: { color: '#000' }
            }
          }
        ]}
        layout={{
            title: {
            text: 'Implied Volatility Surface',
            font: {
                family: 'Inter, system-ui, sans-serif',
                size: 24,
                color: '#111827'
            }
            },
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
            xaxis: { 
                title: 'Strike',
                gridcolor: '#E5E7EB',
                showgrid: true,
                zeroline: false
            },
            yaxis: { 
                title: 'Time to Maturity',
                gridcolor: '#E5E7EB',
                showgrid: true,
                zeroline: false
            },
            zaxis: { 
                title: 'Implied Volatility',
                gridcolor: '#E5E7EB',
                showgrid: true,
                zeroline: false
            },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.2 }
            },
            aspectratio: { x: 1, y: 1, z: 0.7 }
            },
            margin: { t: 50, b: 20, l: 20, r: 20 }
        }}
      />
    </div>
  );
};

export default VolatilitySurface;
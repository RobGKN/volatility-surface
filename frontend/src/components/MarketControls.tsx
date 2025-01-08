import React from 'react';

interface MarketControlsProps {
  spot: number;
  rate: number;
  strikeRange: [number, number];
  maturityRange: [number, number];
  gridDensity: {
    strikes: number;
    maturities: number;
  };
  onUpdate: (params: {
    spot?: number;
    rate?: number;
    strikeRange?: [number, number];
    maturityRange?: [number, number];
    gridDensity?: {
      strikes: number;
      maturities: number;
    };
  }) => void;
}

const MarketControls: React.FC<MarketControlsProps> = ({
  spot,
  rate,
  strikeRange,
  maturityRange,
  gridDensity,
  onUpdate
}) => {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-4">Market Parameters</h2>
      
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Spot Price</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={spot}
              onChange={(e) => onUpdate({ spot: Number(e.target.value) })}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Strike Range (% of Spot)</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={((strikeRange[0] / spot) * 100).toFixed(0)}
              onChange={(e) => {
                const percent = Number(e.target.value);
                onUpdate({ 
                  strikeRange: [spot * (percent / 100), strikeRange[1]] 
                });
              }}
              className="w-20 p-2 border rounded"
            />
            <span className="self-center">to</span>
            <input
              type="number"
              value={((strikeRange[1] / spot) * 100).toFixed(0)}
              onChange={(e) => {
                const percent = Number(e.target.value);
                onUpdate({ 
                  strikeRange: [strikeRange[0], spot * (percent / 100)] 
                });
              }}
              className="w-20 p-2 border rounded"
            />
            <span className="self-center">%</span>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Maturity Range (months)</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={(maturityRange[0] * 12).toFixed(1)}
              onChange={(e) => {
                const months = Number(e.target.value);
                onUpdate({ 
                  maturityRange: [months / 12, maturityRange[1]] 
                });
              }}
              className="w-20 p-2 border rounded"
            />
            <span className="self-center">to</span>
            <input
              type="number"
              value={(maturityRange[1] * 12).toFixed(1)}
              onChange={(e) => {
                const months = Number(e.target.value);
                onUpdate({ 
                  maturityRange: [maturityRange[0], months / 12] 
                });
              }}
              className="w-20 p-2 border rounded"
            />
            <span className="self-center">months</span>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Grid Points</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={gridDensity.strikes}
              onChange={(e) => onUpdate({ 
                gridDensity: { ...gridDensity, strikes: Number(e.target.value) }
              })}
              min={5}
              max={50}
              className="w-20 p-2 border rounded"
            />
            <span className="self-center">×</span>
            <input
              type="number"
              value={gridDensity.maturities}
              onChange={(e) => onUpdate({ 
                gridDensity: { ...gridDensity, maturities: Number(e.target.value) }
              })}
              min={5}
              max={50}
              className="w-20 p-2 border rounded"
            />
          </div>
          <div className="text-xs text-gray-500">
            (strikes × maturities)
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Risk-Free Rate (%)</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={(rate * 100).toFixed(2)}
              onChange={(e) => onUpdate({ rate: Number(e.target.value) / 100 })}
              step="0.1"
              className="w-full p-2 border rounded"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketControls;
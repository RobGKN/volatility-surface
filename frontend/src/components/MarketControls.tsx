import React from 'react';
import { Card, CardTitle, CardContent } from './Card';

interface MarketControlsProps {
  spot: number;
  rate: number;
  strikeRange: [number, number];
  maturityRange: [number, number];
  gridDensity: {
    strikes: number;
    maturities: number;
  };
  onUpdate: (params: Partial<{
    spot: number;
    rate: number;
    strikeRange: [number, number];
    maturityRange: [number, number];
    gridDensity: {
      strikes: number;
      maturities: number;
    };
  }>) => void;
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
    <Card>
      <CardTitle>Market Parameters</CardTitle>
      <CardContent>
        <div className="space-y-4">
          {/* Spot Price */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Spot Price</span>
              <span className="font-mono text-gray-400">{spot}</span>
            </div>
            <input
              type="range"
              min={50}
              max={150}
              step={1}
              value={spot}
              onChange={(e) => onUpdate({ spot: parseFloat(e.target.value) })}
              className="w-full"
            />
          </div>

          {/* Strike Range */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Strike Range (% of Spot)</span>
              <span className="font-mono text-gray-400">
                {((strikeRange[0] / spot) * 100).toFixed(0)}% - {((strikeRange[1] / spot) * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex gap-4">
              <input
                type="range"
                min={50}
                max={99}
                value={((strikeRange[0] / spot) * 100)}
                onChange={(e) => {
                  const percent = parseFloat(e.target.value);
                  onUpdate({ strikeRange: [spot * (percent / 100), strikeRange[1]] });
                }}
                className="w-full"
              />
              <input
                type="range"
                min={101}
                max={150}
                value={((strikeRange[1] / spot) * 100)}
                onChange={(e) => {
                  const percent = parseFloat(e.target.value);
                  onUpdate({ strikeRange: [strikeRange[0], spot * (percent / 100)] });
                }}
                className="w-full"
              />
            </div>
          </div>

          {/* Maturity Range */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Maturity Range (months)</span>
              <span className="font-mono text-gray-400">
                {(maturityRange[0] * 12).toFixed(1)} - {(maturityRange[1] * 12).toFixed(1)}
              </span>
            </div>
            <div className="flex gap-4">
              <input
                type="range"
                min={1}
                max={11}
                step={0.1}
                value={maturityRange[0] * 12}
                onChange={(e) => {
                  const months = parseFloat(e.target.value);
                  onUpdate({ maturityRange: [months / 12, maturityRange[1]] });
                }}
                className="w-full"
              />
              <input
                type="range"
                min={12}
                max={24}
                step={0.1}
                value={maturityRange[1] * 12}
                onChange={(e) => {
                  const months = parseFloat(e.target.value);
                  onUpdate({ maturityRange: [maturityRange[0], months / 12] });
                }}
                className="w-full"
              />
            </div>
          </div>

          {/* Risk-Free Rate */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Risk-Free Rate (%)</span>
              <span className="font-mono text-gray-400">{(rate * 100).toFixed(2)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={10}
              step={0.1}
              value={rate * 100}
              onChange={(e) => onUpdate({ rate: parseFloat(e.target.value) / 100 })}
              className="w-full"
            />
          </div>

          {/* Grid Density */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Grid Points</span>
              <span className="font-mono text-gray-400">
                {gridDensity.strikes} × {gridDensity.maturities}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-xs text-gray-400">Strikes</span>
                <input
                  type="range"
                  min={5}
                  max={50}
                  value={gridDensity.strikes}
                  onChange={(e) => onUpdate({
                    gridDensity: { ...gridDensity, strikes: parseInt(e.target.value) }
                  })}
                  className="w-full"
                />
              </div>
              <div>
                <span className="text-xs text-gray-400">Maturities</span>
                <input
                  type="range"
                  min={5}
                  max={50}
                  value={gridDensity.maturities}
                  onChange={(e) => onUpdate({
                    gridDensity: { ...gridDensity, maturities: parseInt(e.target.value) }
                  })}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MarketControls;
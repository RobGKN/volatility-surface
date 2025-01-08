import React from 'react';

interface ParameterControlsProps {
  parameters: {
    alpha: number;
    beta: number;
    rho: number;
    nu: number;
  };
  onParameterChange: (param: string, value: number) => void;
}

const ParameterControls: React.FC<ParameterControlsProps> = ({ parameters, onParameterChange }) => {
  const parameterConfig = {
    alpha: { min: 0.01, max: 1, step: 0.01, label: 'Alpha (α) - Initial Volatility' },
    beta: { min: 0, max: 1, step: 0.1, label: 'Beta (β) - CEV Parameter' },
    rho: { min: -1, max: 1, step: 0.1, label: 'Rho (ρ) - Correlation' },
    nu: { min: 0, max: 1, step: 0.05, label: 'Nu (ν) - Vol of Vol' }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">SABR Parameters</h2>
      <div className="space-y-4">
        {(Object.keys(parameters) as Array<keyof typeof parameters>).map((param) => (
          <div key={param} className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              {parameterConfig[param].label}
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={parameterConfig[param].min}
                max={parameterConfig[param].max}
                step={parameterConfig[param].step}
                value={parameters[param]}
                onChange={(e) => onParameterChange(param, parseFloat(e.target.value))}
                className="flex-1"
              />
              <input
                type="number"
                value={parameters[param]}
                onChange={(e) => onParameterChange(param, parseFloat(e.target.value))}
                min={parameterConfig[param].min}
                max={parameterConfig[param].max}
                step={parameterConfig[param].step}
                className="w-20 p-1 border rounded text-right"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ParameterControls;
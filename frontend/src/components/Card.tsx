import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = '' }) => (
  <div className={`bg-gray-800 rounded-lg shadow-lg p-4 ${className}`}>
    {children}
  </div>
);

export const CardTitle: React.FC<CardProps> = ({ children, className = '' }) => (
  <h2 className={`text-xl font-bold mb-4 ${className}`}>{children}</h2>
);

export const CardContent: React.FC<CardProps> = ({ children, className = '' }) => (
  <div className={`${className}`}>{children}</div>
);
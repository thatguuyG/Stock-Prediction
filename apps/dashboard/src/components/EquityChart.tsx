'use client';

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { EquityPoint } from '@/types/api';

export function EquityChart({ data }: { data: EquityPoint[] }) {
  if (data.length === 0) {
    return <div className="text-sm opacity-60">No equity data yet — run <code>stockpred reconcile</code>.</div>;
  }
  const series = data.map((d) => ({ ts: new Date(d.ts).toLocaleDateString(), equity: d.equity }));
  return (
    <div className="w-full h-96 border border-neutral-300 dark:border-neutral-800 rounded p-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis dataKey="ts" fontSize={11} />
          <YAxis fontSize={11} />
          <Tooltip />
          <Line type="monotone" dataKey="equity" stroke="#16a34a" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

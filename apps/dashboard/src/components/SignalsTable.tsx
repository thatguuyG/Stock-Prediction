'use client';

import { useState } from 'react';
import type { Signal } from '@/types/api';

const BADGE: Record<Signal['decision'], string> = {
  BUY: 'bg-green-600 text-white',
  SELL: 'bg-red-600 text-white',
  HOLD: 'bg-neutral-400 text-white',
};

export function SignalsTable({ signals }: { signals: Signal[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  if (signals.length === 0) {
    return <div className="text-sm opacity-60">No signals yet — run <code>stockpred run-signals</code>.</div>;
  }
  return (
    <table className="w-full text-sm numeric border-collapse">
      <thead>
        <tr className="border-b border-neutral-300 dark:border-neutral-800 text-left">
          <th className="py-2 pr-4">Date</th>
          <th className="py-2 pr-4">Symbol</th>
          <th className="py-2 pr-4">Decision</th>
          <th className="py-2 pr-4 text-right">Score</th>
          <th className="py-2 pr-4">Reason</th>
          <th className="py-2"></th>
        </tr>
      </thead>
      <tbody>
        {signals.map((s) => {
          const reason = (s.rationale as { reason?: string })?.reason ?? '?';
          return (
            <>
              <tr key={s.id} className="border-b border-neutral-200 dark:border-neutral-900">
                <td className="py-2 pr-4">{new Date(s.ts).toLocaleDateString()}</td>
                <td className="py-2 pr-4 font-semibold">{s.symbol}</td>
                <td className="py-2 pr-4">
                  <span className={`px-2 py-0.5 rounded text-xs ${BADGE[s.decision]}`}>{s.decision}</span>
                </td>
                <td className="py-2 pr-4 text-right">{s.score.toFixed(3)}</td>
                <td className="py-2 pr-4 opacity-70">{reason}</td>
                <td className="py-2">
                  <button
                    className="text-xs underline opacity-60"
                    onClick={() => setExpanded(expanded === s.id ? null : s.id)}
                  >
                    {expanded === s.id ? 'hide' : 'rationale'}
                  </button>
                </td>
              </tr>
              {expanded === s.id && (
                <tr key={`${s.id}-rationale`}>
                  <td colSpan={6} className="py-2 pr-4 bg-neutral-100 dark:bg-neutral-900">
                    <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(s.rationale, null, 2)}</pre>
                  </td>
                </tr>
              )}
            </>
          );
        })}
      </tbody>
    </table>
  );
}

import type { EquityPoint } from '@/types/api';

export function SummaryCard({ risk }: { risk: EquityPoint | null }) {
  if (!risk) {
    return (
      <div className="border border-neutral-300 dark:border-neutral-800 rounded p-4 text-sm">
        No risk state yet — run <code>stockpred reconcile</code>.
      </div>
    );
  }
  const ts = new Date(risk.ts).toLocaleString();
  return (
    <div className="border border-neutral-300 dark:border-neutral-800 rounded p-4 grid grid-cols-2 sm:grid-cols-4 gap-4 numeric text-sm">
      <div><div className="text-xs uppercase opacity-60">Equity</div><div className="text-lg">${risk.equity.toLocaleString()}</div></div>
      <div><div className="text-xs uppercase opacity-60">Cash</div><div className="text-lg">${risk.cash.toLocaleString()}</div></div>
      <div><div className="text-xs uppercase opacity-60">Exposure</div><div className="text-lg">{risk.exposure_pct.toFixed(1)}%</div></div>
      <div><div className="text-xs uppercase opacity-60">Status</div><div className="text-lg">{risk.halted ? '🛑 HALTED' : '✓ live'}</div></div>
      <div className="col-span-2 sm:col-span-4 text-xs opacity-60">as of {ts}</div>
    </div>
  );
}

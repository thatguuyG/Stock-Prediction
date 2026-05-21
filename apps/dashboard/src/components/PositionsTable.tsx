import type { Position } from '@/types/api';

export function PositionsTable({ positions }: { positions: Position[] }) {
  if (positions.length === 0) {
    return <div className="text-sm opacity-60">No open positions.</div>;
  }
  return (
    <table className="w-full text-sm numeric border-collapse">
      <thead>
        <tr className="border-b border-neutral-300 dark:border-neutral-800 text-left">
          <th className="py-2 pr-4">Symbol</th>
          <th className="py-2 pr-4 text-right">Qty</th>
          <th className="py-2 pr-4 text-right">Avg Price</th>
          <th className="py-2 pr-4 text-right">Notional</th>
          <th className="py-2 pr-4">Source</th>
          <th className="py-2">Updated</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((p) => (
          <tr key={p.symbol} className="border-b border-neutral-200 dark:border-neutral-900">
            <td className="py-2 pr-4 font-semibold">{p.symbol}</td>
            <td className="py-2 pr-4 text-right">{p.qty.toLocaleString()}</td>
            <td className="py-2 pr-4 text-right">${p.avg_price.toFixed(2)}</td>
            <td className="py-2 pr-4 text-right">${(p.qty * p.avg_price).toLocaleString()}</td>
            <td className="py-2 pr-4 opacity-70">{p.source}</td>
            <td className="py-2 opacity-70">{new Date(p.updated_at).toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

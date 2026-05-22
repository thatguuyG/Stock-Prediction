import { api } from '@/lib/api';
import { PositionsTable } from '@/components/PositionsTable';
import { SummaryCard } from '@/components/SummaryCard';

export default async function PositionsPage() {
  const [positions, equity] = await Promise.all([
    api.positions().catch(() => []),
    api.equity().catch(() => []),
  ]);
  const latestRisk = equity.length > 0 ? equity[equity.length - 1] : null;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Portfolio</h2>
      <SummaryCard risk={latestRisk} />
      <h3 className="text-lg font-semibold">Open positions</h3>
      <PositionsTable positions={positions} />
    </div>
  );
}

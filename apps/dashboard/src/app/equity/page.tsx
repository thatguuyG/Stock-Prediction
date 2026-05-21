import { api } from '@/lib/api';
import { EquityChart } from '@/components/EquityChart';

export default async function EquityPage() {
  const data = await api.equity().catch(() => []);
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Equity curve</h2>
      <EquityChart data={data} />
    </div>
  );
}

import { api } from '@/lib/api';
import { SignalsTable } from '@/components/SignalsTable';

export default async function SignalsPage() {
  const signals = await api.signals({ limit: 50 }).catch(() => []);
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Recent signals</h2>
      <SignalsTable signals={signals} />
    </div>
  );
}

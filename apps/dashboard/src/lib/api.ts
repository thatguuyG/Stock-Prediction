import type { EquityPoint, Order, Position, Signal } from '@/types/api';

const BASE = '/api';

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  positions: () => getJson<Position[]>('/positions'),
  signals: (params: { limit?: number; decision?: 'BUY' | 'SELL' | 'HOLD' } = {}) => {
    const q = new URLSearchParams();
    if (params.limit) q.set('limit', String(params.limit));
    if (params.decision) q.set('decision', params.decision);
    const qs = q.toString();
    return getJson<Signal[]>(`/signals${qs ? `?${qs}` : ''}`);
  },
  orders: (params: { limit?: number; status?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.limit) q.set('limit', String(params.limit));
    if (params.status) q.set('status', params.status);
    const qs = q.toString();
    return getJson<Order[]>(`/orders${qs ? `?${qs}` : ''}`);
  },
  equity: (params: { from?: string; to?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.from) q.set('from', params.from);
    if (params.to) q.set('to', params.to);
    const qs = q.toString();
    return getJson<EquityPoint[]>(`/equity${qs ? `?${qs}` : ''}`);
  },
};

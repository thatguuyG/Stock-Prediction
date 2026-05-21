// Response models mirroring services/api/schemas.py.
// Keep this hand-rolled until/unless we wire openapi-typescript generation.

export interface Position {
  symbol: string;
  qty: number;
  avg_price: number;
  updated_at: string;
  source: string;
}

export interface Signal {
  id: number;
  symbol: string;
  ts: string;
  model_version: string;
  score: number;
  decision: 'BUY' | 'SELL' | 'HOLD';
  rationale: Record<string, unknown>;
  created_at: string;
}

export interface Order {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  order_type: string;
  limit_price: number | null;
  stop_price: number | null;
  take_profit: number | null;
  status: string;
  submitted_at: string;
  filled_at: string | null;
  broker_order_id: string | null;
  signal_id: number | null;
}

export interface EquityPoint {
  ts: string;
  equity: number;
  cash: number;
  exposure_pct: number;
  max_drawdown: number;
  n_open_positions: number;
  halted: boolean;
}

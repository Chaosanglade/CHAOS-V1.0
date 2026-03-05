# CHAOS V1.0 — RL Meta-Controller Architecture

## Design Principle
RL sits ABOVE the proven alpha layer. It controls SYSTEM BEHAVIOR, not price prediction.

## Three Layers
1. **Alpha Layer** (built): Ensemble models (H1/M30) generating directional signals
2. **Execution & Risk Layer** (built): Risk engine, position sizing, cooldown
3. **RL Meta-Controller** (V4): Learns when to trust/restrict the system

## RL State (Observation Vector)
- regime_state, regime_confidence, realized_volatility, trend_strength
- spread_ratio, liquidity_proxy
- vote_entropy, vote_confidence, agreement_score, enabled_models_count
- open_positions, USD_exposure, current_drawdown, loss_streak, cooldown_flags
- fill_ratio, recent_slippage, spread_spike_flag

## RL Actions (Small, Safe)
- TF execution mode: H1 only | H1+M30
- Agreement threshold adjustment: -0.05 to +0.05
- Risk scalar: 0.50, 0.75, 1.00
- Pair activation mask: Tier1 only | Tier1+Tier2

## Reward Function
reward = pnl_change - 3*drawdown_increase - 0.5*turnover_cost - 5*safety_violation

## Training
- Environment: Replay engine (deterministic)
- Algorithm: PPO (Stable-Baselines3)
- Scenario randomization across all 4 cost scenarios
- Walk-forward training windows
- Action smoothing: risk_scalar change <= 10%/day, threshold change <= 0.03/hour

## Deployment Phases
1. Shadow mode (log actions, don't execute)
2. Phase 1: agreement threshold control only
3. Phase 2: TF execution mode
4. Phase 3: risk scalar
5. Full control (never give all at once)

## Prerequisites
- Walk-forward validation PASS
- Paper trading PASS
- Edge decay monitor GREEN across all pairs

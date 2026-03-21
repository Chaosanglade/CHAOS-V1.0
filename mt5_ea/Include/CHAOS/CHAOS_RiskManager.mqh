//+------------------------------------------------------------------+
//|                                         CHAOS_RiskManager.mqh    |
//|                        CHAOS V1.0 - EA-side Risk Safety Net      |
//|                        Mirrors Python risk engine locally         |
//+------------------------------------------------------------------+
#property copyright "CHAOS V1.0"
#property version   "1.00"
#property strict

#ifndef CHAOS_RISKMANAGER_MQH
#define CHAOS_RISKMANAGER_MQH

#include "CHAOS_Config.mqh"
#include "CHAOS_Logger.mqh"

//+------------------------------------------------------------------+
//| Per (pair, TF) cooldown state                                     |
//+------------------------------------------------------------------+
struct CooldownState {
   string   pair;
   string   tf;
   int      barsRemaining;      // bars left in cooldown
   int      consecutiveLosses;  // running loss streak
};

//+------------------------------------------------------------------+
//| Risk manager globals                                              |
//+------------------------------------------------------------------+
CooldownState g_cooldowns[];       // dynamic array of cooldown states
double        g_dayStartBalance;   // equity snapshot at day start
bool          g_defensiveMode;     // spread spike defensive mode
bool          g_ddBreakerTripped;  // drawdown breaker active
string        g_ddBreakerReason;   // reason string for breaker

//+------------------------------------------------------------------+
//| InitRiskManager — call once in OnInit()                           |
//+------------------------------------------------------------------+
void InitRiskManager() {
   g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_defensiveMode   = false;
   g_ddBreakerTripped = false;
   g_ddBreakerReason = "";
   ArrayResize(g_cooldowns, 0);
   LogInfo("RiskMgr", StringFormat("Initialised. Day-start balance=%.2f", g_dayStartBalance));
}

//+------------------------------------------------------------------+
//| ResetDailyState — call at start of new trading day                |
//+------------------------------------------------------------------+
void ResetDailyState() {
   g_dayStartBalance  = AccountInfoDouble(ACCOUNT_BALANCE);
   g_ddBreakerTripped = false;
   g_ddBreakerReason  = "";
   LogInfo("RiskMgr", StringFormat("Daily reset. New day-start balance=%.2f", g_dayStartBalance));
}

//+------------------------------------------------------------------+
//| CountOpenPositions — count positions with our magic number        |
//+------------------------------------------------------------------+
int CountOpenPositions() {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == CHAOS_MAGIC_NUMBER)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| CountPositionsForPair — per-pair position count                   |
//+------------------------------------------------------------------+
int CountPositionsForPair(const string pair) {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != CHAOS_MAGIC_NUMBER) continue;
      if(PositionGetString(POSITION_SYMBOL) == pair)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| CalculateIntradayDrawdown — current DD from day-start balance     |
//| Returns positive number representing drawdown percentage          |
//+------------------------------------------------------------------+
double CalculateIntradayDrawdown() {
   if(g_dayStartBalance <= 0) return 0.0;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (g_dayStartBalance - equity) / g_dayStartBalance * 100.0;
   return (dd > 0) ? dd : 0.0;
}

//+------------------------------------------------------------------+
//| FindCooldownIndex — locate (pair, tf) entry, -1 if not found      |
//+------------------------------------------------------------------+
int FindCooldownIndex(const string pair, const string tf) {
   for(int i = 0; i < ArraySize(g_cooldowns); i++) {
      if(g_cooldowns[i].pair == pair && g_cooldowns[i].tf == tf)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| EnsureCooldownEntry — get or create a (pair, tf) entry            |
//+------------------------------------------------------------------+
int EnsureCooldownEntry(const string pair, const string tf) {
   int idx = FindCooldownIndex(pair, tf);
   if(idx >= 0) return idx;

   int sz = ArraySize(g_cooldowns);
   ArrayResize(g_cooldowns, sz + 1);
   g_cooldowns[sz].pair             = pair;
   g_cooldowns[sz].tf               = tf;
   g_cooldowns[sz].barsRemaining    = 0;
   g_cooldowns[sz].consecutiveLosses = 0;
   return sz;
}

//+------------------------------------------------------------------+
//| IsInCooldown — check if (pair, tf) is currently cooling down      |
//+------------------------------------------------------------------+
bool IsInCooldown(const string pair, const string tf) {
   int idx = FindCooldownIndex(pair, tf);
   if(idx < 0) return false;
   return (g_cooldowns[idx].barsRemaining > 0);
}

//+------------------------------------------------------------------+
//| SetCooldown — set cooldown bars for (pair, tf)                    |
//+------------------------------------------------------------------+
void SetCooldown(const string pair, const string tf, int bars) {
   int idx = EnsureCooldownEntry(pair, tf);
   g_cooldowns[idx].barsRemaining = bars;
   LogInfo("RiskMgr", StringFormat("Cooldown set: %s %s = %d bars", pair, tf, bars));
}

//+------------------------------------------------------------------+
//| DecrementCooldowns — call once per bar on each active TF          |
//+------------------------------------------------------------------+
void DecrementCooldowns(const string tf) {
   for(int i = 0; i < ArraySize(g_cooldowns); i++) {
      if(g_cooldowns[i].tf == tf && g_cooldowns[i].barsRemaining > 0) {
         g_cooldowns[i].barsRemaining--;
      }
   }
}

//+------------------------------------------------------------------+
//| RecordTradeResult — track consecutive losses per (pair, tf)       |
//| Call after each trade closes. pnl < 0 = loss.                    |
//+------------------------------------------------------------------+
void RecordTradeResult(const string pair, const string tf, double pnl) {
   int idx = EnsureCooldownEntry(pair, tf);

   if(pnl < 0) {
      g_cooldowns[idx].consecutiveLosses++;
      if(g_cooldowns[idx].consecutiveLosses >= CONSECUTIVE_LOSS_THRESHOLD) {
         SetCooldown(pair, tf, COOLDOWN_BARS_LOSS_STREAK);
         LogWarning("RiskMgr", StringFormat(
            "%s %s hit %d consecutive losses -> %d bar cooldown",
            pair, tf, g_cooldowns[idx].consecutiveLosses, COOLDOWN_BARS_LOSS_STREAK));
         g_cooldowns[idx].consecutiveLosses = 0;  // reset after trigger
      }
   } else {
      // Win or breakeven resets the streak
      g_cooldowns[idx].consecutiveLosses = 0;
   }
}

//+------------------------------------------------------------------+
//| GetBaselineSpread — returns typical spread in pips for a pair     |
//+------------------------------------------------------------------+
double GetBaselineSpread(const string pair) {
   return GetBaselineSpreadPips(pair);  // delegates to Config helper
}

//+------------------------------------------------------------------+
//| GetCurrentSpreadPips — live spread in pips                        |
//+------------------------------------------------------------------+
double GetCurrentSpreadPips(const string pair) {
   double ask = SymbolInfoDouble(pair, SYMBOL_ASK);
   double bid = SymbolInfoDouble(pair, SYMBOL_BID);
   double point = SymbolInfoDouble(pair, SYMBOL_POINT);
   if(point <= 0) return 0;

   int digits = (int)SymbolInfoInteger(pair, SYMBOL_DIGITS);
   double pipSize = (digits == 3 || digits == 5) ? point * 10.0 : point;
   if(pipSize <= 0) return 0;

   return (ask - bid) / pipSize;
}

//+------------------------------------------------------------------+
//| CheckSpreadSpike — true if spread > 2.5x baseline                 |
//+------------------------------------------------------------------+
bool CheckSpreadSpike(const string pair) {
   double current  = GetCurrentSpreadPips(pair);
   double baseline = GetBaselineSpread(pair);
   if(baseline <= 0) return false;
   return (current > baseline * SPREAD_SPIKE_MULT);
}

//+------------------------------------------------------------------+
//| GetActiveCooldownCount — number of (pair, tf) combos in cooldown  |
//+------------------------------------------------------------------+
int GetActiveCooldownCount() {
   int count = 0;
   for(int i = 0; i < ArraySize(g_cooldowns); i++) {
      if(g_cooldowns[i].barsRemaining > 0) count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| LocalRiskCheck — master check combining all safety gates          |
//| Returns true if trade is ALLOWED, false if blocked                |
//| Sets reason string explaining the block                           |
//+------------------------------------------------------------------+
bool LocalRiskCheck(const string pair, const string tf,
                    const string signal, string &reason) {
   reason = "";

   //--- 1. Drawdown breaker
   double dd = CalculateIntradayDrawdown();
   if(dd >= DRAWDOWN_BREAKER_PCT) {
      g_ddBreakerTripped = true;
      g_ddBreakerReason = StringFormat("DD=%.2f%% >= %.1f%%", dd, DRAWDOWN_BREAKER_PCT);
      reason = "DRAWDOWN_BREAKER: " + g_ddBreakerReason;
      LogWarning("RiskMgr", reason);
      return false;
   }
   if(g_ddBreakerTripped) {
      reason = "DRAWDOWN_BREAKER_LATCHED: " + g_ddBreakerReason;
      return false;
   }

   //--- 2. Max total positions
   int totalPos = CountOpenPositions();
   if(totalPos >= MAX_TOTAL_POSITIONS) {
      reason = StringFormat("MAX_POSITIONS: %d/%d", totalPos, MAX_TOTAL_POSITIONS);
      return false;
   }

   //--- 3. Max per pair
   int pairPos = CountPositionsForPair(pair);
   if(pairPos >= MAX_POSITIONS_PER_PAIR) {
      reason = StringFormat("MAX_PER_PAIR: %s has %d/%d", pair, pairPos, MAX_POSITIONS_PER_PAIR);
      return false;
   }

   //--- 4. Cooldown check
   if(IsInCooldown(pair, tf)) {
      int idx = FindCooldownIndex(pair, tf);
      int barsLeft = (idx >= 0) ? g_cooldowns[idx].barsRemaining : 0;
      reason = StringFormat("COOLDOWN: %s %s has %d bars remaining", pair, tf, barsLeft);
      return false;
   }

   //--- 5. Spread spike
   if(CheckSpreadSpike(pair)) {
      double current  = GetCurrentSpreadPips(pair);
      double baseline = GetBaselineSpread(pair);
      g_defensiveMode = true;
      reason = StringFormat("SPREAD_SPIKE: %s spread=%.1f > %.1f x %.1f baseline",
                            pair, current, SPREAD_SPIKE_MULT, baseline);
      return false;
   }
   g_defensiveMode = false;

   //--- 6. TF role check — confirm-only TFs cannot execute
   ENUM_TIMEFRAMES tfEnum = StringToTF(tf);
   if(tfEnum != PERIOD_CURRENT && !IsTFTradeEnabled(tfEnum)) {
      reason = StringFormat("TF_CONFIRM_ONLY: %s is not trade-enabled", tf);
      return false;
   }

   //--- 7. Signal must be Long or Short (not Flat)
   if(signal == "FLAT" || signal == "flat" || signal == "0") {
      reason = "SIGNAL_FLAT: no trade on flat signal";
      return false;
   }

   // All checks passed
   return true;
}

//+------------------------------------------------------------------+
//| StringToTF — convert string TF name to ENUM_TIMEFRAMES           |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTF(const string tf) {
   if(tf == "M1")  return PERIOD_M1;
   if(tf == "M5")  return PERIOD_M5;
   if(tf == "M15") return PERIOD_M15;
   if(tf == "M30") return PERIOD_M30;
   if(tf == "H1")  return PERIOD_H1;
   if(tf == "H4")  return PERIOD_H4;
   if(tf == "D1")  return PERIOD_D1;
   if(tf == "W1")  return PERIOD_W1;
   if(tf == "MN1") return PERIOD_MN1;
   return PERIOD_CURRENT;
}

#endif // CHAOS_RISKMANAGER_MQH

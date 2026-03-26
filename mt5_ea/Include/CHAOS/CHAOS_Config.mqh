//+------------------------------------------------------------------+
//|                                              CHAOS_Config.mqh    |
//|                        CHAOS V1.0 - Configuration Constants      |
//|                        Thin EA -> Python ZMQ inference pipeline   |
//+------------------------------------------------------------------+
#property copyright "CHAOS V1.0"
#property version   "1.00"
#property strict

#ifndef CHAOS_CONFIG_MQH
#define CHAOS_CONFIG_MQH

//+------------------------------------------------------------------+
//| EA Identity                                                       |
//+------------------------------------------------------------------+
#define CHAOS_MAGIC_NUMBER     20260306
#define CHAOS_MAGIC            CHAOS_MAGIC_NUMBER
#define CHAOS_EA_COMMENT       "CHAOS_V1"

//+------------------------------------------------------------------+
//| ZeroMQ Connection                                                 |
//+------------------------------------------------------------------+
#define ZMQ_DEFAULT_ENDPOINT   "tcp://127.0.0.1:5555"
#define ZMQ_RECV_TIMEOUT_MS    5000
#define ZMQ_SEND_TIMEOUT_MS    2000
#define ZMQ_HEARTBEAT_SEC      10

//+------------------------------------------------------------------+
//| Universe — 9 pairs                                                |
//+------------------------------------------------------------------+
#define CHAOS_NUM_PAIRS        9

const string CHAOS_PAIRS[] = {
   "EURUSD", "GBPUSD", "USDJPY",
   "AUDUSD", "USDCAD", "USDCHF",
   "NZDUSD", "EURJPY", "GBPJPY"
};

//+------------------------------------------------------------------+
//| Timeframe Roles                                                   |
//|   TRADE_ENABLED  — signals can open/close positions               |
//|   CONFIRM_ONLY   — signals feed ensemble agreement, no execution  |
//+------------------------------------------------------------------+
enum ENUM_TF_ROLE {
   TF_ROLE_TRADE_ENABLED = 0,
   TF_ROLE_CONFIRM_ONLY  = 1,
   TF_ROLE_DISABLED      = 2
};

#define CHAOS_NUM_TIMEFRAMES   4

const ENUM_TIMEFRAMES CHAOS_TIMEFRAMES[] = {
   PERIOD_H1,
   PERIOD_M30,
   PERIOD_M15,
   PERIOD_M5
};

const ENUM_TF_ROLE CHAOS_TF_ROLES[] = {
   TF_ROLE_TRADE_ENABLED,   // H1
   TF_ROLE_TRADE_ENABLED,   // M30
   TF_ROLE_CONFIRM_ONLY,    // M15
   TF_ROLE_CONFIRM_ONLY     // M5
};

const string CHAOS_TF_NAMES[] = {
   "H1", "M30", "M15", "M5"
};

//+------------------------------------------------------------------+
//| Position / Exposure Limits                                        |
//+------------------------------------------------------------------+
#define MAX_TOTAL_POSITIONS    8
#define MAX_POSITIONS_PER_PAIR 1

//+------------------------------------------------------------------+
//| Risk Parameters                                                   |
//+------------------------------------------------------------------+
#define RISK_PCT_PER_TRADE     0.50    // 0.50% of equity per trade
#define DRAWDOWN_BREAKER_PCT   2.0     // 2.0% intraday DD triggers halt
#define SPREAD_SPIKE_MULT      2.5     // spread > 2.5x baseline = defensive

//+------------------------------------------------------------------+
//| Cooldown Settings                                                 |
//+------------------------------------------------------------------+
#define COOLDOWN_BARS_AFTER_EXIT     1     // 1 bar cooldown after exit
#define COOLDOWN_BARS_LOSS_STREAK    30    // 30 bar cooldown after 5 consecutive losses
#define CONSECUTIVE_LOSS_THRESHOLD   5     // losses before streak cooldown fires

//+------------------------------------------------------------------+
//| Data Settings                                                     |
//+------------------------------------------------------------------+
#define BARS_TO_SEND           500    // bars per TF sent to Python server
#define CHAOS_BARS_TO_SEND     BARS_TO_SEND

//+------------------------------------------------------------------+
//| Baseline Spreads (pips) — used for spread spike detection         |
//|  Values represent typical IBKR RAW spread during London/NY        |
//+------------------------------------------------------------------+
double GetBaselineSpreadPips(const string pair) {
   if(pair == "EURUSD") return 0.3;
   if(pair == "GBPUSD") return 0.5;
   if(pair == "USDJPY") return 0.3;
   if(pair == "AUDUSD") return 0.4;
   if(pair == "USDCAD") return 0.5;
   if(pair == "USDCHF") return 0.5;
   if(pair == "NZDUSD") return 0.6;
   if(pair == "EURJPY") return 0.7;
   if(pair == "GBPJPY") return 1.0;
   return 1.0;  // fallback for unknown pair
}

//+------------------------------------------------------------------+
//| Operating Mode                                                    |
//+------------------------------------------------------------------+
enum ENUM_CHAOS_MODE {
   CHAOS_MODE_PAPER = 0,
   CHAOS_MODE_LIVE  = 1
};

//+------------------------------------------------------------------+
//| Helper: TF enum to string                                         |
//+------------------------------------------------------------------+
string TFToString(ENUM_TIMEFRAMES tf) {
   switch(tf) {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Helper: Find pair index in universe (-1 if not found)             |
//+------------------------------------------------------------------+
int GetPairIndex(const string pair) {
   for(int i = 0; i < CHAOS_NUM_PAIRS; i++) {
      if(CHAOS_PAIRS[i] == pair) return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Helper: Find TF index in active TFs (-1 if not found)            |
//+------------------------------------------------------------------+
int GetTFIndex(ENUM_TIMEFRAMES tf) {
   for(int i = 0; i < CHAOS_NUM_TIMEFRAMES; i++) {
      if(CHAOS_TIMEFRAMES[i] == tf) return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Helper: Check if TF is trade-enabled                              |
//+------------------------------------------------------------------+
bool IsTFTradeEnabled(ENUM_TIMEFRAMES tf) {
   int idx = GetTFIndex(tf);
   if(idx < 0) return false;
   return (CHAOS_TF_ROLES[idx] == TF_ROLE_TRADE_ENABLED);
}

#endif // CHAOS_CONFIG_MQH

//+------------------------------------------------------------------+
//|                                             CHAOS_Logger.mqh     |
//|                        CHAOS V1.0 - Decision & Trade Logging     |
//|                        CSV ledger with daily rotation             |
//+------------------------------------------------------------------+
#property copyright "CHAOS V1.0"
#property version   "1.00"
#property strict

#ifndef CHAOS_LOGGER_MQH
#define CHAOS_LOGGER_MQH

#include "CHAOS_Config.mqh"

//+------------------------------------------------------------------+
//| Log directory and file handles                                    |
//+------------------------------------------------------------------+
#define CHAOS_LOG_DIR          "CHAOS_Logs"

// File handles (-1 = not open)
int g_decisionLogHandle = -1;
int g_tradeLogHandle    = -1;

// Current log date — triggers rotation when day changes
string g_currentLogDate = "";

//+------------------------------------------------------------------+
//| CSV Headers                                                       |
//+------------------------------------------------------------------+
const string DECISION_HEADER =
   "timestamp,pair,tf,request_id,signal,confidence,agreement_score,"
   "regime_state,action,risk_approved,risk_reason,reason_codes,"
   "models_voted,models_agreed,lot_size,fill_price,spread_at_decision,"
   "latency_ms,equity,balance,open_positions,daily_pnl,comment";

const string TRADE_HEADER =
   "timestamp,pair,tf,trade_id,action,side,lot_size,fill_price,"
   "spread_pips,slippage_pips,pnl_gross,pnl_net,commission,"
   "regime_state,reason_codes,comment,"
   "commission_per_trade,net_pnl_after_commission,requested_price,"
   "filled_price,slippage_actual";

//+------------------------------------------------------------------+
//| InitLogger — create log directory and open today's files          |
//+------------------------------------------------------------------+
bool InitLogger() {
   // Create log directory if it doesn't exist
   if(!FolderCreate(CHAOS_LOG_DIR)) {
      int err = GetLastError();
      // Error 5019 = folder already exists, that's fine
      if(err != 5019 && err != 0) {
         Print("[CHAOS] ERROR: Cannot create log folder: ", CHAOS_LOG_DIR, " err=", err);
         return false;
      }
   }

   return OpenDailyFiles();
}

//+------------------------------------------------------------------+
//| OpenDailyFiles — open or rotate to new day's CSV files            |
//+------------------------------------------------------------------+
bool OpenDailyFiles() {
   string today = TimeToString(TimeCurrent(), TIME_DATE);
   // Normalise date format YYYY.MM.DD -> YYYY-MM-DD
   StringReplace(today, ".", "-");

   // Already on the right day?
   if(today == g_currentLogDate && g_decisionLogHandle >= 0 && g_tradeLogHandle >= 0)
      return true;

   // Close previous day's handles
   CloseLogs();

   g_currentLogDate = today;

   // Decision ledger
   string decisionFile = CHAOS_LOG_DIR + "/decision_ledger_" + today + ".csv";
   bool decisionExists = FileIsExist(decisionFile);
   g_decisionLogHandle = FileOpen(decisionFile, FILE_WRITE | FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
   if(g_decisionLogHandle < 0) {
      Print("[CHAOS] ERROR: Cannot open decision log: ", decisionFile, " err=", GetLastError());
      return false;
   }
   if(!decisionExists) {
      FileSeek(g_decisionLogHandle, 0, SEEK_END);
      FileWriteString(g_decisionLogHandle, DECISION_HEADER + "\n");
   } else {
      FileSeek(g_decisionLogHandle, 0, SEEK_END);
   }

   // Trade log
   string tradeFile = CHAOS_LOG_DIR + "/trades_" + today + ".csv";
   bool tradeExists = FileIsExist(tradeFile);
   g_tradeLogHandle = FileOpen(tradeFile, FILE_WRITE | FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
   if(g_tradeLogHandle < 0) {
      Print("[CHAOS] ERROR: Cannot open trade log: ", tradeFile, " err=", GetLastError());
      return false;
   }
   if(!tradeExists) {
      FileSeek(g_tradeLogHandle, 0, SEEK_END);
      FileWriteString(g_tradeLogHandle, TRADE_HEADER + "\n");
   } else {
      FileSeek(g_tradeLogHandle, 0, SEEK_END);
   }

   Print("[CHAOS] Logger initialised for ", today);
   return true;
}

//+------------------------------------------------------------------+
//| CloseLogs — flush and close all open file handles                 |
//+------------------------------------------------------------------+
void CloseLogs() {
   if(g_decisionLogHandle >= 0) {
      FileFlush(g_decisionLogHandle);
      FileClose(g_decisionLogHandle);
      g_decisionLogHandle = -1;
   }
   if(g_tradeLogHandle >= 0) {
      FileFlush(g_tradeLogHandle);
      FileClose(g_tradeLogHandle);
      g_tradeLogHandle = -1;
   }
}

//+------------------------------------------------------------------+
//| FlushLogs — force write buffers to disk                           |
//+------------------------------------------------------------------+
void FlushLogs() {
   if(g_decisionLogHandle >= 0) FileFlush(g_decisionLogHandle);
   if(g_tradeLogHandle >= 0)    FileFlush(g_tradeLogHandle);
}

//+------------------------------------------------------------------+
//| LogDecision — write one row to the decision ledger                |
//+------------------------------------------------------------------+
void LogDecision(
      const string pair,
      const string tf,
      const string request_id,
      const string signal,
      double confidence,
      double agreement_score,
      const string regime_state,
      const string action,
      bool risk_approved,
      const string risk_reason,
      const string reason_codes,
      int models_voted,
      int models_agreed,
      double lot_size,
      double fill_price,
      double spread_at_decision,
      int latency_ms,
      double equity,
      double balance,
      int open_positions,
      double daily_pnl,
      const string comment) {

   // Rotate files if day has changed
   OpenDailyFiles();

   if(g_decisionLogHandle < 0) {
      Print("[CHAOS] WARNING: Decision log not open, skipping write");
      return;
   }

   string ts = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS);

   string row = StringFormat(
      "%s,%s,%s,%s,%s,%.4f,%.4f,%s,%s,%s,%s,%s,%d,%d,%.2f,%.5f,%.1f,%d,%.2f,%.2f,%d,%.2f,%s",
      ts, pair, tf, request_id, signal,
      confidence, agreement_score, regime_state, action,
      (risk_approved ? "TRUE" : "FALSE"), risk_reason, reason_codes,
      models_voted, models_agreed, lot_size, fill_price, spread_at_decision,
      latency_ms, equity, balance, open_positions, daily_pnl, comment
   );

   FileWriteString(g_decisionLogHandle, row + "\n");
}

//+------------------------------------------------------------------+
//| LogTrade — write one row to the trade log                         |
//+------------------------------------------------------------------+
void LogTrade(
      const string pair,
      const string tf,
      long trade_id,
      const string action,
      const string side,
      double lot_size,
      double fill_price,
      double spread_pips,
      double slippage_pips,
      double pnl_gross,
      double pnl_net,
      double commission,
      const string regime_state,
      const string reason_codes,
      const string comment,
      double commission_per_trade = 0.0,
      double net_pnl_after_commission = 0.0,
      double requested_price = 0.0,
      double filled_price = 0.0,
      double slippage_actual = 0.0) {

   // Rotate files if day has changed
   OpenDailyFiles();

   if(g_tradeLogHandle < 0) {
      Print("[CHAOS] WARNING: Trade log not open, skipping write");
      return;
   }

   // Auto-compute IBKR fields if not provided
   if(commission_per_trade == 0.0 && commission != 0.0)
      commission_per_trade = commission;
   if(net_pnl_after_commission == 0.0)
      net_pnl_after_commission = pnl_gross - MathAbs(commission_per_trade);
   if(filled_price == 0.0)
      filled_price = fill_price;
   if(slippage_actual == 0.0 && requested_price > 0.0 && filled_price > 0.0)
      slippage_actual = MathAbs(filled_price - requested_price) /
                        SymbolInfoDouble(pair, SYMBOL_POINT);

   string ts = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS);

   string row = StringFormat(
      "%s,%s,%s,%d,%s,%s,%.2f,%.5f,%.1f,%.1f,%.2f,%.2f,%.2f,%s,%s,%s,"
      "%.4f,%.4f,%.5f,%.5f,%.1f",
      ts, pair, tf, trade_id, action, side,
      lot_size, fill_price, spread_pips, slippage_pips,
      pnl_gross, pnl_net, commission,
      regime_state, reason_codes, comment,
      commission_per_trade, net_pnl_after_commission,
      requested_price, filled_price, slippage_actual
   );

   FileWriteString(g_tradeLogHandle, row + "\n");
}

//+------------------------------------------------------------------+
//| LogError — print error to Experts log with CHAOS prefix           |
//+------------------------------------------------------------------+
void LogError(const string component, const string message) {
   Print("[CHAOS][ERROR][", component, "] ", message);
}

//+------------------------------------------------------------------+
//| LogWarning — print warning to Experts log with CHAOS prefix       |
//+------------------------------------------------------------------+
void LogWarning(const string component, const string message) {
   Print("[CHAOS][WARN][", component, "] ", message);
}

//+------------------------------------------------------------------+
//| LogInfo — print info to Experts log with CHAOS prefix             |
//+------------------------------------------------------------------+
void LogInfo(const string component, const string message) {
   Print("[CHAOS][INFO][", component, "] ", message);
}

#endif // CHAOS_LOGGER_MQH

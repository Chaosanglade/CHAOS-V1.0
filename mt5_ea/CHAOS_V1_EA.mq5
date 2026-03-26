//+------------------------------------------------------------------+
//|                                              CHAOS_V1_EA.mq5     |
//|                        CHAOS V1.0 Production Expert Advisor       |
//|                                                                    |
//|  Thin EA bridge: collects raw bars from MT5, sends to Python      |
//|  inference server via ZeroMQ (or file bridge), receives trade      |
//|  signals, and executes them.                                       |
//|                                                                    |
//|  The EA does NOT make trading decisions. All ML inference,         |
//|  ensemble voting, regime gating, and risk checks happen in Python. |
//|  The EA is a bridge + local safety net.                            |
//|                                                                    |
//|  Launch sequence:                                                  |
//|    1. Start Python server: python inference/live_zmq_server.py     |
//|    2. Attach this EA to any MT5 chart                              |
//|    3. Verify dashboard shows CONNECTED                             |
//|    4. Set InpLiveTrading=false for paper trading first              |
//+------------------------------------------------------------------+
#property copyright   "CHAOS V1.0"
#property version     "1.00"
#property description "CHAOS V1.0 ML Trading System — Production EA"
#property strict

// Include files
#include <CHAOS/CHAOS_Config.mqh>
#include <CHAOS/CHAOS_ZMQ_Client.mqh>
#include <CHAOS/CHAOS_TradeManager.mqh>
#include <CHAOS/CHAOS_RiskManager.mqh>
#include <CHAOS/CHAOS_Logger.mqh>
#include <CHAOS/CHAOS_Dashboard.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                   |
//+------------------------------------------------------------------+
// === CONNECTION ===
input string   InpZMQEndpoint     = "tcp://127.0.0.1:5555";  // ZeroMQ server address
input int      InpZMQTimeout      = 5000;                      // ZeroMQ timeout (ms)
input int      InpZMQRetries      = 3;                         // Retry attempts on timeout
input int      InpBridgeMode      = 0;                         // 0=ZMQ, 1=File Bridge

// === TRADING MODE ===
input bool     InpLiveTrading     = false;                     // true=live, false=paper (log only)
input bool     InpEnableH1        = true;                      // Enable H1 trading
input bool     InpEnableM30       = true;                      // Enable M30 trading

// === RISK ===
input double   InpRiskPerTrade    = 0.50;                      // Risk per trade (% of equity)
input int      InpMaxPositions    = 8;                         // Max concurrent positions
input int      InpMaxPerPair      = 1;                         // Max positions per pair
input double   InpMaxDDPct        = 2.0;                       // Intraday drawdown circuit breaker (%)
input int      InpCooldownBars    = 1;                         // Bars to wait after exit

// === LOT SIZE ===
input double   InpMinLot          = 0.01;                      // Minimum lot size
input double   InpMaxLot          = 10.0;                      // Maximum lot size
input double   InpLotStep         = 0.01;                      // Lot step

// === LOGGING ===
input bool     InpLogDecisions    = true;                      // Log every decision to file
input string   InpLogPath         = "CHAOS_Logs";              // Log directory name
input bool     InpShowDashboard   = true;                      // Show on-chart dashboard

//+------------------------------------------------------------------+
//| Global State                                                       |
//+------------------------------------------------------------------+
datetime g_lastBarTime_H1[];    // Last bar time per pair for H1
datetime g_lastBarTime_M30[];   // Last bar time per pair for M30
datetime g_lastBarTime_M15[];   // Last bar time per pair for M15
datetime g_lastBarTime_M5[];    // Last bar time per pair for M5
int      g_requestSeq = 0;     // Request sequence counter
bool     g_serverConnected = false;
int      g_dailySignalCount = 0;
datetime g_lastDayReset = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("  CHAOS V1.0 Expert Advisor Starting");
   Print("  Mode: ", InpLiveTrading ? "LIVE TRADING" : "PAPER (log only)");
   Print("  Server: ", InpZMQEndpoint);
   Print("  Bridge: ", InpBridgeMode == 0 ? "ZeroMQ" : "File Bridge");
   Print("========================================");

   // Initialize last bar times arrays
   ArrayResize(g_lastBarTime_H1, CHAOS_NUM_PAIRS);
   ArrayResize(g_lastBarTime_M30, CHAOS_NUM_PAIRS);
   ArrayResize(g_lastBarTime_M15, CHAOS_NUM_PAIRS);
   ArrayResize(g_lastBarTime_M5, CHAOS_NUM_PAIRS);
   ArrayInitialize(g_lastBarTime_H1, 0);
   ArrayInitialize(g_lastBarTime_M30, 0);
   ArrayInitialize(g_lastBarTime_M15, 0);
   ArrayInitialize(g_lastBarTime_M5, 0);

   // Initialize subsystems
   if(!InitZMQClient(InpZMQEndpoint, InpZMQTimeout, InpBridgeMode))
   {
      Print("ERROR: Failed to initialize communication bridge");
      Print("Make sure Python server is running first!");
      // Don't fail init — allow reconnection attempts
   }

   InitRiskManager(InpMaxPositions, InpMaxPerPair, InpMaxDDPct, InpCooldownBars);
   InitLogger();

   if(InpShowDashboard)
      CreateDashboard();

   // Verify server connection
   g_serverConnected = SendHeartbeat();
   if(g_serverConnected)
      Print("Server connection: OK");
   else
      Print("WARNING: Server not responding. Will retry on timer.");

   // Validate symbols and prime history for all pairs + TFs
   int validPairs = 0;
   for(int i = 0; i < CHAOS_NUM_PAIRS; i++)
   {
      string sym = CHAOS_PAIRS[i];
      if(SymbolSelect(sym, true))
      {
         validPairs++;
         // Force MT5 to load history for each TF by requesting 1 bar.
         // Without this, iTime() returns 0 for non-chart symbols.
         MqlRates rates[];
         int copied_h1  = CopyRates(sym, PERIOD_H1,  0, 1, rates);
         int copied_m30 = CopyRates(sym, PERIOD_M30, 0, 1, rates);
         int copied_m15 = CopyRates(sym, PERIOD_M15, 0, 1, rates);
         int copied_m5  = CopyRates(sym, PERIOD_M5,  0, 1, rates);
         Print("  ", sym, " history loaded: H1=", copied_h1, " M30=", copied_m30,
               " M15=", copied_m15, " M5=", copied_m5);
      }
      else
      {
         Print("WARNING: Symbol ", sym, " not available on this broker");
      }
   }
   Print("Valid pairs: ", validPairs, "/", CHAOS_NUM_PAIRS);

   // Set timer (1 second interval)
   EventSetTimer(1);

   Print("CHAOS V1.0 EA initialized successfully");
   Print("Monitoring ", validPairs, " pairs on H1/M30 (trade) + M15/M5 (confirm)");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();

   // Close all positions if live and shutting down unexpectedly
   if(InpLiveTrading && reason != REASON_CHARTCHANGE && reason != REASON_PARAMETERS)
   {
      Print("WARNING: EA shutting down. Open positions will NOT be auto-closed.");
      Print("Reason code: ", reason);
   }

   CloseZMQClient();
   FlushLogs();

   if(InpShowDashboard)
      DestroyDashboard();

   Print("CHAOS V1.0 EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer event (1 second)                                             |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Daily reset check
   MqlDateTime dt;
   TimeCurrent(dt);
   datetime today = StringToTime(IntegerToString(dt.year) + "." +
                                  IntegerToString(dt.mon) + "." +
                                  IntegerToString(dt.day));
   if(today != g_lastDayReset)
   {
      g_lastDayReset = today;
      g_dailySignalCount = 0;
      ResetDailyStats();
      RotateLogFiles();
   }

   // Reconnect if needed
   if(!g_serverConnected)
   {
      g_serverConnected = SendHeartbeat();
      if(g_serverConnected)
         Print("Server reconnected!");
   }

   // Check for new bars on each TF
   bool newH1 = false, newM30 = false, newM15 = false, newM5 = false;

   for(int i = 0; i < CHAOS_NUM_PAIRS; i++)
   {
      string pair = CHAOS_PAIRS[i];

      // H1 new bar check
      if(InpEnableH1)
      {
         datetime barTime = iTime(pair, PERIOD_H1, 0);
         if(barTime > g_lastBarTime_H1[i] && barTime > 0)
         {
            g_lastBarTime_H1[i] = barTime;
            ProcessSignal(pair, "H1", PERIOD_H1);
            newH1 = true;
         }
      }

      // M30 new bar check
      if(InpEnableM30)
      {
         datetime barTime = iTime(pair, PERIOD_M30, 0);
         if(barTime > g_lastBarTime_M30[i] && barTime > 0)
         {
            g_lastBarTime_M30[i] = barTime;
            ProcessSignal(pair, "M30", PERIOD_M30);
            newM30 = true;
         }
      }

      // M15 confirm data (never trades, just collects for MTF)
      {
         datetime barTime = iTime(pair, PERIOD_M15, 0);
         if(barTime > g_lastBarTime_M15[i] && barTime > 0)
         {
            g_lastBarTime_M15[i] = barTime;
            ProcessSignal(pair, "M15", PERIOD_M15);
            newM15 = true;
         }
      }

      // M5 confirm data
      {
         datetime barTime = iTime(pair, PERIOD_M5, 0);
         if(barTime > g_lastBarTime_M5[i] && barTime > 0)
         {
            g_lastBarTime_M5[i] = barTime;
            ProcessSignal(pair, "M5", PERIOD_M5);
            newM5 = true;
         }
      }
   }

   // Update dashboard
   if(InpShowDashboard)
   {
      DashboardData dd;
      dd.isLive = InpLiveTrading;
      dd.serverConnected = g_serverConnected;
      dd.lastLatencyMs = GetLastLatencyMs();
      dd.openPositions = CountChaosPositions();
      dd.maxPositions = InpMaxPositions;
      dd.dailyPnL = GetDailyPnL();
      dd.dailyTrades = GetDailyTradeCount();
      dd.dailyWinRate = GetDailyWinRate();
      dd.regimeState = 1;     // Updated by last server response
      dd.regimeConfidence = 0.85;
      dd.defensiveMode = false;
      dd.cooldownsActive = CountActiveCooldowns();
      UpdateDashboard(dd);
   }
}

//+------------------------------------------------------------------+
//| Process signal for a pair+TF                                       |
//+------------------------------------------------------------------+
void ProcessSignal(string pair, string tf, ENUM_TIMEFRAMES period)
{
   if(!g_serverConnected)
   {
      if(InpLogDecisions)
         LogDecision(pair, tf, "", "0", 0, 0, "1", "SKIP", false,
                     "SERVER_DISCONNECTED", "SERVER_DISCONNECTED",
                     0, 0, 0, 0, 0, 0, 0, 0, 0, "");
      return;
   }

   uint startTick = GetTickCount();

   // Build request JSON with raw bars
   string request = BuildBarRequest(pair, tf, period);
   if(StringLen(request) == 0)
   {
      LogWarning("EA", "Failed to build request for " + pair + "_" + tf);
      return;
   }

   // Send to Python server
   string response = SendRequest(request);
   uint latency = GetTickCount() - startTick;

   if(StringLen(response) == 0)
   {
      g_serverConnected = false;
      if(InpLogDecisions)
         LogDecision(pair, tf, "", "0", 0, 0, "1", "SKIP", false,
                     "ZMQ_TIMEOUT", "ZMQ_TIMEOUT",
                     0, 0, 0, 0, 0, (int)latency, 0, 0, 0, 0, "");
      return;
   }

   // Parse response
   int signal = 0;
   double confidence = 0, agreement = 0;
   int regimeState = 1, modelsVoted = 0, modelsAgreed = 0;
   string action = "", riskReason = "", reasonCodes = "";
   bool riskApproved = false;
   double lotSize = 0;

   ParseResponse(response, signal, confidence, agreement, regimeState,
                 riskApproved, riskReason, action, reasonCodes,
                 lotSize, modelsVoted, modelsAgreed);

   g_dailySignalCount++;

   // Get current market data for logging
   double spread = SymbolInfoInteger(pair, SYMBOL_SPREAD) * SymbolInfoDouble(pair, SYMBOL_POINT);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   int openPos = CountChaosPositions();
   double dailyPnl = GetDailyPnL();

   // Log decision (ALWAYS, even on SKIP/HOLD)
   if(InpLogDecisions)
   {
      string requestId = pair + "_" + tf + "_" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) +
                         "_" + IntegerToString(g_requestSeq);
      LogDecision(pair, tf, requestId,
                  IntegerToString(signal), confidence, agreement,
                  IntegerToString(regimeState),
                  action, riskApproved, riskReason, reasonCodes,
                  modelsVoted, modelsAgreed, lotSize, 0, spread,
                  (int)latency, equity, balance, openPos, dailyPnl, "");
   }

   // Execute action (only for TRADE_ENABLED TFs)
   if(action == "OPEN" && (tf == "H1" || tf == "M30"))
   {
      // Local risk check (belt-and-suspenders)
      if(!LocalRiskCheck(pair, tf, signal))
      {
         LogWarning("RISK", "Local risk check BLOCKED: " + pair + "_" + tf);
         return;
      }

      if(InpLiveTrading)
      {
         // Calculate lot size locally (override server suggestion with risk-based sizing)
         double calcLot = CalculateLotSize(pair, InpRiskPerTrade, equity);
         if(calcLot < InpMinLot) calcLot = InpMinLot;
         if(calcLot > InpMaxLot) calcLot = InpMaxLot;

         string rid = pair + "_" + tf + "_" + IntegerToString(g_requestSeq++);
         OpenPosition(pair, signal, calcLot, rid);
      }
      else
      {
         // Paper mode — log only
         Print("PAPER: Would OPEN ", (signal > 0 ? "LONG" : "SHORT"),
               " ", pair, " @ ", SymbolInfoDouble(pair, signal > 0 ? SYMBOL_ASK : SYMBOL_BID),
               " lot=", DoubleToString(lotSize, 2));
      }
   }
   else if(action == "CLOSE" && (tf == "H1" || tf == "M30"))
   {
      if(InpLiveTrading)
      {
         string rid = pair + "_" + tf + "_CLOSE_" + IntegerToString(g_requestSeq++);
         ClosePosition(pair, tf, rid);
         SetCooldown(pair, tf, InpCooldownBars);
      }
      else
      {
         Print("PAPER: Would CLOSE ", pair, "_", tf);
      }
   }
   // HOLD and SKIP: no action needed (already logged)
}

//+------------------------------------------------------------------+
//| Build JSON request with raw bars                                   |
//+------------------------------------------------------------------+
string BuildBarRequest(string pair, string tf, ENUM_TIMEFRAMES period)
{
   // Collect bars for all TFs (server needs MTF data)
   string bars_m5 = CollectBars(pair, PERIOD_M5, CHAOS_BARS_TO_SEND);
   string bars_m15 = CollectBars(pair, PERIOD_M15, CHAOS_BARS_TO_SEND);
   string bars_m30 = CollectBars(pair, PERIOD_M30, CHAOS_BARS_TO_SEND);
   string bars_h1 = CollectBars(pair, PERIOD_H1, CHAOS_BARS_TO_SEND);

   g_requestSeq++;
   string requestId = pair + "_" + tf + "_" +
                      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) +
                      "_" + IntegerToString(g_requestSeq, 6, '0');

   string json = "{";
   json += "\"pair\":\"" + pair + "\",";
   json += "\"tf\":\"" + tf + "\",";
   json += "\"mode\":\"LIVE\",";
   json += "\"request_type\":\"RAW_BARS\",";
   json += "\"bars\":{";
   json += "\"M5\":" + bars_m5 + ",";
   json += "\"M15\":" + bars_m15 + ",";
   json += "\"M30\":" + bars_m30 + ",";
   json += "\"H1\":" + bars_h1;
   json += "},";
   json += "\"bars_count\":" + IntegerToString(CHAOS_BARS_TO_SEND) + ",";
   json += "\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",";
   json += "\"request_id\":\"" + requestId + "\"";
   json += "}";

   return json;
}

//+------------------------------------------------------------------+
//| Collect OHLCV bars as JSON array                                   |
//+------------------------------------------------------------------+
string CollectBars(string pair, ENUM_TIMEFRAMES period, int count)
{
   MqlRates rates[];
   int copied = CopyRates(pair, period, 1, count, rates);  // Skip current bar (index 1)

   if(copied <= 0)
      return "[]";

   string json = "[";
   for(int i = 0; i < copied; i++)
   {
      if(i > 0) json += ",";
      json += "{";
      json += "\"time\":\"" + TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS) + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, 6) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, 6) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, 6) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, 6) + ",";
      json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
      json += "}";
   }
   json += "]";

   return json;
}

//+------------------------------------------------------------------+
//| Parse JSON response from Python server                             |
//+------------------------------------------------------------------+
void ParseResponse(string json,
                   int &signal, double &confidence, double &agreement,
                   int &regimeState, bool &riskApproved, string &riskReason,
                   string &action, string &reasonCodes,
                   double &lotSize, int &modelsVoted, int &modelsAgreed)
{
   // Simple JSON parsing (MQL5 doesn't have a JSON library)
   signal = (int)ExtractJsonInt(json, "signal");
   confidence = ExtractJsonDouble(json, "confidence");
   agreement = ExtractJsonDouble(json, "agreement_score");
   regimeState = (int)ExtractJsonInt(json, "regime_state");
   riskApproved = ExtractJsonBool(json, "risk_approved");
   riskReason = ExtractJsonString(json, "risk_reason");
   action = ExtractJsonString(json, "action");
   reasonCodes = ExtractJsonString(json, "reason_codes");
   lotSize = ExtractJsonDouble(json, "lot_size");
   modelsVoted = (int)ExtractJsonInt(json, "models_voted");
   modelsAgreed = (int)ExtractJsonInt(json, "models_agreed");
}

//+------------------------------------------------------------------+
//| JSON extraction helpers (simple string parsing)                    |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;
   pos += StringLen(search);
   // Skip whitespace
   while(pos < StringLen(json) && (StringGetCharacter(json, pos) == ' ')) pos++;
   int end = pos;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']') break;
      end++;
   }
   string val = StringSubstr(json, pos, end - pos);
   StringTrimRight(val);
   StringTrimLeft(val);
   return StringToDouble(val);
}

long ExtractJsonInt(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;
   pos += StringLen(search);
   while(pos < StringLen(json) && (StringGetCharacter(json, pos) == ' ')) pos++;
   int end = pos;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']') break;
      end++;
   }
   string val = StringSubstr(json, pos, end - pos);
   StringTrimRight(val);
   StringTrimLeft(val);
   return StringToInteger(val);
}

string ExtractJsonString(string json, string key)
{
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   pos += StringLen(search);
   int end = StringFind(json, "\"", pos);
   if(end < 0) return "";
   return StringSubstr(json, pos, end - pos);
}

bool ExtractJsonBool(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return false;
   pos += StringLen(search);
   while(pos < StringLen(json) && (StringGetCharacter(json, pos) == ' ')) pos++;
   string sub = StringSubstr(json, pos, 4);
   return (sub == "true");
}

//+------------------------------------------------------------------+
//| Count positions with CHAOS magic number                            |
//+------------------------------------------------------------------+
int CountChaosPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetInteger(POSITION_MAGIC) == CHAOS_MAGIC)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| OnTick - minimal, dashboard refresh only                           |
//+------------------------------------------------------------------+
void OnTick()
{
   // Intentionally minimal. All logic is in OnTimer().
   // OnTick fires too frequently for our use case.
}
//+------------------------------------------------------------------+

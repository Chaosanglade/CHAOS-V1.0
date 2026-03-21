//+------------------------------------------------------------------+
//| CHAOS_TradeManager.mqh                                             |
//| CHAOS V1.0 — Trade Execution & Position Management                 |
//+------------------------------------------------------------------+
#ifndef CHAOS_TRADE_MANAGER_MQH
#define CHAOS_TRADE_MANAGER_MQH

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>
#include <Trade\DealInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| Constants                                                          |
//+------------------------------------------------------------------+
#define CHAOS_MAGIC     20260306    // Magic number for all CHAOS trades
#define CHAOS_DEVIATION 20         // Max slippage in points (2 pips)

//+------------------------------------------------------------------+
//| Position info structure                                            |
//+------------------------------------------------------------------+
struct ChaosPositionInfo
{
   ulong    ticket;
   int      type;        // POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1
   double   volume;
   double   profit;
   double   openPrice;
   datetime openTime;
   string   comment;
   bool     valid;
};

//+------------------------------------------------------------------+
//| CHAOS_TradeManager class                                           |
//+------------------------------------------------------------------+
class CHAOS_TradeManager
{
private:
   CTrade            m_trade;
   CPositionInfo     m_position;
   CSymbolInfo       m_symbolInfo;
   CAccountInfo      m_accountInfo;
   CDealInfo         m_dealInfo;

   ulong             m_magic;
   int               m_deviation;
   string            m_lastError;

   // Daily tracking
   datetime          m_dailyResetDate;
   double            m_dailyClosedPnL;
   int               m_dailyTradeCount;
   int               m_dailyWins;
   int               m_dailyLosses;

   // Internal helpers
   void              ResetDailyIfNeeded();
   void              LogTrade(string action, string pair, double lots,
                              double price, string requestId, int errorCode);
   double            GetATR(string pair, ENUM_TIMEFRAMES tf, int period, int shift);
   bool              ValidateSymbol(string pair);

public:
                     CHAOS_TradeManager();
                    ~CHAOS_TradeManager();

   bool              Init();

   // Position operations
   bool              OpenPosition(string pair, int signal, double lotSize, string requestId);
   bool              ClosePosition(string pair, string tf, string requestId);
   bool              CloseAllPositions(string reason);
   bool              HasPosition(string pair);
   ChaosPositionInfo GetPositionInfo(string pair);

   // Sizing
   double            CalculateLotSize(string pair, double riskPct, double equity,
                                      double stopPips = 0, ENUM_TIMEFRAMES atrTf = PERIOD_H1,
                                      int atrPeriod = 14);

   // Counting & stats
   int               CountOpenPositions();
   double            GetDailyPnL();
   int               GetDailyTradeCount();
   double            GetDailyWinRate();

   // Accessors
   string            GetLastError()  { return m_lastError; }
   ulong             GetMagic()      { return m_magic; }
};

//+------------------------------------------------------------------+
//| Constructor                                                        |
//+------------------------------------------------------------------+
CHAOS_TradeManager::CHAOS_TradeManager()
{
   m_magic          = CHAOS_MAGIC;
   m_deviation      = CHAOS_DEVIATION;
   m_lastError      = "";
   m_dailyResetDate = 0;
   m_dailyClosedPnL = 0;
   m_dailyTradeCount = 0;
   m_dailyWins      = 0;
   m_dailyLosses    = 0;
}

//+------------------------------------------------------------------+
//| Destructor                                                         |
//+------------------------------------------------------------------+
CHAOS_TradeManager::~CHAOS_TradeManager()
{
}

//+------------------------------------------------------------------+
//| Initialize trade manager                                           |
//+------------------------------------------------------------------+
bool CHAOS_TradeManager::Init()
{
   m_trade.SetExpertMagicNumber(m_magic);
   m_trade.SetDeviationInPoints(m_deviation);
   m_trade.SetTypeFilling(ORDER_FILLING_IOC); // Most brokers support IOC
   m_trade.SetAsyncMode(false);               // Synchronous execution

   // Verify trading is allowed
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      m_lastError = "Trading not allowed in terminal settings";
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return false;
   }

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      m_lastError = "Trading not allowed for this EA";
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return false;
   }

   PrintFormat("[CHAOS_TRADE] TradeManager initialized. Magic=%d, Deviation=%d pts",
               m_magic, m_deviation);
   return true;
}

//+------------------------------------------------------------------+
//| Open a position                                                    |
//| signal: 1=BUY, -1=SELL                                            |
//+------------------------------------------------------------------+
bool CHAOS_TradeManager::OpenPosition(string pair, int signal, double lotSize, string requestId)
{
   ResetDailyIfNeeded();

   if(signal != 1 && signal != -1)
   {
      m_lastError = StringFormat("Invalid signal %d (must be 1 or -1)", signal);
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return false;
   }

   if(!ValidateSymbol(pair))
      return false;

   m_symbolInfo.Name(pair);
   m_symbolInfo.RefreshRates();

   // Check if we already have a position for this pair
   if(HasPosition(pair))
   {
      m_lastError = StringFormat("Already have open position for %s", pair);
      PrintFormat("[CHAOS_TRADE] WARNING: %s", m_lastError);
      return false;
   }

   // Normalize lot size
   double minLot  = SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(pair, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(pair, SYMBOL_VOLUME_STEP);

   if(lotStep > 0)
      lotSize = MathFloor(lotSize / lotStep) * lotStep;

   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   lotSize = NormalizeDouble(lotSize, 2);

   if(lotSize < minLot)
   {
      m_lastError = StringFormat("Lot size %.4f below minimum %.4f for %s",
                                 lotSize, minLot, pair);
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return false;
   }

   string comment = "CHAOS_" + requestId;

   ENUM_ORDER_TYPE orderType;
   double price;

   if(signal == 1) // BUY
   {
      orderType = ORDER_TYPE_BUY;
      price = m_symbolInfo.Ask();
   }
   else // SELL
   {
      orderType = ORDER_TYPE_SELL;
      price = m_symbolInfo.Bid();
   }

   PrintFormat("[CHAOS_TRADE] Opening %s %s %.4f lots @ %.5f [%s]",
               (signal == 1 ? "BUY" : "SELL"), pair, lotSize, price, requestId);

   bool result = m_trade.PositionOpen(pair, orderType, lotSize, price, 0, 0, comment);

   uint retcode = m_trade.ResultRetcode();

   if(result && (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_PLACED))
   {
      double fillPrice = m_trade.ResultPrice();
      ulong ticket = m_trade.ResultOrder();
      m_dailyTradeCount++;

      PrintFormat("[CHAOS_TRADE] SUCCESS: %s %s %.4f lots filled @ %.5f, ticket=%d [%s]",
                  (signal == 1 ? "BUY" : "SELL"), pair, lotSize, fillPrice, ticket, requestId);
      LogTrade("OPEN", pair, lotSize, fillPrice, requestId, (int)retcode);
      return true;
   }
   else
   {
      m_lastError = StringFormat("OrderSend failed: retcode=%d, desc=%s",
                                 retcode, m_trade.ResultRetcodeDescription());
      PrintFormat("[CHAOS_TRADE] ERROR: %s [%s]", m_lastError, requestId);
      LogTrade("OPEN_FAIL", pair, lotSize, price, requestId, (int)retcode);
      return false;
   }
}

//+------------------------------------------------------------------+
//| Close position for a pair                                          |
//+------------------------------------------------------------------+
bool CHAOS_TradeManager::ClosePosition(string pair, string tf, string requestId)
{
   ResetDailyIfNeeded();

   int total = PositionsTotal();
   bool found = false;

   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != m_magic)
         continue;

      string posPair = PositionGetString(POSITION_SYMBOL);
      if(posPair != pair)
         continue;

      found = true;

      double volume = PositionGetDouble(POSITION_VOLUME);
      double profit = PositionGetDouble(POSITION_PROFIT);
      double swap   = PositionGetDouble(POSITION_SWAP);
      double totalPnL = profit + swap;

      PrintFormat("[CHAOS_TRADE] Closing %s position ticket=%d, vol=%.4f, PnL=%.2f [%s/%s]",
                  pair, ticket, volume, totalPnL, tf, requestId);

      bool result = m_trade.PositionClose(ticket, m_deviation);
      uint retcode = m_trade.ResultRetcode();

      if(result && (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_PLACED))
      {
         double fillPrice = m_trade.ResultPrice();
         m_dailyClosedPnL += totalPnL;

         if(totalPnL >= 0)
            m_dailyWins++;
         else
            m_dailyLosses++;

         PrintFormat("[CHAOS_TRADE] SUCCESS: Closed %s @ %.5f, PnL=%.2f [%s/%s]",
                     pair, fillPrice, totalPnL, tf, requestId);
         LogTrade("CLOSE", pair, volume, fillPrice, requestId, (int)retcode);
      }
      else
      {
         m_lastError = StringFormat("Close failed: retcode=%d, desc=%s",
                                    retcode, m_trade.ResultRetcodeDescription());
         PrintFormat("[CHAOS_TRADE] ERROR: %s [%s/%s]", m_lastError, tf, requestId);
         LogTrade("CLOSE_FAIL", pair, volume, 0, requestId, (int)retcode);
         return false;
      }
   }

   if(!found)
   {
      m_lastError = StringFormat("No open position found for %s with magic %d", pair, m_magic);
      PrintFormat("[CHAOS_TRADE] WARNING: %s", m_lastError);
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Emergency close all CHAOS positions                                |
//+------------------------------------------------------------------+
bool CHAOS_TradeManager::CloseAllPositions(string reason)
{
   PrintFormat("[CHAOS_TRADE] EMERGENCY CLOSE ALL: %s", reason);

   int total = PositionsTotal();
   int closed = 0;
   int failed = 0;

   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != m_magic)
         continue;

      string posPair = PositionGetString(POSITION_SYMBOL);
      double volume  = PositionGetDouble(POSITION_VOLUME);
      double profit  = PositionGetDouble(POSITION_PROFIT);
      double swap    = PositionGetDouble(POSITION_SWAP);
      double totalPnL = profit + swap;

      bool result = m_trade.PositionClose(ticket, m_deviation * 2); // Double deviation for emergency
      uint retcode = m_trade.ResultRetcode();

      if(result && (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_PLACED))
      {
         m_dailyClosedPnL += totalPnL;
         if(totalPnL >= 0) m_dailyWins++; else m_dailyLosses++;
         closed++;
         PrintFormat("[CHAOS_TRADE] Closed %s ticket=%d, PnL=%.2f [%s]",
                     posPair, ticket, totalPnL, reason);
      }
      else
      {
         failed++;
         PrintFormat("[CHAOS_TRADE] FAILED to close %s ticket=%d, retcode=%d [%s]",
                     posPair, ticket, retcode, reason);
      }
   }

   PrintFormat("[CHAOS_TRADE] Emergency close complete: %d closed, %d failed [%s]",
               closed, failed, reason);

   if(failed > 0)
   {
      m_lastError = StringFormat("%d positions failed to close", failed);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Check if we have an open position for a pair                       |
//+------------------------------------------------------------------+
bool CHAOS_TradeManager::HasPosition(string pair)
{
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != m_magic)
         continue;

      if(PositionGetString(POSITION_SYMBOL) == pair)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Get position info for a pair                                       |
//+------------------------------------------------------------------+
ChaosPositionInfo CHAOS_TradeManager::GetPositionInfo(string pair)
{
   ChaosPositionInfo info;
   info.valid = false;
   info.ticket = 0;
   info.type = -1;
   info.volume = 0;
   info.profit = 0;
   info.openPrice = 0;
   info.openTime = 0;
   info.comment = "";

   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != m_magic)
         continue;

      if(PositionGetString(POSITION_SYMBOL) != pair)
         continue;

      info.valid     = true;
      info.ticket    = ticket;
      info.type      = (int)PositionGetInteger(POSITION_TYPE);
      info.volume    = PositionGetDouble(POSITION_VOLUME);
      info.profit    = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      info.openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      info.openTime  = (datetime)PositionGetInteger(POSITION_TIME);
      info.comment   = PositionGetString(POSITION_COMMENT);
      break;
   }

   return info;
}

//+------------------------------------------------------------------+
//| Calculate risk-based lot size                                      |
//| If stopPips > 0, uses fixed pip stop. Otherwise uses ATR.         |
//+------------------------------------------------------------------+
double CHAOS_TradeManager::CalculateLotSize(string pair, double riskPct, double equity,
                                            double stopPips, ENUM_TIMEFRAMES atrTf,
                                            int atrPeriod)
{
   if(!ValidateSymbol(pair))
      return 0;

   if(equity <= 0)
      equity = m_accountInfo.Equity();

   double riskAmount = equity * riskPct / 100.0;

   // Determine stop distance in pips
   double pipStop = stopPips;

   if(pipStop <= 0)
   {
      // Use ATR-based stop (1.5x ATR)
      double atrValue = GetATR(pair, atrTf, atrPeriod, 1);
      if(atrValue <= 0)
      {
         m_lastError = "ATR calculation failed, using default 50 pip stop";
         PrintFormat("[CHAOS_TRADE] WARNING: %s", m_lastError);
         pipStop = 50.0;
      }
      else
      {
         double point = SymbolInfoDouble(pair, SYMBOL_POINT);
         int digits = (int)SymbolInfoInteger(pair, SYMBOL_DIGITS);
         double pipSize = (digits == 3 || digits == 5) ? point * 10 : point;
         pipStop = (atrValue * 1.5) / pipSize;
      }
   }

   if(pipStop <= 0)
      pipStop = 50.0; // Safety fallback

   // Calculate pip value
   double point = SymbolInfoDouble(pair, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(pair, SYMBOL_DIGITS);
   double pipSize = (digits == 3 || digits == 5) ? point * 10 : point;
   double tickValue = SymbolInfoDouble(pair, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(pair, SYMBOL_TRADE_TICK_SIZE);

   if(tickSize <= 0 || tickValue <= 0)
   {
      m_lastError = "Cannot get tick value/size for " + pair;
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return 0;
   }

   double pipValue = tickValue * (pipSize / tickSize);

   if(pipValue <= 0)
   {
      m_lastError = "Pip value calculation returned <= 0 for " + pair;
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return 0;
   }

   // Lot size = risk$ / (stopPips * pipValue)
   double lots = riskAmount / (pipStop * pipValue);

   // Normalize to lot step
   double minLot  = SymbolInfoDouble(pair, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(pair, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(pair, SYMBOL_VOLUME_STEP);

   if(lotStep > 0)
      lots = MathFloor(lots / lotStep) * lotStep;

   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = NormalizeDouble(lots, 2);

   PrintFormat("[CHAOS_TRADE] LotCalc %s: risk=%.1f%% equity=%.2f riskAmt=%.2f stop=%.1f pips pipVal=%.4f -> %.4f lots",
               pair, riskPct, equity, riskAmount, pipStop, pipValue, lots);

   return lots;
}

//+------------------------------------------------------------------+
//| Count open positions with our magic number                         |
//+------------------------------------------------------------------+
int CHAOS_TradeManager::CountOpenPositions()
{
   int count = 0;
   int total = PositionsTotal();

   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionGetInteger(POSITION_MAGIC) == m_magic)
         count++;
   }

   return count;
}

//+------------------------------------------------------------------+
//| Get total daily PnL (closed + floating)                            |
//+------------------------------------------------------------------+
double CHAOS_TradeManager::GetDailyPnL()
{
   ResetDailyIfNeeded();

   // Start with closed PnL
   double totalPnL = m_dailyClosedPnL;

   // Add floating PnL from open positions
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(PositionGetInteger(POSITION_MAGIC) != m_magic)
         continue;

      totalPnL += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
   }

   return totalPnL;
}

//+------------------------------------------------------------------+
//| Get number of trades today                                         |
//+------------------------------------------------------------------+
int CHAOS_TradeManager::GetDailyTradeCount()
{
   ResetDailyIfNeeded();
   return m_dailyTradeCount;
}

//+------------------------------------------------------------------+
//| Get daily win rate                                                 |
//+------------------------------------------------------------------+
double CHAOS_TradeManager::GetDailyWinRate()
{
   ResetDailyIfNeeded();

   int totalTrades = m_dailyWins + m_dailyLosses;
   if(totalTrades == 0)
      return 0;

   return (double)m_dailyWins / (double)totalTrades * 100.0;
}

//+------------------------------------------------------------------+
//| Reset daily counters if new day                                    |
//+------------------------------------------------------------------+
void CHAOS_TradeManager::ResetDailyIfNeeded()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

   if(today != m_dailyResetDate)
   {
      if(m_dailyResetDate > 0)
      {
         PrintFormat("[CHAOS_TRADE] Daily reset. Previous day PnL=%.2f, Trades=%d, WR=%.1f%%",
                     m_dailyClosedPnL, m_dailyTradeCount,
                     (m_dailyWins + m_dailyLosses) > 0
                        ? (double)m_dailyWins / (m_dailyWins + m_dailyLosses) * 100.0 : 0);
      }
      m_dailyResetDate  = today;
      m_dailyClosedPnL  = 0;
      m_dailyTradeCount = 0;
      m_dailyWins       = 0;
      m_dailyLosses     = 0;
   }
}

//+------------------------------------------------------------------+
//| Log trade action                                                   |
//+------------------------------------------------------------------+
void CHAOS_TradeManager::LogTrade(string action, string pair, double lots,
                                  double price, string requestId, int errorCode)
{
   string logMsg = StringFormat("%s | %s | %s | %.4f lots | %.5f | err=%d | req=%s",
                                TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
                                action, pair, lots, price, errorCode, requestId);

   // Write to file log
   string logFile = "CHAOS_Trades\\trade_log.csv";
   int handle = FileOpen(logFile, FILE_WRITE | FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(handle != INVALID_HANDLE)
   {
      FileSeek(handle, 0, SEEK_END);
      FileWrite(handle,
                TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
                action, pair,
                DoubleToString(lots, 4),
                DoubleToString(price, 5),
                IntegerToString(errorCode),
                requestId);
      FileClose(handle);
   }
}

//+------------------------------------------------------------------+
//| Get ATR value                                                      |
//+------------------------------------------------------------------+
double CHAOS_TradeManager::GetATR(string pair, ENUM_TIMEFRAMES tf, int period, int shift)
{
   int atrHandle = iATR(pair, tf, period);
   if(atrHandle == INVALID_HANDLE)
      return 0;

   double atrBuffer[];
   ArraySetAsSeries(atrBuffer, true);

   if(CopyBuffer(atrHandle, 0, shift, 1, atrBuffer) <= 0)
   {
      IndicatorRelease(atrHandle);
      return 0;
   }

   double value = atrBuffer[0];
   IndicatorRelease(atrHandle);
   return value;
}

//+------------------------------------------------------------------+
//| Validate that a symbol exists and is tradeable                     |
//+------------------------------------------------------------------+
bool CHAOS_TradeManager::ValidateSymbol(string pair)
{
   if(!SymbolInfoInteger(pair, SYMBOL_EXIST))
   {
      m_lastError = StringFormat("Symbol %s does not exist", pair);
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return false;
   }

   if(!SymbolInfoInteger(pair, SYMBOL_SELECT))
   {
      // Try to add it to Market Watch
      SymbolSelect(pair, true);
   }

   long tradeMode = SymbolInfoInteger(pair, SYMBOL_TRADE_MODE);
   if(tradeMode == SYMBOL_TRADE_MODE_DISABLED)
   {
      m_lastError = StringFormat("Trading disabled for %s", pair);
      PrintFormat("[CHAOS_TRADE] ERROR: %s", m_lastError);
      return false;
   }

   return true;
}

#endif // CHAOS_TRADE_MANAGER_MQH

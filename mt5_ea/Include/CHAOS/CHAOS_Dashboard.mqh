//+------------------------------------------------------------------+
//|                                          CHAOS_Dashboard.mqh     |
//|                        CHAOS V1.0 - On-Chart Status Dashboard    |
//|                        Top-left panel with live system state      |
//+------------------------------------------------------------------+
#property copyright "CHAOS V1.0"
#property version   "1.00"
#property strict

#ifndef CHAOS_DASHBOARD_MQH
#define CHAOS_DASHBOARD_MQH

#include "CHAOS_Config.mqh"

//+------------------------------------------------------------------+
//| Dashboard Layout Constants                                        |
//+------------------------------------------------------------------+
#define DASH_PREFIX        "CHAOS_DASH_"
#define DASH_X_ORIGIN      10
#define DASH_Y_ORIGIN      25
#define DASH_WIDTH         280
#define DASH_ROW_HEIGHT    18
#define DASH_LABEL_X       16
#define DASH_VALUE_X       155
#define DASH_FONT_SIZE     9
#define DASH_HEADER_SIZE   10

// Colours
#define DASH_BG_COLOR          C'20,20,30'
#define DASH_BORDER_COLOR      C'60,60,80'
#define DASH_HEADER_COLOR      clrGold
#define DASH_LABEL_COLOR       C'180,180,200'
#define DASH_VALUE_COLOR       clrWhite
#define DASH_GREEN             C'0,200,80'
#define DASH_RED               C'220,50,50'
#define DASH_ORANGE            C'255,165,0'
#define DASH_FONT              "Consolas"

//+------------------------------------------------------------------+
//| Dashboard row names (order determines layout)                     |
//+------------------------------------------------------------------+
enum ENUM_DASH_ROW {
   ROW_HEADER = 0,
   ROW_MODE,
   ROW_SERVER,
   ROW_LATENCY,
   ROW_SEP1,
   ROW_POSITIONS,
   ROW_DAILY_PNL,
   ROW_DAILY_TRADES,
   ROW_WIN_RATE,
   ROW_SEP2,
   ROW_REGIME,
   ROW_DEFENSIVE,
   ROW_COOLDOWNS,
   ROW_SEP3,
   ROW_LAST_SIGNAL,
   ROW_LAST_ACTION,
   ROW_LAST_REASON,
   ROW_TOTAL
};

//+------------------------------------------------------------------+
//| Dashboard state — updated externally, rendered by UpdateDashboard |
//+------------------------------------------------------------------+
struct DashboardData {
   ENUM_CHAOS_MODE mode;
   bool            serverConnected;
   int             latencyMs;
   int             openPositions;
   double          dailyPnl;
   int             dailyTrades;
   double          winRate;         // 0-100
   string          regimeState;
   bool            defensiveMode;
   int             cooldownsActive;
   string          lastSignal;      // e.g. "EURUSD M30 LONG"
   string          lastAction;      // e.g. "BUY 0.10"
   string          lastReason;      // e.g. "COOLDOWN: 3 bars left"
};

DashboardData g_dashData;

//+------------------------------------------------------------------+
//| Helper: unique object name per row                                |
//+------------------------------------------------------------------+
string DashName(const string suffix) {
   return DASH_PREFIX + suffix;
}

//+------------------------------------------------------------------+
//| Helper: create background rectangle                               |
//+------------------------------------------------------------------+
void CreateDashBackground(int rows) {
   string name = DashName("BG");
   int height = DASH_Y_ORIGIN + (rows * DASH_ROW_HEIGHT) + 10;

   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

   ObjectSetInteger(0, name, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, DASH_X_ORIGIN - 6);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, DASH_Y_ORIGIN - 6);
   ObjectSetInteger(0, name, OBJPROP_XSIZE,     DASH_WIDTH);
   ObjectSetInteger(0, name, OBJPROP_YSIZE,     height);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR,   DASH_BG_COLOR);
   ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, DASH_BORDER_COLOR);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,     1);
   ObjectSetInteger(0, name, OBJPROP_BACK,      false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,    true);
}

//+------------------------------------------------------------------+
//| Helper: create a text label at a specific row                     |
//+------------------------------------------------------------------+
void CreateLabel(const string id, int row, int xOffset, const string text,
                 color clr, int fontSize = DASH_FONT_SIZE) {
   string name = DashName(id);

   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);

   ObjectSetInteger(0, name, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, DASH_X_ORIGIN + xOffset);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, DASH_Y_ORIGIN + (row * DASH_ROW_HEIGHT));
   ObjectSetString(0, name,  OBJPROP_TEXT,       text);
   ObjectSetString(0, name,  OBJPROP_FONT,       DASH_FONT);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,   fontSize);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_BACK,       false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE,  false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
}

//+------------------------------------------------------------------+
//| Helper: update existing label text and colour                     |
//+------------------------------------------------------------------+
void SetLabel(const string id, const string text, color clr = DASH_VALUE_COLOR) {
   string name = DashName(id);
   if(ObjectFind(0, name) >= 0) {
      ObjectSetString(0, name, OBJPROP_TEXT, text);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   }
}

//+------------------------------------------------------------------+
//| CreateDashboard — build all chart objects                          |
//+------------------------------------------------------------------+
void CreateDashboard() {
   // Initialize dashboard data defaults
   g_dashData.mode             = CHAOS_MODE_PAPER;
   g_dashData.serverConnected  = false;
   g_dashData.latencyMs        = 0;
   g_dashData.openPositions    = 0;
   g_dashData.dailyPnl         = 0.0;
   g_dashData.dailyTrades      = 0;
   g_dashData.winRate           = 0.0;
   g_dashData.regimeState      = "UNKNOWN";
   g_dashData.defensiveMode    = false;
   g_dashData.cooldownsActive  = 0;
   g_dashData.lastSignal       = "---";
   g_dashData.lastAction       = "---";
   g_dashData.lastReason       = "---";

   // Background panel
   CreateDashBackground(ROW_TOTAL);

   // Header
   CreateLabel("HDR", ROW_HEADER, DASH_LABEL_X - 6,
               "CHAOS V1.0", DASH_HEADER_COLOR, DASH_HEADER_SIZE);

   // Row labels (left column)
   CreateLabel("LBL_MODE",       ROW_MODE,         DASH_LABEL_X, "Mode:",           DASH_LABEL_COLOR);
   CreateLabel("LBL_SERVER",     ROW_SERVER,        DASH_LABEL_X, "Server:",         DASH_LABEL_COLOR);
   CreateLabel("LBL_LATENCY",    ROW_LATENCY,       DASH_LABEL_X, "Latency:",        DASH_LABEL_COLOR);
   CreateLabel("LBL_POSITIONS",  ROW_POSITIONS,     DASH_LABEL_X, "Positions:",      DASH_LABEL_COLOR);
   CreateLabel("LBL_PNL",        ROW_DAILY_PNL,     DASH_LABEL_X, "Daily PnL:",      DASH_LABEL_COLOR);
   CreateLabel("LBL_TRADES",     ROW_DAILY_TRADES,  DASH_LABEL_X, "Daily Trades:",   DASH_LABEL_COLOR);
   CreateLabel("LBL_WINRATE",    ROW_WIN_RATE,      DASH_LABEL_X, "Win Rate:",       DASH_LABEL_COLOR);
   CreateLabel("LBL_REGIME",     ROW_REGIME,        DASH_LABEL_X, "Regime:",         DASH_LABEL_COLOR);
   CreateLabel("LBL_DEFENSIVE",  ROW_DEFENSIVE,     DASH_LABEL_X, "Defensive:",      DASH_LABEL_COLOR);
   CreateLabel("LBL_COOLDOWNS",  ROW_COOLDOWNS,     DASH_LABEL_X, "Cooldowns:",      DASH_LABEL_COLOR);
   CreateLabel("LBL_SIGNAL",     ROW_LAST_SIGNAL,   DASH_LABEL_X, "Last Signal:",    DASH_LABEL_COLOR);
   CreateLabel("LBL_ACTION",     ROW_LAST_ACTION,   DASH_LABEL_X, "Last Action:",    DASH_LABEL_COLOR);
   CreateLabel("LBL_REASON",     ROW_LAST_REASON,   DASH_LABEL_X, "Last Reason:",    DASH_LABEL_COLOR);

   // Separator labels (visual dividers)
   CreateLabel("SEP1", ROW_SEP1, DASH_LABEL_X, "----------------------------", C'40,40,60');
   CreateLabel("SEP2", ROW_SEP2, DASH_LABEL_X, "----------------------------", C'40,40,60');
   CreateLabel("SEP3", ROW_SEP3, DASH_LABEL_X, "----------------------------", C'40,40,60');

   // Value labels (right column) — initialised with defaults
   CreateLabel("VAL_MODE",       ROW_MODE,         DASH_VALUE_X, "PAPER",         DASH_ORANGE);
   CreateLabel("VAL_SERVER",     ROW_SERVER,        DASH_VALUE_X, "DISCONNECTED",  DASH_RED);
   CreateLabel("VAL_LATENCY",    ROW_LATENCY,       DASH_VALUE_X, "--- ms",        DASH_VALUE_COLOR);
   CreateLabel("VAL_POSITIONS",  ROW_POSITIONS,     DASH_VALUE_X, "0/8",           DASH_VALUE_COLOR);
   CreateLabel("VAL_PNL",        ROW_DAILY_PNL,     DASH_VALUE_X, "$0.00",         DASH_VALUE_COLOR);
   CreateLabel("VAL_TRADES",     ROW_DAILY_TRADES,  DASH_VALUE_X, "0",             DASH_VALUE_COLOR);
   CreateLabel("VAL_WINRATE",    ROW_WIN_RATE,      DASH_VALUE_X, "0.0%",          DASH_VALUE_COLOR);
   CreateLabel("VAL_REGIME",     ROW_REGIME,        DASH_VALUE_X, "UNKNOWN",       DASH_VALUE_COLOR);
   CreateLabel("VAL_DEFENSIVE",  ROW_DEFENSIVE,     DASH_VALUE_X, "OFF",           DASH_GREEN);
   CreateLabel("VAL_COOLDOWNS",  ROW_COOLDOWNS,     DASH_VALUE_X, "0",             DASH_VALUE_COLOR);
   CreateLabel("VAL_SIGNAL",     ROW_LAST_SIGNAL,   DASH_VALUE_X, "---",           DASH_VALUE_COLOR);
   CreateLabel("VAL_ACTION",     ROW_LAST_ACTION,   DASH_VALUE_X, "---",           DASH_VALUE_COLOR);
   CreateLabel("VAL_REASON",     ROW_LAST_REASON,   DASH_VALUE_X, "---",           DASH_VALUE_COLOR);

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| UpdateDashboard — refresh all values from g_dashData              |
//+------------------------------------------------------------------+
void UpdateDashboard() {
   // Mode
   if(g_dashData.mode == CHAOS_MODE_LIVE)
      SetLabel("VAL_MODE", "LIVE", DASH_GREEN);
   else
      SetLabel("VAL_MODE", "PAPER", DASH_ORANGE);

   // Server
   if(g_dashData.serverConnected)
      SetLabel("VAL_SERVER", "CONNECTED", DASH_GREEN);
   else
      SetLabel("VAL_SERVER", "DISCONNECTED", DASH_RED);

   // Latency
   if(g_dashData.serverConnected) {
      color latClr = (g_dashData.latencyMs < 100) ? DASH_GREEN :
                     (g_dashData.latencyMs < 500) ? DASH_ORANGE : DASH_RED;
      SetLabel("VAL_LATENCY", StringFormat("%d ms", g_dashData.latencyMs), latClr);
   } else {
      SetLabel("VAL_LATENCY", "--- ms", DASH_LABEL_COLOR);
   }

   // Positions
   color posClr = (g_dashData.openPositions >= MAX_TOTAL_POSITIONS) ? DASH_RED :
                  (g_dashData.openPositions > 0) ? DASH_GREEN : DASH_VALUE_COLOR;
   SetLabel("VAL_POSITIONS",
            StringFormat("%d/%d", g_dashData.openPositions, MAX_TOTAL_POSITIONS), posClr);

   // Daily PnL
   color pnlClr = (g_dashData.dailyPnl > 0) ? DASH_GREEN :
                  (g_dashData.dailyPnl < 0) ? DASH_RED : DASH_VALUE_COLOR;
   string pnlSign = (g_dashData.dailyPnl >= 0) ? "+" : "";
   SetLabel("VAL_PNL", StringFormat("%s$%.2f", pnlSign, g_dashData.dailyPnl), pnlClr);

   // Daily Trades
   SetLabel("VAL_TRADES", IntegerToString(g_dashData.dailyTrades), DASH_VALUE_COLOR);

   // Win Rate
   color wrClr = (g_dashData.winRate >= 55.0) ? DASH_GREEN :
                 (g_dashData.winRate >= 45.0) ? DASH_VALUE_COLOR : DASH_RED;
   SetLabel("VAL_WINRATE", StringFormat("%.1f%%", g_dashData.winRate), wrClr);

   // Regime
   color regClr = DASH_VALUE_COLOR;
   if(g_dashData.regimeState == "TRENDING") regClr = DASH_GREEN;
   else if(g_dashData.regimeState == "VOLATILE") regClr = DASH_RED;
   else if(g_dashData.regimeState == "RANGING") regClr = DASH_ORANGE;
   SetLabel("VAL_REGIME", g_dashData.regimeState, regClr);

   // Defensive Mode
   if(g_dashData.defensiveMode)
      SetLabel("VAL_DEFENSIVE", "ON", DASH_RED);
   else
      SetLabel("VAL_DEFENSIVE", "OFF", DASH_GREEN);

   // Cooldowns Active
   color cdClr = (g_dashData.cooldownsActive > 0) ? DASH_ORANGE : DASH_VALUE_COLOR;
   SetLabel("VAL_COOLDOWNS", IntegerToString(g_dashData.cooldownsActive), cdClr);

   // Last Signal
   color sigClr = DASH_VALUE_COLOR;
   if(StringFind(g_dashData.lastSignal, "LONG") >= 0)  sigClr = DASH_GREEN;
   if(StringFind(g_dashData.lastSignal, "SHORT") >= 0) sigClr = DASH_RED;
   SetLabel("VAL_SIGNAL", g_dashData.lastSignal, sigClr);

   // Last Action
   color actClr = DASH_VALUE_COLOR;
   if(StringFind(g_dashData.lastAction, "BUY") >= 0)   actClr = DASH_GREEN;
   if(StringFind(g_dashData.lastAction, "SELL") >= 0)   actClr = DASH_RED;
   if(StringFind(g_dashData.lastAction, "SKIP") >= 0)   actClr = DASH_ORANGE;
   SetLabel("VAL_ACTION", g_dashData.lastAction, actClr);

   // Last Reason
   SetLabel("VAL_REASON", g_dashData.lastReason, DASH_LABEL_COLOR);

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| DestroyDashboard — remove all chart objects                       |
//+------------------------------------------------------------------+
void DestroyDashboard() {
   int total = ObjectsTotal(0);
   for(int i = total - 1; i >= 0; i--) {
      string name = ObjectName(0, i);
      if(StringFind(name, DASH_PREFIX) == 0) {
         ObjectDelete(0, name);
      }
   }
   ChartRedraw(0);
}

#endif // CHAOS_DASHBOARD_MQH

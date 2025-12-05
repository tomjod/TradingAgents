//+------------------------------------------------------------------+
//|                                               TradingBridge.mq5  |
//|                        Python-MQL5 Socket Bridge                  |
//|                        Ultra-low latency execution               |
//+------------------------------------------------------------------+
#property copyright "TradingAgents"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   InpHost = "127.0.0.1";     // Server IP
input int      InpPort = 5555;             // Server Port
input int      InpMagic = 999001;          // Magic Number
input string   InpSymbol = "XAUUSDm";      // Symbol to trade
input double   InpDefaultLot = 0.01;       // Default lot size
input int      InpSlippage = 10;           // Max slippage in points
input int      InpTickInterval = 100;      // Tick send interval (ms)

//--- Global variables
int            g_socket = INVALID_HANDLE;
bool           g_connected = false;
datetime       g_lastTickTime = 0;
CTrade         g_trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("TradingBridge EA Starting...");
    Print("Symbol: ", InpSymbol);
    Print("Port: ", InpPort);
    
    // Setup trade object
    g_trade.SetExpertMagicNumber(InpMagic);
    g_trade.SetDeviationInPoints(InpSlippage);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    // Create socket
    g_socket = SocketCreate();
    if(g_socket == INVALID_HANDLE)
    {
        Print("Error creating socket: ", GetLastError());
        return INIT_FAILED;
    }
    
    // Connect to Python server
    if(!SocketConnect(g_socket, InpHost, InpPort, 5000))
    {
        Print("Cannot connect to Python server at ", InpHost, ":", InpPort);
        Print("Error: ", GetLastError());
        Print("Make sure Python bridge_server.py is running!");
        SocketClose(g_socket);
        return INIT_FAILED;
    }
    
    g_connected = true;
    Print("✅ Connected to Python Brain at ", InpHost, ":", InpPort);
    
    // Send initial handshake
    SendMessage("{\"type\":\"handshake\",\"ea\":\"TradingBridge\",\"symbol\":\"" + InpSymbol + "\"}");
    
    // Set timer for regular updates
    EventSetMillisecondTimer(InpTickInterval);
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    if(g_socket != INVALID_HANDLE)
    {
        SendMessage("{\"type\":\"disconnect\"}");
        SocketClose(g_socket);
    }
    
    Print("TradingBridge EA Stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer function - sends tick data                                   |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Timer function - sends tick data                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    // Reconnection logic
    if(!g_connected)
    {
        static datetime last_attempt = 0;
        if(TimeCurrent() - last_attempt < 5) return; // Retry every 5s
        
        last_attempt = TimeCurrent();
        Print("🔄 Attempting to reconnect...");
        
        // Reset socket
        if(g_socket != INVALID_HANDLE) SocketClose(g_socket);
        g_socket = SocketCreate();
        
        if(SocketConnect(g_socket, InpHost, InpPort, 1000))
        {
             g_connected = true;
             Print("✅ Reconnected to Python Brain");
             SendMessage("{\"type\":\"handshake\",\"ea\":\"TradingBridge\",\"symbol\":\"" + InpSymbol + "\"}");
        }
        return;
    }
    
    // Get current tick
    MqlTick tick;
    if(!SymbolInfoTick(InpSymbol, tick)) return;
    
    // Only send if tick changed (Price or Time)
    static double last_bid = 0.0;
    static double last_ask = 0.0;
    
    if(tick.time == g_lastTickTime && 
       tick.bid == last_bid && 
       tick.ask == last_ask) return;
       
    g_lastTickTime = tick.time;
    last_bid = tick.bid;
    last_ask = tick.ask;
    
    // Build tick message
    string msg = StringFormat(
        "{\"type\":\"tick\",\"symbol\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"spread\":%.1f,\"time\":\"%s\"}",
        InpSymbol,
        tick.bid,
        tick.ask,
        (tick.ask - tick.bid) / SymbolInfoDouble(InpSymbol, SYMBOL_POINT),
        TimeToString(tick.time, TIME_DATE|TIME_SECONDS)
    );
    
    SendMessage(msg);
    
    // Check for incoming commands
    ProcessIncomingMessages();
}

//+------------------------------------------------------------------+
//| OnTick - also check messages on each tick                          |
//+------------------------------------------------------------------+
void OnTick()
{
    if(!g_connected) return;
    ProcessIncomingMessages();
}

//+------------------------------------------------------------------+
//| Send message to Python                                             |
//+------------------------------------------------------------------+
bool SendMessage(string message)
{
    if(g_socket == INVALID_HANDLE) return false;
    
    // Add newline as message delimiter
    message += "\n";
    
    uchar data[];
    StringToCharArray(message, data, 0, WHOLE_ARRAY, CP_UTF8);
    
    int len = ArraySize(data) - 1; // Exclude null terminator
    if(SocketSend(g_socket, data, len) != len)
    {
        Print("Send failed: ", GetLastError());
        g_connected = false;
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Process incoming messages from Python                              |
//+------------------------------------------------------------------+
void ProcessIncomingMessages()
{
    if(g_socket == INVALID_HANDLE) return;
    
    // Check if data available (non-blocking)
    uint available = SocketIsReadable(g_socket);
    if(available == 0) return;
    
    // Read data
    uchar buffer[];
    ArrayResize(buffer, available);
    
    int received = SocketRead(g_socket, buffer, available, 100);
    if(received <= 0) return;
    
    // Convert to string
    string message = CharArrayToString(buffer, 0, received, CP_UTF8);
    
    // Process each line (messages are newline-delimited)
    string lines[];
    int count = StringSplit(message, '\n', lines);
    
    for(int i = 0; i < count; i++)
    {
        if(StringLen(lines[i]) > 0)
        {
            ProcessCommand(lines[i]);
        }
    }
}

//+------------------------------------------------------------------+
//| Process a single command from Python                               |
//+------------------------------------------------------------------+
void ProcessCommand(string json)
{
    Print("📩 Received: ", json);
    
    // Parse action type
    string action = GetJsonValue(json, "action");
    
    if(action == "BUY")
    {
        ExecuteBuy(json);
    }
    else if(action == "SELL")
    {
        ExecuteSell(json);
    }
    else if(action == "CLOSE")
    {
        ClosePosition(json);
    }
    else if(action == "CLOSE_ALL")
    {
        CloseAllPositions();
    }
    else if(action == "GET_POSITIONS")
    {
        SendPositions();
    }
    else if(action == "GET_ACCOUNT")
    {
        SendAccountInfo();
    }
    else if(action == "GET_HISTORY")
    {
        int count = (int)StringToInteger(GetJsonValue(json, "count"));
        if(count <= 0) count = 200; // Default
        SendHistory(count);
    }
    else if(action == "MODIFY")
    {
        ExecuteModify(json);
    }
    else if(action == "PING")
    {
        SendMessage("{\"type\":\"pong\",\"time\":\"" + TimeToString(TimeCurrent()) + "\"}");
    }
    else
    {
        Print("Unknown action: ", action);
    }
}

//+------------------------------------------------------------------+
//| Modify Position (SL/TP)                                            |
//+------------------------------------------------------------------+
void ExecuteModify(string json)
{
    ulong ticket = (ulong)StringToInteger(GetJsonValue(json, "ticket"));
    double sl = StringToDouble(GetJsonValue(json, "sl"));
    double tp = StringToDouble(GetJsonValue(json, "tp"));
    
    if(g_trade.PositionModify(ticket, sl, tp))
    {
         string response = StringFormat(
             "{\"type\":\"execution\",\"action\":\"MODIFY\",\"status\":\"filled\",\"ticket\":%d,\"sl\":%.5f,\"tp\":%.5f}",
             ticket, sl, tp
         );
         SendMessage(response);
         Print("✅ Modified #", ticket, " SL:", sl, " TP:", tp);
    }
    else
    {
        string response = StringFormat(
             "{\"type\":\"execution\",\"action\":\"MODIFY\",\"status\":\"error\",\"error\":\"%s\",\"retcode\":%d}",
             g_trade.ResultComment(), g_trade.ResultRetcode()
         );
         SendMessage(response);
         Print("Modify failed: ", g_trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Send historical data                                               |
//+------------------------------------------------------------------+
void SendHistory(int count)
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    
    int copied = CopyRates(InpSymbol, PERIOD_M5, 0, count, rates);
    if(copied <= 0)
    {
        Print("Failed to copy rates: ", GetLastError());
        return;
    }
    
    Print("Sending ", copied, " M5 candles...");
    
    // Send start marker
    SendMessage("{\"type\":\"history_start\",\"count\":" + IntegerToString(copied) + "}");
    
    // Send each candle individually to avoid buffer overflow
    for(int i = copied - 1; i >= 0; i--)
    {
        string candle = StringFormat(
            "{\"type\":\"candle\",\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_vol\":%I64d}",
            TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
            rates[i].open,
            rates[i].high,
            rates[i].low,
            rates[i].close,
            rates[i].tick_volume
        );
        SendMessage(candle);
    }
    
    // Send end marker
    SendMessage("{\"type\":\"history_end\"}");
    Print("History sent.");
}

//+------------------------------------------------------------------+
//| Execute BUY order                                                  |
//+------------------------------------------------------------------+
void ExecuteBuy(string json)
{
    double volume = StringToDouble(GetJsonValue(json, "volume"));
    if(volume <= 0) volume = InpDefaultLot;
    
    double sl_points = StringToDouble(GetJsonValue(json, "sl_points"));
    double tp_points = StringToDouble(GetJsonValue(json, "tp_points"));
    
    MqlTick tick;
    SymbolInfoTick(InpSymbol, tick);
    
    double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
    double sl = (sl_points > 0) ? tick.ask - sl_points * point : 0;
    double tp = (tp_points > 0) ? tick.ask + tp_points * point : 0;
    
    string comment = GetJsonValue(json, "comment");
    if(comment == "") comment = "PythonBrain";
    
    if(g_trade.Buy(volume, InpSymbol, tick.ask, sl, tp, comment))
    {
        ulong ticket = g_trade.ResultOrder();
        double price = g_trade.ResultPrice();
        
        string response = StringFormat(
            "{\"type\":\"execution\",\"action\":\"BUY\",\"status\":\"filled\",\"ticket\":%d,\"price\":%.5f,\"volume\":%.2f}",
            ticket, price, volume
        );
        SendMessage(response);
        Print("✅ BUY executed: Ticket=", ticket, " Price=", price);
    }
    else
    {
        string response = StringFormat(
            "{\"type\":\"execution\",\"action\":\"BUY\",\"status\":\"error\",\"error\":\"%s\",\"retcode\":%d}",
            g_trade.ResultComment(), g_trade.ResultRetcode()
        );
        SendMessage(response);
        Print("❌ BUY failed: ", g_trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Execute SELL order                                                 |
//+------------------------------------------------------------------+
void ExecuteSell(string json)
{
    double volume = StringToDouble(GetJsonValue(json, "volume"));
    if(volume <= 0) volume = InpDefaultLot;
    
    double sl_points = StringToDouble(GetJsonValue(json, "sl_points"));
    double tp_points = StringToDouble(GetJsonValue(json, "tp_points"));
    
    MqlTick tick;
    SymbolInfoTick(InpSymbol, tick);
    
    double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
    double sl = (sl_points > 0) ? tick.bid + sl_points * point : 0;
    double tp = (tp_points > 0) ? tick.bid - tp_points * point : 0;
    
    string comment = GetJsonValue(json, "comment");
    if(comment == "") comment = "PythonBrain";
    
    if(g_trade.Sell(volume, InpSymbol, tick.bid, sl, tp, comment))
    {
        ulong ticket = g_trade.ResultOrder();
        double price = g_trade.ResultPrice();
        
        string response = StringFormat(
            "{\"type\":\"execution\",\"action\":\"SELL\",\"status\":\"filled\",\"ticket\":%d,\"price\":%.5f,\"volume\":%.2f}",
            ticket, price, volume
        );
        SendMessage(response);
        Print("✅ SELL executed: Ticket=", ticket, " Price=", price);
    }
    else
    {
        string response = StringFormat(
            "{\"type\":\"execution\",\"action\":\"SELL\",\"status\":\"error\",\"error\":\"%s\",\"retcode\":%d}",
            g_trade.ResultComment(), g_trade.ResultRetcode()
        );
        SendMessage(response);
        Print("❌ SELL failed: ", g_trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Close specific position                                            |
//+------------------------------------------------------------------+
void ClosePosition(string json)
{
    ulong ticket = (ulong)StringToInteger(GetJsonValue(json, "ticket"));
    
    if(g_trade.PositionClose(ticket))
    {
        SendMessage("{\"type\":\"close\",\"status\":\"success\",\"ticket\":" + IntegerToString(ticket) + "}");
        Print("✅ Position closed: ", ticket);
    }
    else
    {
        SendMessage("{\"type\":\"close\",\"status\":\"error\",\"ticket\":" + IntegerToString(ticket) + "}");
        Print("❌ Close failed: ", g_trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Close all positions                                                |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    int closed = 0;
    
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL) == InpSymbol)
            {
                if(g_trade.PositionClose(ticket))
                    closed++;
            }
        }
    }
    
    SendMessage("{\"type\":\"close_all\",\"closed\":" + IntegerToString(closed) + "}");
    Print("Closed ", closed, " positions");
}

//+------------------------------------------------------------------+
//| Send current positions to Python                                   |
//+------------------------------------------------------------------+
void SendPositions()
{
    string positions = "[";
    bool first = true;
    
    for(int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL) != InpSymbol) continue;
            
            if(!first) positions += ",";
            first = false;
            
            positions += StringFormat(
                "{\"ticket\":%d,\"type\":\"%s\",\"volume\":%.2f,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"profit\":%.2f}",
                ticket,
                (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL",
                PositionGetDouble(POSITION_VOLUME),
                PositionGetDouble(POSITION_PRICE_OPEN),
                PositionGetDouble(POSITION_SL),
                PositionGetDouble(POSITION_TP),
                PositionGetDouble(POSITION_PROFIT)
            );
        }
    }
    
    positions += "]";
    SendMessage("{\"type\":\"positions\",\"data\":" + positions + "}");
}

//+------------------------------------------------------------------+
//| Send account info to Python                                        |
//+------------------------------------------------------------------+
void SendAccountInfo()
{
    string info = StringFormat(
        "{\"type\":\"account\",\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,\"free_margin\":%.2f,\"margin_level\":%.2f}",
        AccountInfoDouble(ACCOUNT_BALANCE),
        AccountInfoDouble(ACCOUNT_EQUITY),
        AccountInfoDouble(ACCOUNT_MARGIN),
        AccountInfoDouble(ACCOUNT_MARGIN_FREE),
        AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)
    );
    SendMessage(info);
}

//+------------------------------------------------------------------+
//| Simple JSON value extractor                                        |
//+------------------------------------------------------------------+
string GetJsonValue(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if(start == -1) return "";
    
    start += StringLen(search);
    
    // Skip whitespace
    while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ')) start++;
    
    // Check if value is string (starts with ")
    if(StringGetCharacter(json, start) == '"')
    {
        start++;
        int end = StringFind(json, "\"", start);
        if(end == -1) return "";
        return StringSubstr(json, start, end - start);
    }
    
    // Numeric or other value
    int end = start;
    while(end < StringLen(json))
    {
        ushort c = StringGetCharacter(json, end);
        if(c == ',' || c == '}' || c == ']') break;
        end++;
    }
    
    return StringSubstr(json, start, end - start);
}
//+------------------------------------------------------------------+

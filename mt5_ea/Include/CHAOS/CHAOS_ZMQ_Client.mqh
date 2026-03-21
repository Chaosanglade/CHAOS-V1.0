//+------------------------------------------------------------------+
//| CHAOS_ZMQ_Client.mqh                                              |
//| CHAOS V1.0 — Communication Bridge                                 |
//| Dual-mode: ZeroMQ DLL or File-Based Bridge                        |
//+------------------------------------------------------------------+
#ifndef CHAOS_ZMQ_CLIENT_MQH
#define CHAOS_ZMQ_CLIENT_MQH

//+------------------------------------------------------------------+
//| Configuration                                                      |
//+------------------------------------------------------------------+
enum ENUM_BRIDGE_MODE
{
   BRIDGE_ZMQ  = 0,   // ZeroMQ via libzmq.dll
   BRIDGE_FILE = 1    // File-based bridge (safe fallback)
};

input ENUM_BRIDGE_MODE InpBridgeMode      = BRIDGE_FILE;          // Bridge mode
input string           InpZmqEndpoint     = "tcp://127.0.0.1:5555"; // ZMQ endpoint
input int              InpRequestTimeoutMs = 5000;                 // Request timeout (ms)
input int              InpHeartbeatIntervalSec = 10;               // Heartbeat interval (sec)
input string           InpBridgeFolder    = "CHAOS_Bridge";        // File bridge folder (in MQL5/Files/)

//+------------------------------------------------------------------+
//| ZeroMQ DLL imports (requires libzmq.dll in MQL5/Libraries/)       |
//+------------------------------------------------------------------+
#import "libzmq.dll"
   // Context
   long zmq_ctx_new();
   int  zmq_ctx_destroy(long context);
   // Socket
   long zmq_socket(long context, int type);
   int  zmq_close(long socket);
   int  zmq_connect(long socket, const char &endpoint[]);
   int  zmq_disconnect(long socket, const char &endpoint[]);
   int  zmq_setsockopt(long socket, int option_name, const char &option_value[], int option_len);
   int  zmq_setsockopt(long socket, int option_name, int &option_value, int option_len);
   // Messaging
   int  zmq_send(long socket, const char &buf[], int len, int flags);
   int  zmq_recv(long socket, char &buf[], int len, int flags);
   // Polling
   int  zmq_poll(int &items[], int nitems, long timeout);
   // Error
   int  zmq_errno();
   string zmq_strerror(int errnum);
#import

//+------------------------------------------------------------------+
//| ZMQ constants                                                      |
//+------------------------------------------------------------------+
#define ZMQ_REQ        3
#define ZMQ_RCVTIMEO  27
#define ZMQ_SNDTIMEO  28
#define ZMQ_LINGER    17
#define ZMQ_DONTWAIT   1

//+------------------------------------------------------------------+
//| Bridge response structure                                          |
//+------------------------------------------------------------------+
struct BridgeResponse
{
   bool   success;
   string payload;
   int    latencyMs;
};

//+------------------------------------------------------------------+
//| CHAOS_ZMQ_Client class                                             |
//+------------------------------------------------------------------+
class CHAOS_ZMQ_Client
{
private:
   // Mode
   ENUM_BRIDGE_MODE  m_mode;

   // ZMQ handles
   long              m_zmqContext;
   long              m_zmqSocket;
   bool              m_zmqConnected;
   string            m_zmqEndpoint;

   // File bridge
   string            m_bridgeFolder;
   string            m_requestFile;
   string            m_responseFile;
   string            m_heartbeatFile;
   int               m_requestCounter;

   // Shared state
   int               m_timeoutMs;
   int               m_heartbeatIntervalSec;
   datetime          m_lastHeartbeat;
   bool              m_connected;
   int               m_lastLatencyMs;
   string            m_lastError;

   // --- ZMQ private methods ---
   bool              ZmqConnect();
   void              ZmqDisconnect();
   string            ZmqSendRecv(string message);

   // --- File bridge private methods ---
   bool              FileWriteAtomic(string folder, string filename, string content);
   string            FilePollResponse(int timeoutMs);
   void              FileCleanup();
   string            GenerateRequestId();

public:
                     CHAOS_ZMQ_Client();
                    ~CHAOS_ZMQ_Client();

   // Initialization
   bool              Init(ENUM_BRIDGE_MODE mode = BRIDGE_FILE,
                          string zmqEndpoint = "tcp://127.0.0.1:5555",
                          int timeoutMs = 5000,
                          int heartbeatSec = 10,
                          string bridgeFolder = "CHAOS_Bridge");
   void              Close();

   // Communication
   string            SendRequest(string jsonRequest);
   bool              SendHeartbeat();

   // Status
   bool              IsConnected()       { return m_connected; }
   int               GetLastLatencyMs()  { return m_lastLatencyMs; }
   string            GetLastError()      { return m_lastError; }
   ENUM_BRIDGE_MODE  GetMode()           { return m_mode; }
};

//+------------------------------------------------------------------+
//| Constructor                                                        |
//+------------------------------------------------------------------+
CHAOS_ZMQ_Client::CHAOS_ZMQ_Client()
{
   m_zmqContext   = 0;
   m_zmqSocket    = 0;
   m_zmqConnected = false;
   m_zmqEndpoint  = "";
   m_bridgeFolder = "";
   m_requestFile  = "";
   m_responseFile = "";
   m_heartbeatFile = "";
   m_requestCounter = 0;
   m_timeoutMs    = 5000;
   m_heartbeatIntervalSec = 10;
   m_lastHeartbeat = 0;
   m_connected    = false;
   m_lastLatencyMs = 0;
   m_lastError    = "";
   m_mode         = BRIDGE_FILE;
}

//+------------------------------------------------------------------+
//| Destructor                                                         |
//+------------------------------------------------------------------+
CHAOS_ZMQ_Client::~CHAOS_ZMQ_Client()
{
   Close();
}

//+------------------------------------------------------------------+
//| Initialize the bridge                                              |
//+------------------------------------------------------------------+
bool CHAOS_ZMQ_Client::Init(ENUM_BRIDGE_MODE mode,
                            string zmqEndpoint,
                            int timeoutMs,
                            int heartbeatSec,
                            string bridgeFolder)
{
   m_mode         = mode;
   m_zmqEndpoint  = zmqEndpoint;
   m_timeoutMs    = timeoutMs;
   m_heartbeatIntervalSec = heartbeatSec;
   m_bridgeFolder = bridgeFolder;
   m_connected    = false;

   if(m_mode == BRIDGE_ZMQ)
   {
      PrintFormat("[CHAOS_ZMQ] Initializing ZeroMQ mode, endpoint=%s", m_zmqEndpoint);
      if(!ZmqConnect())
      {
         PrintFormat("[CHAOS_ZMQ] ZMQ init failed: %s — falling back to FILE mode", m_lastError);
         m_mode = BRIDGE_FILE;
         // Fall through to file init
      }
      else
      {
         m_connected = true;
         PrintFormat("[CHAOS_ZMQ] ZeroMQ connected to %s", m_zmqEndpoint);
         return true;
      }
   }

   // File bridge init
   PrintFormat("[CHAOS_ZMQ] Initializing File Bridge mode, folder=%s", m_bridgeFolder);

   m_requestFile   = m_bridgeFolder + "\\request.json";
   m_responseFile  = m_bridgeFolder + "\\response.json";
   m_heartbeatFile = m_bridgeFolder + "\\heartbeat.json";

   // Ensure the bridge folder exists by writing a test file
   string testFile = m_bridgeFolder + "\\init_test.tmp";
   int handle = FileOpen(testFile, FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      // Try without FILE_COMMON (MQL5/Files/ relative)
      handle = FileOpen(testFile, FILE_WRITE | FILE_TXT);
      if(handle == INVALID_HANDLE)
      {
         m_lastError = StringFormat("Cannot create bridge folder: %s (err=%d)",
                                    m_bridgeFolder, GetLastError());
         PrintFormat("[CHAOS_ZMQ] ERROR: %s", m_lastError);
         return false;
      }
   }
   FileWrite(handle, "CHAOS_INIT");
   FileClose(handle);
   FileDelete(testFile);

   // Clean up any stale files
   FileCleanup();

   m_connected = true;
   m_requestCounter = 0;
   PrintFormat("[CHAOS_ZMQ] File Bridge initialized. Folder: %s", m_bridgeFolder);
   return true;
}

//+------------------------------------------------------------------+
//| Close the bridge                                                   |
//+------------------------------------------------------------------+
void CHAOS_ZMQ_Client::Close()
{
   if(m_mode == BRIDGE_ZMQ)
   {
      ZmqDisconnect();
   }
   else
   {
      FileCleanup();
   }
   m_connected = false;
   PrintFormat("[CHAOS_ZMQ] Bridge closed (mode=%s)",
               m_mode == BRIDGE_ZMQ ? "ZMQ" : "FILE");
}

//+------------------------------------------------------------------+
//| Send request and receive response                                  |
//+------------------------------------------------------------------+
string CHAOS_ZMQ_Client::SendRequest(string jsonRequest)
{
   if(!m_connected)
   {
      m_lastError = "Not connected";
      return "";
   }

   uint startTick = GetTickCount();

   string response = "";

   if(m_mode == BRIDGE_ZMQ)
   {
      response = ZmqSendRecv(jsonRequest);
   }
   else
   {
      // File bridge mode
      m_requestCounter++;
      string reqId = GenerateRequestId();

      // Wrap the request with metadata
      string wrappedRequest = StringFormat(
         "{\"request_id\":\"%s\",\"timestamp\":%d,\"payload\":%s}",
         reqId, (int)TimeCurrent(), jsonRequest
      );

      // Write request atomically
      if(!FileWriteAtomic(m_bridgeFolder, "request.json", wrappedRequest))
      {
         m_lastError = "Failed to write request file";
         PrintFormat("[CHAOS_ZMQ] ERROR: %s", m_lastError);
         return "";
      }

      // Poll for response
      response = FilePollResponse(m_timeoutMs);

      if(response == "")
      {
         m_lastError = "Response timeout";
         PrintFormat("[CHAOS_ZMQ] ERROR: Request %s timed out after %d ms", reqId, m_timeoutMs);
         // Clean up the stale request
         FileDelete(m_requestFile);
      }
   }

   m_lastLatencyMs = (int)(GetTickCount() - startTick);

   if(response != "")
   {
      PrintFormat("[CHAOS_ZMQ] Request completed in %d ms", m_lastLatencyMs);
   }

   return response;
}

//+------------------------------------------------------------------+
//| Send heartbeat, check if server is alive                           |
//+------------------------------------------------------------------+
bool CHAOS_ZMQ_Client::SendHeartbeat()
{
   if(!m_connected)
      return false;

   // Throttle heartbeats
   if(TimeCurrent() - m_lastHeartbeat < m_heartbeatIntervalSec)
      return true; // Assume still alive within interval

   string request = "{\"type\":\"heartbeat\",\"timestamp\":" +
                    IntegerToString((int)TimeCurrent()) + "}";

   string response = SendRequest(request);

   if(response != "")
   {
      m_lastHeartbeat = TimeCurrent();
      return true;
   }

   m_lastError = "Heartbeat failed — server may be down";
   PrintFormat("[CHAOS_ZMQ] WARNING: %s", m_lastError);
   return false;
}

//+------------------------------------------------------------------+
//| ZMQ: Connect to endpoint                                           |
//+------------------------------------------------------------------+
bool CHAOS_ZMQ_Client::ZmqConnect()
{
   // Create context
   m_zmqContext = zmq_ctx_new();
   if(m_zmqContext == 0)
   {
      m_lastError = "zmq_ctx_new failed — is libzmq.dll in MQL5/Libraries/?";
      return false;
   }

   // Create REQ socket
   m_zmqSocket = zmq_socket(m_zmqContext, ZMQ_REQ);
   if(m_zmqSocket == 0)
   {
      m_lastError = "zmq_socket failed";
      zmq_ctx_destroy(m_zmqContext);
      m_zmqContext = 0;
      return false;
   }

   // Set timeouts
   int rcvTimeout = m_timeoutMs;
   int sndTimeout = m_timeoutMs;
   int linger = 0;
   zmq_setsockopt(m_zmqSocket, ZMQ_RCVTIMEO, rcvTimeout, sizeof(int));
   zmq_setsockopt(m_zmqSocket, ZMQ_SNDTIMEO, sndTimeout, sizeof(int));
   zmq_setsockopt(m_zmqSocket, ZMQ_LINGER, linger, sizeof(int));

   // Connect
   char endpoint[];
   StringToCharArray(m_zmqEndpoint, endpoint);
   int rc = zmq_connect(m_zmqSocket, endpoint);
   if(rc != 0)
   {
      int err = zmq_errno();
      m_lastError = StringFormat("zmq_connect failed: errno=%d", err);
      zmq_close(m_zmqSocket);
      zmq_ctx_destroy(m_zmqContext);
      m_zmqSocket = 0;
      m_zmqContext = 0;
      return false;
   }

   m_zmqConnected = true;
   return true;
}

//+------------------------------------------------------------------+
//| ZMQ: Disconnect                                                    |
//+------------------------------------------------------------------+
void CHAOS_ZMQ_Client::ZmqDisconnect()
{
   if(m_zmqSocket != 0)
   {
      if(m_zmqConnected && m_zmqEndpoint != "")
      {
         char endpoint[];
         StringToCharArray(m_zmqEndpoint, endpoint);
         zmq_disconnect(m_zmqSocket, endpoint);
      }
      zmq_close(m_zmqSocket);
      m_zmqSocket = 0;
   }
   if(m_zmqContext != 0)
   {
      zmq_ctx_destroy(m_zmqContext);
      m_zmqContext = 0;
   }
   m_zmqConnected = false;
}

//+------------------------------------------------------------------+
//| ZMQ: Send message and receive response                             |
//+------------------------------------------------------------------+
string CHAOS_ZMQ_Client::ZmqSendRecv(string message)
{
   if(!m_zmqConnected || m_zmqSocket == 0)
   {
      m_lastError = "ZMQ socket not connected";
      return "";
   }

   // Send
   char sendBuf[];
   int len = StringToCharArray(message, sendBuf) - 1; // exclude null terminator
   if(len <= 0)
   {
      m_lastError = "Empty message";
      return "";
   }

   int bytesSent = zmq_send(m_zmqSocket, sendBuf, len, 0);
   if(bytesSent < 0)
   {
      int err = zmq_errno();
      m_lastError = StringFormat("zmq_send failed: errno=%d", err);
      PrintFormat("[CHAOS_ZMQ] ERROR: %s", m_lastError);

      // REQ socket is now in bad state — reconnect
      ZmqDisconnect();
      ZmqConnect();
      return "";
   }

   // Receive
   char recvBuf[];
   ArrayResize(recvBuf, 65536); // 64KB buffer
   int bytesRecv = zmq_recv(m_zmqSocket, recvBuf, 65536, 0);
   if(bytesRecv < 0)
   {
      int err = zmq_errno();
      m_lastError = StringFormat("zmq_recv failed (timeout?): errno=%d", err);
      PrintFormat("[CHAOS_ZMQ] ERROR: %s", m_lastError);

      // REQ socket is now in bad state — reconnect
      ZmqDisconnect();
      ZmqConnect();
      return "";
   }

   string response = CharArrayToString(recvBuf, 0, bytesRecv);
   return response;
}

//+------------------------------------------------------------------+
//| File Bridge: Atomic write (write .tmp then rename)                 |
//+------------------------------------------------------------------+
bool CHAOS_ZMQ_Client::FileWriteAtomic(string folder, string filename, string content)
{
   string tmpFile = folder + "\\request.tmp";
   string finalFile = folder + "\\" + filename;

   // Write to .tmp file
   int handle = FileOpen(tmpFile, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("[CHAOS_ZMQ] FileOpen failed for %s, error=%d", tmpFile, GetLastError());
      return false;
   }

   FileWriteString(handle, content);
   FileClose(handle);

   // Delete existing target if present
   if(FileIsExist(finalFile))
      FileDelete(finalFile);

   // Rename .tmp -> final
   bool moved = FileMove(tmpFile, 0, finalFile, FILE_REWRITE);
   if(!moved)
   {
      // Fallback: if FileMove fails, write directly
      PrintFormat("[CHAOS_ZMQ] FileMove failed, writing directly to %s", finalFile);
      handle = FileOpen(finalFile, FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(handle == INVALID_HANDLE)
         return false;
      FileWriteString(handle, content);
      FileClose(handle);
   }

   return true;
}

//+------------------------------------------------------------------+
//| File Bridge: Poll for response file with timeout                   |
//+------------------------------------------------------------------+
string CHAOS_ZMQ_Client::FilePollResponse(int timeoutMs)
{
   uint startTick = GetTickCount();
   int pollIntervalMs = 50; // Check every 50ms

   while((int)(GetTickCount() - startTick) < timeoutMs)
   {
      if(FileIsExist(m_responseFile))
      {
         // Small delay to ensure file is fully written
         Sleep(10);

         int handle = FileOpen(m_responseFile, FILE_READ | FILE_TXT | FILE_ANSI);
         if(handle != INVALID_HANDLE)
         {
            string content = "";
            while(!FileIsEnding(handle))
            {
               content += FileReadString(handle);
            }
            FileClose(handle);

            // Delete response file after reading
            FileDelete(m_responseFile);

            if(StringLen(content) > 0)
               return content;
         }
      }

      Sleep(pollIntervalMs);
   }

   return ""; // Timeout
}

//+------------------------------------------------------------------+
//| File Bridge: Clean up stale files                                  |
//+------------------------------------------------------------------+
void CHAOS_ZMQ_Client::FileCleanup()
{
   if(FileIsExist(m_requestFile))
      FileDelete(m_requestFile);
   if(FileIsExist(m_responseFile))
      FileDelete(m_responseFile);
   if(FileIsExist(m_bridgeFolder + "\\request.tmp"))
      FileDelete(m_bridgeFolder + "\\request.tmp");
}

//+------------------------------------------------------------------+
//| Generate unique request ID                                         |
//+------------------------------------------------------------------+
string CHAOS_ZMQ_Client::GenerateRequestId()
{
   return StringFormat("REQ_%s_%06d",
                       TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
                       m_requestCounter);
}

#endif // CHAOS_ZMQ_CLIENT_MQH

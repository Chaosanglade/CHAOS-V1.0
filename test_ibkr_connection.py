from ib_insync import IB

ib = IB()
try:
    ib.connect("127.0.0.1", 7497, clientId=1, timeout=5)
    print("CONNECTED")
    print(ib.isConnected())
    ib.disconnect()
except Exception as e:
    print("CONNECT_FAIL")
    print(repr(e))

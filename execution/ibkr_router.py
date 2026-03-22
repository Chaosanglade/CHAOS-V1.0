from dataclasses import dataclass
from ib_insync import IB, Forex, MarketOrder


@dataclass
class OrderIntent:
    pair: str
    side: str
    quantity: float
    request_id: str
    paper: bool = True


class IBKRRouter:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    def connect(self) -> bool:
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=5)
            return self.ib.isConnected()
        except Exception:
            return False

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()

    def build_contract(self, pair: str):
        return Forex(pair)

    def submit_market_order(self, intent: OrderIntent):
        contract = self.build_contract(intent.pair)
        order = MarketOrder(intent.side, intent.quantity)
        trade = self.ib.placeOrder(contract, order)
        return trade


if __name__ == "__main__":
    router = IBKRRouter()
    ok = router.connect()
    print("CONNECTED" if ok else "CONNECT_FAIL")
    if ok:
        router.disconnect()

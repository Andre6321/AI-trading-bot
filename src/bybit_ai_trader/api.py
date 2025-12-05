# Contents of /bybit_ai_trader/bybit_ai_trader/src/bybit_ai_trader/api.py

import requests

class BybitAPI:
    BASE_URL = "https://api.bybit.com"

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def _get(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params)
        return self._handle_response(response)

    def _post(self, endpoint, data=None):
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.post(url, json=data)
        return self._handle_response(response)

    def _handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    def get_server_time(self):
        return self._get("/v2/public/time")

    def get_order_book(self, symbol):
        params = {"symbol": symbol}
        return self._get("/v2/public/orderBook/L2", params)

    def place_order(self, symbol, side, order_type, qty, price=None):
        data = {
            "api_key": self.api_key,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "qty": qty,
            "price": price,
            "time_in_force": "GoodTillCancel"
        }
        return self._post("/v2/private/order/create", data)
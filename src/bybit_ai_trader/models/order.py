class Order:
    def __init__(self, order_id, symbol, side, quantity, price, status='pending'):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.status = status

    def execute(self):
        self.status = 'executed'

    def cancel(self):
        self.status = 'canceled'

    def __repr__(self):
        return f"Order({self.order_id}, {self.symbol}, {self.side}, {self.quantity}, {self.price}, {self.status})"
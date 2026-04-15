import time
import pandas as pd
import random
import numpy as np
from datetime import datetime
from . import config
from .events import FillEvent
from tqdm import tqdm
from scipy.stats import truncnorm


def _generate_order_id():
    ts = int(time.time() * 1000)
    rand = random.randint(1, 9999)
    return f"{ts}_{rand:04d}"


class Order:

    def __init__(self, id, direction, price, size, type, origin="market_maker", level=0):
        self._id = id
        self._direction = direction
        self._price = price
        self._size = size
        self._type = type
        self._origin = origin
        self._level = level   # ladder level (1 = best quote, 0 = not a ladder order)
        self._order_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @property
    def _dict_repr(self):
        return {
            "Id": self._id,
            "Direction": self._direction,
            "Price": self._price,
            "Size": self._size,
            "Type": self._type,
            "Origin": self._origin,
            "Level": self._level,
            "Time": self._order_time,
        }


class Order_book:

    def __init__(self, spread_init=0.1):

        #initial spread when randomly initializing the order book
        self._spread_init = spread_init
        self._df_order_book = pd.DataFrame(columns=["Id", "Direction", "Price", "Size", "Type", "Origin", "Level", "Time"])
        self._df_matches = pd.DataFrame(columns=["MatchId", "ClientOrderId", "MmOrderId", "Direction", "Price", "MatchedSize", "Time"])

        self._listener = []

        # [ADDED] Registry of resting MM orders: {order_id -> {"price", "direction", "age"}}
        # Updated on every add/cancel so the Quoter can inspect resting state without
        # querying the full DataFrame each step.
        self._mm_resting: dict = {}

        # [ADDED] Quoter callback — set via register_quoter_listener().
        # Called with a FillEvent whenever a MM order is (partially or fully) matched.
        self._fill_callback = None

        # [ADDED] Step counter incremented externally via tick() to track order age.
        self._current_step: int = 0

    @property
    def _df_bid_book(self):
        return self._df_order_book[self._df_order_book["Direction"] == "buy"].drop(columns=["Direction"])

    @property
    def _df_ask_book(self):
        return self._df_order_book[self._df_order_book["Direction"] == "sell"].drop(columns=["Direction"])

    def __repr__(self):

        global_book = (
            "====== Global Order Book ======\n"
            f"{self._df_order_book.head()}\n\n"
            "=> Stats:\n"
            f"{self._df_order_book.describe(include='all')}\n"
        )

        bid_book = (
            "====== Buy Order Book ======\n"
            f"{self._df_bid_book.head()}\n\n"
            "=> Stats:\n"
            f"{self._df_bid_book.describe(include='all')}\n"
        )

        ask_book = (
            "====== Sell Order Book ======\n"
            f"{self._df_ask_book.head()}\n\n"
            "=> Stats:\n"
            f"{self._df_ask_book.describe(include='all')}\n"
        )

        return f"{global_book}\n{bid_book}\n{ask_book}"

    # [ADDED] Register the Quoter's on_fill callback so the book can notify it on fills.
    def register_quoter_listener(self, callback) -> None:
        """
        callback must match the signature: callback(FillEvent) -> None
        Typically this is quoter.on_fill.
        """
        self._fill_callback = callback

    # [ADDED] Called by the simulation loop at each step to age all resting MM orders.
    def tick(self, step: int) -> None:
        """Advance the internal step counter and increment age of all resting MM orders."""
        self._current_step = step
        for oid in self._mm_resting:
            self._mm_resting[oid]["age"] += 1

    # [ADDED] Expose resting MM order registry to the Quoter (read-only view).
    @property
    def mm_resting_orders(self) -> dict:
        """
        Returns a snapshot of all resting MM orders as:
          {order_id: {"price": float, "direction": str, "level": int, "age": int}}
        The Quoter uses this to decide which orders to cancel selectively.
        """
        return dict(self._mm_resting)

    def add_order(self, order: Order):
        self._df_order_book.loc[order._id] = order._dict_repr

        # [ADDED] Track MM orders in the resting registry with age = 0
        if order._origin == "market_maker":
            self._mm_resting[order._id] = {
                "price":     order._price,
                "direction": order._direction,
                "level":     order._level,
                "age":       0,
            }

    def generate_price_from_last(self, last_price, mu=1, sigma=0.05, order="buy"):

        if order == "buy":
            a, b = -np.inf, (1.1 - mu) / sigma
        else:
            a, b = (0.9 - mu) / sigma, np.inf

        multiplier = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        return multiplier * last_price

    def return_last_buy_order(self):
        return self._df_bid_book.sort_values(["Time"]).iloc[-1]

    def return_last_sell_order(self):
        return self._df_ask_book.sort_values(["Time"]).iloc[-1]

    def _generate_random_order(self, origin="market_maker"):

        if self._df_order_book.empty:

            order_type = random.choice(config.ORDER_TYPE)

            if order_type == "limit_order":

                #first bid order
                direction = "buy"
                price_buy = round(np.random.normal(100, 0.02), 4)
                size_buy = abs(round(np.random.normal(2000, 100)))
                self.add_order(Order(_generate_order_id(), direction, price_buy, size_buy, order_type, origin))

                #first ask order
                direction = "sell"
                price_sell = round(price_buy * (1 + self._spread_init), 4)
                size_sell = abs(round(np.random.normal(2000, 100)))
                self.add_order(Order(_generate_order_id(), direction, price_sell, size_sell, order_type, origin))

        else:
            order_type = random.choice(config.ORDER_TYPE)
            direction = random.choice(config.ORDER_DIRECTION)

            if direction == "buy":
                last_price = self.return_last_buy_order()["Price"]
                new_price = self.generate_price_from_last(last_price, order="buy")
                # never let a bid cross the best ask
                if not self._df_ask_book.empty and origin == "market_maker":
                    best_ask = self._df_ask_book["Price"].min()
                    new_price = min(new_price, best_ask - 0.0001)
                size = abs(round(np.random.normal(2000, 100)))
                self.add_order(Order(_generate_order_id(), direction, round(new_price, 4), size, order_type, origin))

            if direction == "sell":
                last_price = self.return_last_sell_order()["Price"]
                new_price = self.generate_price_from_last(last_price, order="sell")
                # never let an ask cross the best bid
                if not self._df_bid_book.empty and origin == "market_maker":
                    best_bid = self._df_bid_book["Price"].max()
                    new_price = max(new_price, best_bid + 0.0001)
                size = abs(round(np.random.normal(2000, 100)))
                self.add_order(Order(_generate_order_id(), direction, round(new_price, 4), size, order_type, origin))

    def cancel_orders(self, ids: list):
        """Cancel orders by ID."""
        self._df_order_book = self._df_order_book.drop(
            index=[i for i in ids if i in self._df_order_book.index]
        )
        # [ADDED] Keep resting registry in sync
        for oid in ids:
            self._mm_resting.pop(oid, None)

    def cancel_all_mm_orders(self):
        """Cancel all market maker orders (convenience for full repricing)."""
        mm_ids = self._df_order_book[self._df_order_book["Origin"] == "market_maker"].index.tolist()
        self.cancel_orders(mm_ids)

    def _generate_n_random_order(self, number_of_order):
        for _ in tqdm(range(0, number_of_order), "Generating order book"):
            self._generate_random_order()

    def generate_random_orders(self, n, origin="market_maker") -> list:
        """Generate n random Orders from current book state without adding them.
        Requires the book to be non-empty (seed with _generate_n_random_order first)."""
        if self._df_order_book.empty:
            raise ValueError("Book is empty — seed with _generate_n_random_order first")
        orders = []
        for _ in range(n):
            order_type = random.choice(config.ORDER_TYPE)
            direction = random.choice(config.ORDER_DIRECTION)
            if direction == "buy":
                last_price = self.return_last_buy_order()["Price"]
                new_price = self.generate_price_from_last(last_price, order="buy")
                if not self._df_ask_book.empty and origin == "market_maker":
                    new_price = min(new_price, self._df_ask_book["Price"].min() - 0.0001)
            else:
                last_price = self.return_last_sell_order()["Price"]
                new_price = self.generate_price_from_last(last_price, order="sell")
                if not self._df_bid_book.empty and origin == "market_maker":
                    new_price = max(new_price, self._df_bid_book["Price"].max() + 0.0001)
            size = abs(round(np.random.normal(2000, 100)))
            orders.append(Order(_generate_order_id(), direction, round(new_price, 4), size, order_type, origin))
        return orders

    def add_orders_batch(self, orders: list):
        """Add a list of Order objects to the book, then attempt clearing."""
        for order in tqdm(orders, desc="Adding orders"):
            self.add_order(order)
        self._try_clear()

    def _try_clear(self):
        """Match client orders against market maker orders (price-time priority, partial fills)."""

        # --- client buys vs MM asks ---
        client_buy_ids = self._df_order_book[
            (self._df_order_book["Direction"] == "buy") &
            (self._df_order_book["Origin"] == "client")
        ].sort_values(["Price", "Time"], ascending=[False, True]).index.tolist()

        for cb_id in client_buy_ids:
            if cb_id not in self._df_order_book.index:
                continue

            mm_ask_ids = self._df_order_book[
                (self._df_order_book["Direction"] == "sell") &
                (self._df_order_book["Origin"] == "market_maker")
            ].sort_values(["Price", "Time"], ascending=[True, True]).index.tolist()

            for ma_id in mm_ask_ids:
                if cb_id not in self._df_order_book.index:
                    break
                if ma_id not in self._df_order_book.index:
                    continue

                cb_price = self._df_order_book.loc[cb_id, "Price"]
                ma_price = self._df_order_book.loc[ma_id, "Price"]

                if cb_price < ma_price:
                    break  # MM asks sorted asc — no cheaper ask exists

                cb_size = self._df_order_book.loc[cb_id, "Size"]
                ma_size = self._df_order_book.loc[ma_id, "Size"]
                matched_size = min(cb_size, ma_size)
                ma_level = self._df_order_book.loc[ma_id, "Level"]

                self._df_matches.loc[len(self._df_matches)] = {
                    "MatchId": _generate_order_id(),
                    "ClientOrderId": cb_id,
                    "MmOrderId": ma_id,
                    "Direction": "buy",
                    "Price": ma_price,
                    "MatchedSize": matched_size,
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                }

                self._df_order_book.loc[cb_id, "Size"] -= matched_size
                self._df_order_book.loc[ma_id, "Size"] -= matched_size

                if self._df_order_book.loc[ma_id, "Size"] == 0:
                    self._df_order_book = self._df_order_book.drop(ma_id)
                    # [ADDED] Remove from resting registry and notify the Quoter
                    self._mm_resting.pop(ma_id, None)
                    self._fire_fill(ma_id, "sell", ma_price, matched_size, ma_level, is_full_fill=True)

                if cb_id in self._df_order_book.index and self._df_order_book.loc[cb_id, "Size"] == 0:
                    self._df_order_book = self._df_order_book.drop(cb_id)

                # [ADDED] Partial fill on MM ask: fire fill event for the matched portion
                # but leave the order in the book (not yet fully consumed)
                elif ma_id in self._df_order_book.index and matched_size > 0:
                    self._fire_fill(ma_id, "sell", ma_price, matched_size, ma_level, is_full_fill=False)

        # --- client sells vs MM bids ---
        client_sell_ids = self._df_order_book[
            (self._df_order_book["Direction"] == "sell") &
            (self._df_order_book["Origin"] == "client")
        ].sort_values(["Price", "Time"], ascending=[True, True]).index.tolist()

        for cs_id in client_sell_ids:
            if cs_id not in self._df_order_book.index:
                continue

            mm_bid_ids = self._df_order_book[
                (self._df_order_book["Direction"] == "buy") &
                (self._df_order_book["Origin"] == "market_maker")
            ].sort_values(["Price", "Time"], ascending=[False, True]).index.tolist()

            for mb_id in mm_bid_ids:
                if cs_id not in self._df_order_book.index:
                    break
                if mb_id not in self._df_order_book.index:
                    continue

                cs_price = self._df_order_book.loc[cs_id, "Price"]
                mb_price = self._df_order_book.loc[mb_id, "Price"]

                if cs_price > mb_price:
                    break  # MM bids sorted desc — no higher bid exists

                cs_size = self._df_order_book.loc[cs_id, "Size"]
                mb_size = self._df_order_book.loc[mb_id, "Size"]
                matched_size = min(cs_size, mb_size)
                mb_level = self._df_order_book.loc[mb_id, "Level"]

                self._df_matches.loc[len(self._df_matches)] = {
                    "MatchId": _generate_order_id(),
                    "ClientOrderId": cs_id,
                    "MmOrderId": mb_id,
                    "Direction": "sell",
                    "Price": mb_price,
                    "MatchedSize": matched_size,
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                }

                self._df_order_book.loc[cs_id, "Size"] -= matched_size
                self._df_order_book.loc[mb_id, "Size"] -= matched_size

                if self._df_order_book.loc[mb_id, "Size"] == 0:
                    self._df_order_book = self._df_order_book.drop(mb_id)
                    # [ADDED] Remove from resting registry and notify the Quoter
                    self._mm_resting.pop(mb_id, None)
                    self._fire_fill(mb_id, "buy", mb_price, matched_size, mb_level, is_full_fill=True)

                if cs_id in self._df_order_book.index and self._df_order_book.loc[cs_id, "Size"] == 0:
                    self._df_order_book = self._df_order_book.drop(cs_id)

                # [ADDED] Partial fill on MM bid: fire fill event for the matched portion
                elif mb_id in self._df_order_book.index and matched_size > 0:
                    self._fire_fill(mb_id, "buy", mb_price, matched_size, mb_level, is_full_fill=False)

    # [ADDED] Internal helper: fire a FillEvent to the registered Quoter callback.
    def _fire_fill(self, order_id: str, direction: str, price: float, size: float, level: int, is_full_fill: bool) -> None:
        """
        Notify the Quoter that one of its resting orders was (partially or fully) matched.
        Only fires if a callback has been registered via register_quoter_listener().
        direction is the MM order direction ("buy" = MM bid was hit, "sell" = MM ask was hit).
        level is the ladder level of the filled order (1 = best quote).
        """
        if self._fill_callback is None:
            return

        self._fill_callback(FillEvent(
            order_id     = order_id,
            direction    = direction,
            price        = price,
            size         = size,
            step         = self._current_step,
            level        = level,
            is_full_fill = is_full_fill,
        ))

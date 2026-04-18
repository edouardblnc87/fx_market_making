import random
import numpy as np
import pandas as pd
from datetime import datetime
from . import config
from .events import FillEvent
from tqdm import tqdm
from scipy.stats import truncnorm

# ── Fast order-ID generator (simple counter, no datetime/random overhead) ────
_id_counter: int = 0

def _generate_order_id() -> str:
    global _id_counter
    _id_counter += 1
    return str(_id_counter)

def generate_order_id() -> str:
    """Public wrapper — use this in notebooks/tests to generate order IDs."""
    return _generate_order_id()


class Order:

    def __init__(self, id, direction, price, size, type, origin="market_maker", level=0):
        self._id        = id
        self._direction = direction
        self._price     = price
        self._size      = size
        self._type      = type
        self._origin    = origin
        self._level     = level

    @property
    def _dict_repr(self):
        return {
            "Id":        self._id,
            "Direction": self._direction,
            "Price":     self._price,
            "Size":      self._size,
            "Type":      self._type,
            "Origin":    self._origin,
            "Level":     self._level,
        }


class Order_book:

    def __init__(self, spread_init: float = 0.1, n_levels: int = 10):
        self._spread_init   = spread_init
        self.n_levels       = n_levels

        # Hot-path storage: plain dict, no pandas overhead.
        # {order_id: {direction, price, size, origin, level, seq}}
        self._orders: dict = {}
        self._seq: int = 0          # monotonic sequence for time-priority

        # Match log: list of dicts, converted to DataFrame on demand.
        self._match_log: list = []

        # Resting MM registry (unchanged public contract).
        self._mm_resting: dict = {}

        self._fill_callback = None
        self._current_step: int = 0

    # ── Backward-compatible DataFrame views (display / analysis only) ─────────

    @property
    def _df_order_book(self) -> pd.DataFrame:
        if not self._orders:
            return pd.DataFrame(columns=["Id", "Direction", "Price", "Size",
                                         "Type", "Origin", "Level"])
        rows = [{"Id": oid, **o} for oid, o in self._orders.items()]
        df = pd.DataFrame(rows).set_index("Id")
        # remove internal seq column from public view
        return df.drop(columns=["seq"], errors="ignore")

    @property
    def _df_bid_book(self) -> pd.DataFrame:
        df = self._df_order_book
        return df[df["Direction"] == "buy"].drop(columns=["Direction"])

    @property
    def _df_ask_book(self) -> pd.DataFrame:
        df = self._df_order_book
        return df[df["Direction"] == "sell"].drop(columns=["Direction"])

    @property
    def _df_matches(self) -> pd.DataFrame:
        if not self._match_log:
            return pd.DataFrame(columns=["MatchId", "ClientOrderId", "MmOrderId",
                                         "Direction", "Price", "MatchedSize",
                                         "Level", "Step"])
        return pd.DataFrame(self._match_log)

    # ── Core interface ────────────────────────────────────────────────────────

    def register_quoter_listener(self, callback) -> None:
        self._fill_callback = callback

    def tick(self, step: int) -> None:
        self._current_step = step
        for oid in self._mm_resting:
            self._mm_resting[oid]["age"] += 1

    @property
    def mm_resting_orders(self) -> dict:
        return dict(self._mm_resting)

    def add_order(self, order: Order) -> None:
        self._seq += 1
        self._orders[order._id] = {
            "direction": order._direction,
            "price":     order._price,
            "size":      order._size,
            "origin":    order._origin,
            "level":     order._level,
            "seq":       self._seq,
        }
        if order._origin == "market_maker":
            self._mm_resting[order._id] = {
                "price":          order._price,
                "direction":      order._direction,
                "level":          order._level,
                "age":            0,
                "original_size":  order._size,
                "remaining_size": order._size,
            }

    def cancel_orders(self, ids: list) -> None:
        for oid in ids:
            self._orders.pop(oid, None)
            self._mm_resting.pop(oid, None)

    def cancel_all_mm_orders(self) -> None:
        mm_ids = [oid for oid, o in self._orders.items() if o["origin"] == "market_maker"]
        self.cancel_orders(mm_ids)

    def post_mm_quotes(self, quotes: list) -> None:
        for order in quotes:
            self.add_order(order)

    def route_client_order(self, order: Order) -> None:
        self.add_order(order)
        self.try_clear()

    # ── Matching engine ───────────────────────────────────────────────────────

    def try_clear(self) -> None:
        """
        Match client orders against MM orders (price-time priority, partial fills).

        MM orders are sorted ONCE per try_clear() call — not once per client order.
        All state lives in self._orders (plain dict), avoiding pandas overhead.
        """
        orders = self._orders

        # ── Client buys vs MM asks ────────────────────────────────────────────
        mm_asks = sorted(
            [(oid, o) for oid, o in orders.items()
             if o["origin"] == "market_maker" and o["direction"] == "sell"],
            key=lambda x: (x[1]["price"], x[1]["seq"]),
        )
        client_buys = sorted(
            [(oid, o) for oid, o in orders.items()
             if o["origin"] == "client" and o["direction"] == "buy"],
            key=lambda x: (-x[1]["price"], x[1]["seq"]),
        )

        for cb_id, _ in client_buys:
            if cb_id not in orders:
                continue
            for ma_id, ma in mm_asks:
                if cb_id not in orders:
                    break
                if ma_id not in orders:
                    continue
                if orders[cb_id]["price"] < ma["price"]:
                    break

                matched = min(orders[cb_id]["size"], orders[ma_id]["size"])
                orders[cb_id]["size"] -= matched
                orders[ma_id]["size"] -= matched

                self._match_log.append({
                    "MatchId":       _generate_order_id(),
                    "ClientOrderId": cb_id,
                    "MmOrderId":     ma_id,
                    "Direction":     "buy",
                    "Price":         ma["price"],
                    "MatchedSize":   matched,
                    "Level":         ma["level"],
                    "Step":          self._current_step,
                })

                if orders[ma_id]["size"] == 0:
                    del orders[ma_id]
                    self._mm_resting.pop(ma_id, None)
                    self._fire_fill(ma_id, "sell", ma["price"], matched, ma["level"], True)
                elif matched > 0:
                    self._fire_fill(ma_id, "sell", ma["price"], matched, ma["level"], False)

                if cb_id in orders and orders[cb_id]["size"] == 0:
                    del orders[cb_id]

        # ── Client sells vs MM bids ───────────────────────────────────────────
        mm_bids = sorted(
            [(oid, o) for oid, o in orders.items()
             if o["origin"] == "market_maker" and o["direction"] == "buy"],
            key=lambda x: (-x[1]["price"], x[1]["seq"]),
        )
        client_sells = sorted(
            [(oid, o) for oid, o in orders.items()
             if o["origin"] == "client" and o["direction"] == "sell"],
            key=lambda x: (x[1]["price"], x[1]["seq"]),
        )

        for cs_id, _ in client_sells:
            if cs_id not in orders:
                continue
            for mb_id, mb in mm_bids:
                if cs_id not in orders:
                    break
                if mb_id not in orders:
                    continue
                if orders[cs_id]["price"] > mb["price"]:
                    break

                matched = min(orders[cs_id]["size"], orders[mb_id]["size"])
                orders[cs_id]["size"] -= matched
                orders[mb_id]["size"] -= matched

                self._match_log.append({
                    "MatchId":       _generate_order_id(),
                    "ClientOrderId": cs_id,
                    "MmOrderId":     mb_id,
                    "Direction":     "sell",
                    "Price":         mb["price"],
                    "MatchedSize":   matched,
                    "Level":         mb["level"],
                    "Step":          self._current_step,
                })

                if orders[mb_id]["size"] == 0:
                    del orders[mb_id]
                    self._mm_resting.pop(mb_id, None)
                    self._fire_fill(mb_id, "buy", mb["price"], matched, mb["level"], True)
                elif matched > 0:
                    self._fire_fill(mb_id, "buy", mb["price"], matched, mb["level"], False)

                if cs_id in orders and orders[cs_id]["size"] == 0:
                    del orders[cs_id]

    def _fire_fill(self, order_id, direction, price, size, level, is_full_fill) -> None:
        if not is_full_fill and order_id in self._mm_resting:
            self._mm_resting[order_id]["remaining_size"] -= size
        if self._fill_callback is None:
            return
        self._fill_callback(FillEvent(
            order_id=order_id,
            direction=direction,
            price=price,
            size=size,
            step=self._current_step,
            level=level,
            is_full_fill=is_full_fill,
        ))

    # ── Display helpers ───────────────────────────────────────────────────────

    def __repr__(self):
        df = self._df_order_book
        return (
            f"Order_book — {len(self._orders)} resting orders "
            f"({sum(1 for o in self._orders.values() if o['direction']=='buy')} bids, "
            f"{sum(1 for o in self._orders.values() if o['direction']=='sell')} asks)\n"
            f"{df.head()}"
        )

    def display_mm_quotes(self) -> None:
        mm = [(oid, o) for oid, o in self._orders.items() if o["origin"] == "market_maker"]
        if not mm:
            print("No resting MM orders in the book.")
            return
        asks = sorted([(o["level"], o["price"], o["size"]) for _, o in mm if o["direction"] == "sell"])
        bids = sorted([(o["level"], o["price"], o["size"]) for _, o in mm if o["direction"] == "buy"],
                      key=lambda x: -x[1])
        print("=" * 52)
        print(f"{'ASKS (sell)':<52}")
        print("-" * 52)
        print(f"  {'Lvl':>4}  {'Price':>12}  {'Size':>14}")
        print("-" * 52)
        for lvl, price, size in asks:
            print(f"  {int(lvl):>4}  {price:>12.5f}  {size:>14.2f}")
        print("=" * 52)
        print(f"  {'--- MID ---':^46}")
        print("=" * 52)
        for lvl, price, size in bids:
            print(f"  {int(lvl):>4}  {price:>12.5f}  {size:>14.2f}")
        print("-" * 52)
        print(f"{'BIDS (buy)':<52}")
        print("=" * 52)
        print(f"  Total MM orders: {len(mm)}  ({len(asks)} asks, {len(bids)} bids)")

    # ── Legacy / random-order helpers (used in exploratory notebooks) ─────────

    def generate_price_from_last(self, last_price, mu=1, sigma=0.05, order="buy"):
        if order == "buy":
            a, b = -np.inf, (1.1 - mu) / sigma
        else:
            a, b = (0.9 - mu) / sigma, np.inf
        return truncnorm.rvs(a, b, loc=mu, scale=sigma) * last_price

    def return_last_buy_order(self):
        return self._df_bid_book.sort_values(["seq"] if "seq" in self._df_bid_book.columns else []).iloc[-1]

    def return_last_sell_order(self):
        return self._df_ask_book.sort_values(["seq"] if "seq" in self._df_ask_book.columns else []).iloc[-1]

    def _generate_random_order(self, origin="market_maker"):
        if not self._orders:
            price_buy  = round(np.random.normal(100, 0.02), 4)
            size_buy   = abs(round(np.random.normal(2000, 100)))
            self.add_order(Order(_generate_order_id(), "buy",  price_buy,  size_buy,  "limit_order", origin))
            price_sell = round(price_buy * (1 + self._spread_init), 4)
            size_sell  = abs(round(np.random.normal(2000, 100)))
            self.add_order(Order(_generate_order_id(), "sell", price_sell, size_sell, "limit_order", origin))
        else:
            direction  = random.choice(config.ORDER_DIRECTION)
            order_type = random.choice(config.ORDER_TYPE)
            bids = [(o["price"], oid) for oid, o in self._orders.items() if o["direction"] == "buy"]
            asks = [(o["price"], oid) for oid, o in self._orders.items() if o["direction"] == "sell"]
            if direction == "buy" and bids:
                last_price = max(bids)[0]
                new_price  = self.generate_price_from_last(last_price, order="buy")
                if asks and origin == "market_maker":
                    new_price = min(new_price, min(asks)[0] - 0.0001)
                self.add_order(Order(_generate_order_id(), "buy", round(new_price, 4),
                                     abs(round(np.random.normal(2000, 100))), order_type, origin))
            elif direction == "sell" and asks:
                last_price = min(asks)[0]
                new_price  = self.generate_price_from_last(last_price, order="sell")
                if bids and origin == "market_maker":
                    new_price = max(new_price, max(bids)[0] + 0.0001)
                self.add_order(Order(_generate_order_id(), "sell", round(new_price, 4),
                                     abs(round(np.random.normal(2000, 100))), order_type, origin))

    def _generate_n_random_order(self, number_of_order):
        for _ in tqdm(range(number_of_order), desc="Generating order book"):
            self._generate_random_order()

    def generate_random_orders(self, n, origin="market_maker") -> list:
        if not self._orders:
            raise ValueError("Book is empty — seed with _generate_n_random_order first")
        orders = []
        for _ in range(n):
            direction  = random.choice(config.ORDER_DIRECTION)
            order_type = random.choice(config.ORDER_TYPE)
            bids = [o["price"] for o in self._orders.values() if o["direction"] == "buy"]
            asks = [o["price"] for o in self._orders.values() if o["direction"] == "sell"]
            if direction == "buy" and bids:
                new_price = self.generate_price_from_last(max(bids), order="buy")
                if asks and origin == "market_maker":
                    new_price = min(new_price, min(asks) - 0.0001)
            elif direction == "sell" and asks:
                new_price = self.generate_price_from_last(min(asks), order="sell")
                if bids and origin == "market_maker":
                    new_price = max(new_price, max(bids) + 0.0001)
            else:
                continue
            orders.append(Order(_generate_order_id(), direction, round(new_price, 4),
                                abs(round(np.random.normal(2000, 100))), order_type, origin))
        return orders

    def _add_orders_batch(self, orders: list):
        for order in tqdm(orders, desc="Adding orders"):
            self.add_order(order)

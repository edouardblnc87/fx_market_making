from dataclasses import dataclass


@dataclass
class FillEvent:
    """Fired by the OrderBook and consumed by the Quoter."""
    order_id:     str
    direction:    str    # "buy" or "sell" — the MM order direction that was hit
    price:        float
    size:         float  # matched portion only (may be < original order size)
    step:         int
    level:        int    # ladder level of the filled MM order (1 = best, 10 = deepest)
    is_full_fill: bool = True  # False = partial, order still resting in the book

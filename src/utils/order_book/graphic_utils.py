"""Visualization utilities for the order book."""

from . import order_book_impl
import matplotlib.pyplot as plt

def plot_order_book(order_book : order_book_impl.Order_book):
    """Plot the cumulative bid and ask depth of the order book as a step chart."""

    if order_book._df_bid_book.empty or order_book._df_ask_book.empty:
        print("Order book is empty.")
        return

    bid = (
        order_book._df_bid_book
        .groupby("Price", as_index=False)["Size"]
        .sum()
        .sort_values("Price", ascending=False)
    )

    ask = (
        order_book._df_ask_book
        .groupby("Price", as_index=False)["Size"]
        .sum()
        .sort_values("Price", ascending=True)
    )

    bid["CumSize"] = bid["Size"].cumsum()
    ask["CumSize"] = ask["Size"].cumsum()

    bid = bid.sort_values("Price")
    ask = ask.sort_values("Price")

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    ax.step(
        bid["Price"],
        bid["CumSize"],
        where="post",
        linewidth=2.5,
        color="#00ff88"
    )

    ax.fill_between(
        bid["Price"],
        bid["CumSize"],
        step="post",
        alpha=0.25,
        color="#00ff88"
    )

    ax.step(
        ask["Price"],
        ask["CumSize"],
        where="post",
        linewidth=2.5,
        color="#ff4d4d"
    )

    ax.fill_between(
        ask["Price"],
        ask["CumSize"],
        step="post",
        alpha=0.25,
        color="#ff4d4d"
    )

    ax.set_xlabel("Price", fontsize=13, color="white")
    ax.set_ylabel("Cumulative Size", fontsize=13, color="white")
    ax.set_title("Order Book Depth", fontsize=16, color="white", pad=15)

    ax.tick_params(colors="white")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    plt.tight_layout()
    plt.show()

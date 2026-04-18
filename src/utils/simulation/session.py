import pickle
import pathlib

_DEFAULT_PATH = pathlib.Path(__file__).parent / "saved_session.pkl"


def save_session(stock, market_B, market_C, market_maker, book, controller,
                 path=None):
    """
    Pickle the full simulation state to disk.

    Parameters
    ----------
    path : file path (str or Path). Defaults to simulation/saved_session.pkl
    """
    path = pathlib.Path(path) if path else _DEFAULT_PATH
    session = {
        "stock":        stock,
        "market_B":     market_B,
        "market_C":     market_C,
        "market_maker": market_maker,
        "book":         book,
        "controller":   controller,
    }
    with open(path, "wb") as f:
        pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Session saved → {path}")


def load_session(path=None):
    """
    Load a pickled simulation session from disk.

    Returns
    -------
    stock, market_B, market_C, market_maker, book, controller
    """
    path = pathlib.Path(path) if path else _DEFAULT_PATH
    with open(path, "rb") as f:
        session = pickle.load(f)
    print(f"Session loaded ← {path}")
    return (
        session["stock"],
        session["market_B"],
        session["market_C"],
        session["market_maker"],
        session["book"],
        session["controller"],
    )

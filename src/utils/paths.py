import os

def get_market_path(market):
    """
    Returns the strict, single source of truth directory path for market outputs.
    Ensures the directory exists before returning.
    """
    path = f"outputs/results/{market}/"
    os.makedirs(path, exist_ok=True)
    return path

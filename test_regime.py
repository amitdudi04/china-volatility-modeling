import pandas as pd
import numpy as np
import os
from src.data_loader import DataLoader
from src.regime_model import RegimeModel
from src.utils import get_market_path

loader = DataLoader()
data = loader.run()["CSI_300"]

rm = RegimeModel()
res = rm.build_regime_proxy(data, "CSI_300")
print("Regime execution passed.")

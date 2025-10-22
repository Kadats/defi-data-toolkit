import pandas as pd
import numpy as np
from defi_data_toolkit.indicators import calculate_sma

def test_calculate_sma_simples():
    """Testa o cálculo de uma SMA simples."""
    # 1. Preparação (Arrange)
    data = {'Close': [10, 12, 14, 16, 18]}
    df = pd.DataFrame(data)
    window = 3

    # 2. Ação (Act)
    sma_series = calculate_sma(df, column='Close', window=window)

    # 3. Verificação (Assert)
    # Os dois primeiros valores devem ser NaN (não há dados suficientes)
    assert pd.isna(sma_series.iloc[0])
    assert pd.isna(sma_series.iloc[1])
    # O terceiro valor é a média de (10, 12, 14) = 12
    assert sma_series.iloc[2] == 12.0
    # O quarto valor é a média de (12, 14, 16) = 14
    assert sma_series.iloc[3] == 14.0
    # O quinto valor é a média de (14, 16, 18) = 16
    assert sma_series.iloc[4] == 16.0


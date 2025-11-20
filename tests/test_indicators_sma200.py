import pandas as pd
from defi_data_toolkit.indicators import calculate_sma


def test_calculate_sma_200_basic():
    """Verifica que SMA com janela 200 produz NaNs nos primeiros 199 elementos e
    calcula corretamente o primeiro valor disponível como média dos 200 primeiros."""
    # cria 201 valores simples: 1..201
    n = 201
    closes = list(range(1, n + 1))
    df = pd.DataFrame({"Close": closes})

    sma = calculate_sma(df, column="Close", window=200)

    # primeiros 199 elementos devem ser NaN
    assert sma.iloc[0:199].isna().all()

    # o elemento no índice 199 (200º valor) deve ser a média de 1..200
    expected = sum(range(1, 201)) / 200.0
    assert abs(sma.iloc[199] - expected) < 1e-9

    # o elemento seguinte (índice 200) deve ser a média de 2..201
    expected2 = sum(range(2, 202)) / 200.0
    assert abs(sma.iloc[200] - expected2) < 1e-9

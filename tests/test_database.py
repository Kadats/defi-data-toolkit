# Em tests/test_database.py
import sqlite3
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from defi_data_toolkit.database import create_connection, save_klines_to_db

# Usamos o 'mocker' do pytest-mock
@pytest.fixture
def mock_sqlite(mocker):
    """Cria um mock para toda a biblioteca sqlite3."""
    mock_conn = MagicMock(spec=sqlite3.Connection)
    mock_cursor = MagicMock(spec=sqlite3.Cursor)
    mock_conn.cursor.return_value = mock_cursor

    # Simula a função 'connect'
    mock_connect = mocker.patch('sqlite3.connect', return_value=mock_conn)

    return {
        "connect": mock_connect,
        "conn": mock_conn,
        "cursor": mock_cursor
    }

def test_create_connection_success(mock_sqlite):
    """Testa se a conexão é criada com o arquivo correto."""
    db_path = "/tmp/test.db"
    conn = create_connection(db_path)

    # Verifica se 'sqlite3.connect' foi chamado com o caminho certo
    mock_sqlite["connect"].assert_called_with(db_path)
    assert conn is not None

def test_save_klines_to_db(mock_sqlite):
    """Testa se 'save_klines_to_db' tenta commitar dados no banco."""
    # 1. Preparação (Arrange)
    data = {
        'Open_time': [pd.Timestamp('2025-10-20'), pd.Timestamp('2025-10-21')],
        'Close_time': [pd.Timestamp('2025-10-20 23:59'), pd.Timestamp('2025-10-21 23:59')],
        'Open': [100, 110], 'High': [120, 130], 'Low': [90, 105], 'Close': [110, 120],
        'Volume': [1000, 2000], 'Quote_asset_volume': [110000, 220000], 'Number_of_trades': [50, 60],
        'Taker_buy_base_asset_volume': [500, 1000], 'Taker_buy_quote_asset_volume': [55000, 110000]
    }
    df = pd.DataFrame(data)

    # 2. Ação (Act)
    save_klines_to_db(df, "test_table", "dummy.db")

    # 3. Verificação (Assert) - AJUSTADA
    # Em vez de assert_called_once, verificamos se foi chamado PELO MENOS uma vez.
    # Isso acomoda o commit dentro de create_table e qualquer outro commit potencial.
    mock_sqlite["conn"].commit.assert_called()

    # Opcional: Verificar se o cursor foi usado para executar algo (indicativo de create_table ou to_sql)
    mock_sqlite["cursor"].execute.assert_called()


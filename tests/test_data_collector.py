# Em tests/test_data_collector.py
import pytest
import pandas as pd
from unittest.mock import MagicMock, call, patch
from defi_data_toolkit.data_collector import (
    get_klines_from_api, get_fear_and_greed_index, get_uniswap_pool_daily_data,
    fetch_all_klines,
    APIClient
)

@pytest.fixture
def mock_api_client(mocker):
    """Cria um mock para a classe APIClient."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Simula a resposta JSON da Binance para klines (lista de listas)
    mock_response.json.return_value = [
        [1672531200000, "40000", "41000", "39000", "40500", "1000", 1672617599999, "40500000", 100, "500", "20250000", "0"],
        [1672617600000, "40500", "41500", "40000", "41000", "1200", 1672703999999, "49200000", 120, "600", "24600000", "0"]
    ]

    mock_get = MagicMock(return_value=mock_response)

    # Substitui o método 'get' da classe APIClient pelo nosso mock
    mocker.patch.object(APIClient, 'get', mock_get)

    return mock_get # Retorna o mock do método 'get' para podermos verificar chamadas

def test_get_klines_from_api_success(mock_api_client):
    """Testa se get_klines_from_api processa uma resposta válida."""
    # 1. Arrange (já feito pela fixture)
    symbol = "BTCUSDT"
    interval = "1d"

    # 2. Act
    result = get_klines_from_api(symbol, interval, binance_api_base_url="http://dummy.com") # URL é irrelevante por causa do mock

    # 3. Assert
    # Verifica se a função retornou uma lista com 2 itens (as duas velas)
    assert isinstance(result, list)
    assert len(result) == 2
    # Verifica se o método 'get' do APIClient foi chamado
    mock_api_client.assert_called_once()

def test_get_fear_and_greed_index_success(mock_api_client):
    """Testa se get_fear_and_greed_index processa uma resposta válida."""
    # 1. Arrange
    # Configura o mock_response para simular a resposta da API F&G
    mock_response_fng = MagicMock()
    mock_response_fng.status_code = 200
    mock_response_fng.json.return_value = {
        "name": "Fear and Greed Index",
        "data": [
            {"value": "30", "value_classification": "Fear", "timestamp": "1672617600"}, # Mais recente
            {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1672531200"} # Mais antigo
        ],
        "metadata": {"error": None}
    }
    # Fazemos o mock_api_client (que é o mock do método 'get') retornar esta nova resposta
    # Usamos side_effect para que ele retorne respostas diferentes em chamadas diferentes (se necessário),
    # mas neste caso, como só há uma chamada, podemos só reconfigurar o return_value.
    # Vamos reconfigurar o mock_get para esta resposta específica.
    mock_api_client.return_value = mock_response_fng

    # 2. Act
    result = get_fear_and_greed_index(limit=2, fng_api_url="http://dummy_fng.com")

    # 3. Assert
    # Verifica se a função retornou uma lista com 2 itens
    assert isinstance(result, list)
    assert len(result) == 2
    # Verifica se os dados foram invertidos corretamente (mais antigo primeiro)
    assert result[0]['value'] == "25"
    assert result[1]['value'] == "30"
    # Verifica se o método 'get' do APIClient foi chamado com os parâmetros corretos
    mock_api_client.assert_called_once_with(
        "http://dummy_fng.com", # A URL base foi passada corretamente
        params={'limit': 2}     # Os parâmetros da query foram passados corretamente
    )

@pytest.fixture
def mock_thegraph_post(mocker):
    """Cria um mock específico para o session.post usado pelo TheGraph."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Simula a resposta JSON da API GraphQL do The Graph
    mock_response.json.return_value = {
        "data": {
            "poolDayDatas": [
                {"date": "1672531200", "volumeUSD": "10000.50", "tvlUSD": "5000000.00"},
                {"date": "1672617600", "volumeUSD": "12000.75", "tvlUSD": "5100000.00"}
            ]
        }
    }
    
    # IMPORTANTE: Mockamos o MÉTODO POST da SESSÃO do APIClient
    # Usamos patch diretamente na classe requests.Session que o APIClient usa internamente
    mock_session_post = mocker.patch('requests.Session.post', return_value=mock_response)
    
    return mock_session_post # Retorna o mock do método 'post'

def test_get_uniswap_pool_daily_data_success(mock_thegraph_post):
    """Testa se get_uniswap_pool_daily_data processa uma resposta GraphQL válida."""
    # 1. Arrange
    pool_id = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" # Exemplo
    api_key = "dummy_key"
    subgraph_ids = {"polygon": "dummy_subgraph_id"}
    
    # 2. Act
    result = get_uniswap_pool_daily_data(
        pool_id=pool_id,
        thegraph_api_key=api_key,
        thegraph_subgraph_ids=subgraph_ids,
        default_network='polygon',
        thegraph_base_url="https://dummy_thegraph.com/api/"
    )
    
    # 3. Assert
    # Verifica se a função retornou uma lista com 2 itens
    assert isinstance(result, list)
    assert len(result) == 2
    # Verifica a conversão dos dados (timestamp para ms, strings para float)
    assert result[0]['timestamp'] == 1672531200000
    assert result[0]['volumeUSD'] == 10000.50
    assert result[0]['tvlUSD'] == 5000000.00
    
    # Verifica se o método 'post' da sessão foi chamado
    mock_thegraph_post.assert_called_once()
    # Pega os argumentos da chamada para verificar a URL e o payload
    args, kwargs = mock_thegraph_post.call_args
    # Verifica a URL construída
    expected_url = f"https://dummy_thegraph.com/api/{api_key}/subgraphs/id/{subgraph_ids['polygon']}"
    assert args[0] == expected_url
    # Verifica se a query GraphQL contém o pool_id correto (inspeciona o payload JSON)
    assert f'pool: "{pool_id.lower()}"' in kwargs['json']['query']
    # Verifica se o cabeçalho de Autorização foi enviado
    assert 'Authorization' in kwargs['headers']
    assert kwargs['headers']['Authorization'] == f'Bearer {api_key}'

def test_fetch_all_klines_looping(mocker):
    """Testa se fetch_all_klines faz múltiplos pedidos para buscar todo o histórico."""
    # 1. Arrange
    symbol = "BTCUSDT"
    interval = "1h"
    start_ts_ms = 1672531200000 # 2023-01-01 00:00:00
    end_ts_ms = 1672531200000 + (3 * 3600 * 1000) # 3 horas depois

    # Simula a função get_klines_from_api (que fetch_all_klines chama internamente)
    # Vamos fazer ela retornar diferentes dados em chamadas consecutivas
    mock_get_klines = mocker.patch('defi_data_toolkit.data_collector.get_klines_from_api')

    # Resposta da primeira chamada (simula buscar as últimas 2 horas)
    response_1 = [
        # Hora 2
        [start_ts_ms + (2*3600*1000), "40500", "41500", "40000", "41000", "1200", start_ts_ms + (3*3600*1000)-1, "49200000", 120, "600", "24600000", "0"],
        # Hora 1
        [start_ts_ms + (1*3600*1000), "40000", "41000", "39000", "40500", "1000", start_ts_ms + (2*3600*1000)-1, "40500000", 100, "500", "20250000", "0"],
    ]
    # Resposta da segunda chamada (simula buscar a hora 0)
    response_2 = [
         # Hora 0
        [start_ts_ms, "39500", "40000", "39000", "40000", "800", start_ts_ms + (1*3600*1000)-1, "31600000", 80, "400", "15800000", "0"],
    ]

    # Configura o mock para retornar as respostas em sequência
    mock_get_klines.side_effect = [response_1, response_2, []] # A terceira chamada retorna vazio para parar o loop

    # 2. Act
    result_df = fetch_all_klines(symbol, interval, start_ts_ms, end_ts_ms, max_klines_per_request=2) # Limite pequeno para forçar o loop

    # 3. Assert
    # Verifica se o DataFrame resultante tem 3 linhas (3 horas)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 3
    # Verifica se a função get_klines_from_api foi chamada 3 vezes
    assert mock_get_klines.call_count == 2
    # Verifica os argumentos da primeira chamada (busca a partir do end_timestamp)
    mock_get_klines.assert_any_call(symbol, interval, 2, end_ts_ms, binance_api_base_url=None)
    # Verifica os argumentos da segunda chamada (busca a partir do timestamp da vela mais antiga da primeira resposta)
    expected_second_call_end_time = response_1[1][0] - 1 # Timestamp da hora 1 (mais antiga) menos 1ms
    mock_get_klines.assert_any_call(symbol, interval, 2, expected_second_call_end_time, binance_api_base_url=None)
    # Verifica a ordem das velas no resultado final (mais antiga primeiro)
    assert result_df['Open_time'].iloc[0].value // 10**6 == start_ts_ms
    assert result_df['Open_time'].iloc[2].value // 10**6 == start_ts_ms + (2*3600*1000)


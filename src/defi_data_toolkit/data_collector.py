import json
import time
import logging
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Module is configuration-free; callers must inject configuration values.
DEFAULT_KLINES_LIMIT = 1000
_DEFAULT_POLYGON_POOL_ID = "0x847b64f9d3a95e977d157866447a5c0a5dfa0ee5"


class APIClient:
    """HTTP client with Session, retries and exponential backoff.

    This client is safe to reuse across calls and APIs. It mounts a
    `HTTPAdapter` configured with `urllib3.util.Retry` so transient errors
    (connection errors, timeouts, and 5xx responses) are retried using
    exponential backoff.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        status_forcelist=(500, 502, 503, 504),
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.session = requests.Session()

        # store retry policy for use in explicit retry loop
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist

        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _build_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/{path_or_url.lstrip('/')}"
        return path_or_url

    def get(self, path_or_url: str, params: dict = None, headers: dict = None) -> requests.Response:
        url = self._build_url(path_or_url)
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as exc:
                # Decide whether to retry: for HTTPError, check status code if available
                status = None
                if hasattr(exc, "response") and exc.response is not None:
                    status = getattr(exc.response, "status_code", None)
                # If this was the last attempt, re-raise
                if attempt == attempts:
                    raise
                # If we have a status code and it's not in status_forcelist, do not retry
                if status is not None and status not in self.status_forcelist:
                    raise
                # Otherwise sleep with exponential backoff and retry
                sleep_for = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_for)
                continue


# No module-level APIClient instances; callers must pass base_url and the client


# Coletas Binance
def get_klines_from_api(symbol: str, interval: str, limit: int = DEFAULT_KLINES_LIMIT, end_time: int = None, binance_api_base_url: str = None) -> list:
    """Coleta dados de velas (OHLCV) da API da Binance.

    Mantém a assinatura e o comportamento de retorno (lista) inalterados.
    """
    endpoint = "/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time

    if not binance_api_base_url:
        raise ValueError("binance_api_base_url must be provided to get_klines_from_api")
    base_url = binance_api_base_url
    client = APIClient(base_url=base_url, timeout=10, max_retries=3, backoff_factor=1)
    try:
        resp = client.get(endpoint, params=params)
        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.error("Erro ao decodificar a resposta JSON da Binance (klines): %s", resp.text)
            return []
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar à API da Binance para klines (url=%s): %s", client._build_url(endpoint), e)
        return []

def fetch_all_klines(symbol: str, interval: str, start_timestamp: int, end_timestamp: int, max_klines_per_request: int = DEFAULT_KLINES_LIMIT, binance_api_base_url: str = None) -> pd.DataFrame:
    """
    Coleta todas as velas entre start_timestamp e end_timestamp, lidando com o limite da API.

    Args:
        symbol (str): O par de trading (ex: "BTCUSDT").
        interval (str): O período da vela (ex: "1h", "1d").
        start_timestamp (int): Timestamp de início em milissegundos.
        end_timestamp (int): Timestamp de término em milissegundos.
        max_klines_per_request (int): Limite máximo de velas por requisição da API.

    Returns:
        pd.DataFrame: Um DataFrame Pandas com os dados OHLCV.
    """
    all_data = []
    current_end_time = end_timestamp

    while True:
        klines = get_klines_from_api(symbol, interval, max_klines_per_request, current_end_time, binance_api_base_url=binance_api_base_url)
        if not klines:
            break

        klines_df = pd.DataFrame(klines, columns=[
            'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_asset_volume', 'Number_of_trades',
            'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
        ])
        
        klines_df['Open_time'] = pd.to_datetime(klines_df['Open_time'], unit='ms')
        klines_df['Close_time'] = pd.to_datetime(klines_df['Close_time'], unit='ms')
        
        # Converte colunas numéricas para float (originalmente são strings)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'Quote_asset_volume', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume']
        for col in numeric_cols:
            klines_df[col] = pd.to_numeric(klines_df[col])

        if 'Ignore' in klines_df.columns:
            klines_df = klines_df.drop(columns=['Ignore'])

        klines_df = klines_df[klines_df['Open_time'] >= pd.to_datetime(start_timestamp, unit='ms')]
        
        if klines_df.empty:
            break

        all_data.insert(0, klines_df)
        
        current_end_time = klines_df['Open_time'].min().value // 10**6 - 1

        if klines_df['Open_time'].min() <= pd.to_datetime(start_timestamp, unit='ms'):
            break

        time.sleep(0.1)

    if all_data:
        final_df = pd.concat(all_data).drop_duplicates(subset=['Open_time']).sort_values('Open_time').reset_index(drop=True)
        return final_df
    return pd.DataFrame()

def get_funding_rate_history(symbol: str, limit: int = 100, binance_futures_api_base_url: str = None) -> list:
    """
    Coleta o histórico de Funding Rate da API da Binance Futures.

    Args:
        symbol (str): O par de trading (ex: "BTCUSDT").
        limit (int): O número de entradas a retornar (máximo 1000).

    Returns:
        list: Uma lista de dicionários com os dados do Funding Rate, ou uma lista vazia.
    """
    endpoint = "/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    if not binance_futures_api_base_url:
        raise ValueError("binance_futures_api_base_url must be provided to get_funding_rate_history")
    base_url = binance_futures_api_base_url
    client = APIClient(base_url=base_url, timeout=10, max_retries=3, backoff_factor=1)
    try:
        resp = client.get(endpoint, params=params)
        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.error("Erro ao decodificar a resposta JSON do Funding Rate: %s", resp.text)
            return []
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar à API da Binance Futures para Funding Rate (url=%s): %s", client._build_url(endpoint), e)
        return []

def get_open_interest(symbol: str, binance_futures_api_base_url: str = None) -> dict:
    """
    Coleta o Open Interest atual da API da Binance Futures.

    Args:
        symbol (str): O par de trading (ex: "BTCUSDT").

    Returns:
        dict: Um dicionário com os dados do Open Interest, ou None em caso de erro.
    """
    endpoint = "/fapi/v1/openInterest"
    params = {"symbol": symbol}
    if not binance_futures_api_base_url:
        raise ValueError("binance_futures_api_base_url must be provided to get_open_interest")
    base_url = binance_futures_api_base_url
    client = APIClient(base_url=base_url, timeout=10, max_retries=3, backoff_factor=1)
    try:
        resp = client.get(endpoint, params=params)
        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.error("Erro ao decodificar a resposta JSON do Open Interest: %s", resp.text)
            return None
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar à API da Binance Futures para Open Interest (url=%s): %s", client._build_url(endpoint), e)
        return None


# Coletas Fear and Greed Index
def get_fear_and_greed_index(limit: int = 1, start_date_unix_sec: int = None, fng_api_url: str = None) -> list: # Modificado para retornar lista
    """
    Coleta o Fear and Greed Index da Alternative.me API.

    Args:
        limit (int): Número de dias de dados a retornar (padrão: 1 para o mais recente).
        start_date_unix_sec (int): Timestamp de início em segundos Unix (opcional).

    Returns:
        list: Uma lista de dicionários contendo os dados do índice, ou lista vazia em caso de erro.
    """
    params = {"limit": limit}
    if start_date_unix_sec:
        params["date_from"] = start_date_unix_sec  # A API FNG usa 'date_from'

    if not fng_api_url:
        raise ValueError("fng_api_url must be provided to get_fear_and_greed_index")
    base_url = fng_api_url
    client = APIClient(base_url=base_url, timeout=10, max_retries=2, backoff_factor=0.5)
    try:
        resp = client.get(base_url, params=params)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("Erro ao decodificar a resposta JSON do Fear and Greed Index: %s", resp.text)
            return []

        if data and "data" in data and len(data["data"]) > 0:
            # A API retorna os dados mais recentes primeiro, precisamos inverter para o DB
            return list(reversed(data["data"]))
        return []
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar à API do Fear and Greed Index (url=%s): %s", client._build_url(base_url), e)
        return []


# Coletas On-Chain Blockchair    
def get_bitcoin_network_fees(blockchair_api_url: str = None) -> dict:
    """
    Coleta estatísticas básicas da rede Bitcoin, incluindo taxas de transação.

    Returns:
        dict: Um dicionário com as estatísticas da rede, ou None em caso de erro.
    """
    if not blockchair_api_url:
        raise ValueError("blockchair_api_url must be provided to get_bitcoin_network_fees")
    base_url = blockchair_api_url
    client = APIClient(base_url=base_url, timeout=10, max_retries=2, backoff_factor=0.5)
    try:
        resp = client.get(base_url)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("Erro ao decodificar a resposta JSON da Blockchair: %s", resp.text)
            return None

        if data and "data" in data:
            return data["data"]
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar à API da Blockchair (url=%s): %s", client._build_url(base_url), e)
        return None
    except Exception as e:
        logger.exception("Ocorreu um erro inesperado ao obter dados on-chain: %s", e)
        return None


def get_implied_volatility_history(index_name: str = "BTC_DVOL", resolution: str = "1D", start_timestamp_ms: int = None, end_timestamp_ms: int = None, limit: int = 1000, deribit_base_url: Optional[str] = None) -> list:
    """Busca o histórico de volatilidade implícita do índice Deribit (ex: BTC_DVOL).

    Retorna uma lista de dicionários com chaves 'timestamp' (ms) e 'volatility'. Mantém assinatura
    simples para não quebrar chamadas existentes.
    """
    endpoint = "/public/get_volatility_index_data"
    # Deribit expects 'index_name' as the parameter name
    # Deribit requires a 'currency' parameter for many volatility endpoints (e.g. BTC_DVOL -> currency=BTC)
    currency = None
    if isinstance(index_name, str) and "_" in index_name:
        currency = index_name.split("_")[0]

    params = {
        "index_name": index_name,
        "resolution": resolution,
    }
    if currency:
        params["currency"] = currency
    if start_timestamp_ms:
        # Deribit API expects timestamps in milliseconds
        params["start_timestamp"] = int(start_timestamp_ms)
    if end_timestamp_ms:
        params["end_timestamp"] = int(end_timestamp_ms)

    if not deribit_base_url:
        raise ValueError("deribit_base_url must be provided to get_implied_volatility_history")
    base_url = deribit_base_url
    client = APIClient(base_url=base_url, timeout=10, max_retries=3, backoff_factor=1)
    try:
        resp = client.get(endpoint, params=params)
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            logger.error("Erro ao decodificar JSON da Deribit (IV): %s", resp.text)
            return []

        # payload format: {"jsonrpc":"2.0","result":{...}} or result directly
        result = payload.get("result") if isinstance(payload, dict) else payload
        if not result:
            logger.error("Resposta inesperada da Deribit para IV: %s", payload)
            return []

        # Deribit may return arrays of timestamps and values, a list of dicts, or a list-of-lists
        out = []
        # Case A: result contains 'data' as list of dicts
        if isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
            for item in result["data"]:
                # item can be a dict or a list/tuple
                if isinstance(item, dict):
                    ts = item.get("timestamp") or item.get("t")
                    vol = item.get("value") or item.get("volatility") or item.get("v")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    ts = item[0]
                    # many Deribit volatility series return [timestamp, open, high, low, close]
                    # so take the last element as the representative volatility value
                    vol = item[-1]
                else:
                    continue

                if ts is None or vol is None:
                    continue
                out.append({"timestamp": int(ts), "volatility": float(vol)})
            return out

        # Case B: result contains parallel arrays 'timestamps' and 'values' or 'values'
        if isinstance(result, dict) and ("timestamps" in result and "values" in result):
            ts_list = result.get("timestamps")
            val_list = result.get("values")
            for ts, val in zip(ts_list, val_list):
                out.append({"timestamp": int(ts), "volatility": float(val)})
            return out

        # Case C: result itself is a list of {timestamp, value}
        if isinstance(result, list):
            for item in result:
                ts = item.get("timestamp") or item.get("t")
                vol = item.get("value") or item.get("volatility") or item.get("v")
                if ts is None or vol is None:
                    continue
                out.append({"timestamp": int(ts), "volatility": float(vol)})
            return out

        # Fallback: unexpected format
        logger.error("Formato inesperado de resposta Deribit (IV): %s", payload)
        return []
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar à API Deribit (IV) endpoint %s: %s", client._build_url(endpoint), e)
        return []


def get_uniswap_pool_daily_data(
    pool_id: str = _DEFAULT_POLYGON_POOL_ID,
    start_timestamp_ms: int = None,
    end_timestamp_ms: int = None,
    limit: int = 1000,
    thegraph_base_url: str = None,
    thegraph_api_key: str = None,
    thegraph_subgraph_ids: dict = None,
    default_network: str = 'polygon'
) -> list:
    """Consulta o subgraph da Uniswap v3 (The Graph) para recuperar poolDayDatas diários.

    Retorna uma lista de dicionários com chaves: 'timestamp' (ms), 'volumeUSD' (float), 'tvlUSD' (float).
    """
    # Build final URL using base + API key + subgraph id
    subgraph_id = "5zvR82QoaXyYfDEKLZ9t6v9adgnpTXyYp8gSbxTGVENFV"
    api_key = thegraph_api_key or ''
    if not api_key:
        logger.error("THEGRAPH_API_KEY not configured; cannot query The Graph gateway.")
        return []
    api_key = str(api_key).strip()
    if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
        api_key = api_key[1:-1]
    api_key = api_key.strip()
    if not api_key:
        logger.error("THEGRAPH_API_KEY is empty after sanitization; cannot query The Graph gateway.")
        return []
    # Allow caller to inject mapping of network -> subgraph id. Fall back to module defaults.
    subgraph_map = thegraph_subgraph_ids or {
        "mainnet": "ELUcwgpm14LKPLrdduc6pTfS_LpC7xdM14iBC_19I70",
        "polygon": "3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm",
    }
    network = default_network
    subgraph_id = subgraph_map.get(network)
    if not subgraph_id:
        logger.error("No subgraph id configured for network '%s'", network)
        return []

    # Base gateway URL
    if not thegraph_base_url:
        raise ValueError("thegraph_base_url must be provided to get_uniswap_pool_daily_data")
    base = thegraph_base_url
    if not base.endswith('/'):
        base = base + '/'

    # Build final URL using the gateway path that includes the API key and subgraph id if needed
    # We'll prefer Authorization header for authentication; keep URL as base + api_key + '/subgraphs/id/' + subgraph_id
    final_url = f"{base}{api_key}/subgraphs/id/{subgraph_id}"

    # Prepare Authorization header (primary auth method)
    headers = { 'Authorization': f'Bearer {api_key}' }

    # Log masked endpoint (do not reveal API key)
    masked = f"{base}<API_KEY>/subgraphs/id/{subgraph_id}"
    logger.debug("Querying The Graph endpoint: %s (network=%s)", masked, network)

    url = final_url

    # poolDayData.date is a unix timestamp in seconds representing the day
    start_sec = None
    end_sec = None
    if start_timestamp_ms:
        start_sec = int(start_timestamp_ms // 1000)
    if end_timestamp_ms:
        end_sec = int(end_timestamp_ms // 1000)

    # GraphQL query
    where_clauses = [f'pool: "{pool_id.lower()}"']
    if start_sec:
        where_clauses.append(f'date_gte: {start_sec}')
    if end_sec:
        where_clauses.append(f'date_lte: {end_sec}')
    where = ", ".join(where_clauses)

    query = f"""
    {{
      poolDayDatas(where: {{ {where} }}, orderBy: date, orderDirection: asc, first: {limit}) {{
        date
        volumeUSD
        tvlUSD
      }}
    }}
    """

    try:
        client = APIClient(base_url=base, timeout=10, max_retries=3, backoff_factor=1)
        # Use session.post to benefit from mounted adapters/retries and send Authorization header
        resp = client.session.post(url, json={"query": query}, headers=headers, timeout=client.timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao conectar ao The Graph (Uniswap subgraph) url=%s: %s", base, e)
        return []
    except ValueError:
        logger.error("Resposta inválida do The Graph (não JSON): %s", resp.text if resp is not None else "<no response>")
        return []

    result = payload.get("data") if isinstance(payload, dict) else None
    if not result or "poolDayDatas" not in result:
        logger.error("Resposta inesperada do The Graph para poolDayDatas: %s", payload)
        return []

    out = []
    for item in result["poolDayDatas"]:
        try:
            ts_sec = int(item.get("date"))
            vol = float(item.get("volumeUSD") or 0.0)
            tvl = float(item.get("tvlUSD") or 0.0)
            out.append({"timestamp": int(ts_sec * 1000), "volumeUSD": vol, "tvlUSD": tvl})
        except Exception:
            continue

    return out
    

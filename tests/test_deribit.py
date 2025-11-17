import logging
import sys
import os
import time

# Adiciona 'src' ao caminho
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from defi_data_toolkit.data_collector import get_implied_volatility_history

logging.basicConfig(level=logging.INFO)

print("--- Teste Final Deribit (Com Correção) ---")

# Data de início: 1 de Novembro de 2024
START_TS = 1730419200000 

# A função agora calcula o end_timestamp e currency automaticamente,
# mas vamos passar apenas o essencial para testar a lógica interna.
data = get_implied_volatility_history(
    index_name="BTC_DVOL",
    deribit_base_url="https://www.deribit.com/api/v2",
    start_timestamp_ms=START_TS
)

if data:
    print(f"✅ SUCESSO! Recebidos {len(data)} registros.")
    print(f"Primeiro: {data[0]}")
    print(f"Último: {data[-1]}")
else:
    print("❌ FALHA: Nenhum dado retornado (verifique os logs acima).")
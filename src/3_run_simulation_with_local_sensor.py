import random
import time
import joblib
import numpy as np
import pandas as pd

# --- Configurações da Simulação ---
INTENSIDADES = ["Leve", "Moderada", "Forte", "Extrema"]
INTENSIDADE_MM_H = {
    "Leve": (0.1, 5.0),
    "Moderada": (5.0, 18.0),
    "Forte": (18.0, 40.0),
    "Extrema": (40.0, 80.0)
}
SIM_DURATION = 10  # Número de ciclos de simulação
SENSOR_COD_ESTACAO = "SP001"  # Exemplo de código de estação

# --- Função para simular evento de chuva local ---
def simular_evento_chuva():
    intensidade = random.choices(INTENSIDADES, weights=[0.5, 0.3, 0.15, 0.05])[0]
    mm_h = round(random.uniform(*INTENSIDADE_MM_H[intensidade]), 2)
    return intensidade, mm_h

# --- Função para carregar modelo treinado ---
def carregar_modelo(path_modelo):
    return joblib.load(path_modelo)

# --- Função para predição do modelo ML ---
def prever_risco_ml(modelo, acumulado_chuva_1_h_mm, cod_estacao):
    X = pd.DataFrame([{
        "acumulado_chuva_1_h_mm": acumulado_chuva_1_h_mm,
        "cod_estacao": cod_estacao
    }])
    pred = modelo.predict(X)[0]
    return int(pred)

# --- Função para decisão combinada ---
def determinar_risco_final(risco_ml, intensidade_local, mm_h_local):
    # Lógica simples: se intensidade local for "Forte" ou "Extrema", aumenta o risco em 1 nível (máx 2)
    ajuste = 0
    if intensidade_local == "Forte":
        ajuste = 1
    elif intensidade_local == "Extrema":
        ajuste = 2
    risco_final = min(risco_ml + ajuste, 2)
    return risco_final

# --- Função para exibir alerta ---
def exibir_alerta(risco_final):
    cores = {0: "Verde", 1: "Amarelo", 2: "Vermelho"}
    sons = {0: "Silêncio", 1: "Bip curto", 2: "Bip contínuo"}
    print(f"ALERTA: Nível de risco = {risco_final} ({cores[risco_final]}) | Alerta sonoro: {sons[risco_final]}")

# --- Loop principal de simulação ---
def main():
    print("=== FloodGuard - Simulação de Sensor Local (ESP32 em Python) ===")
    modelo = carregar_modelo("ml_model/cemaden_flood_risk_model_pipeline.joblib")
    for ciclo in range(SIM_DURATION):
        print(f"\n[Ciclo {ciclo+1}]")
        intensidade_local, mm_h_local = simular_evento_chuva()
        print(f"Chuva local: Intensidade = {intensidade_local} | Acumulado 1h = {mm_h_local} mm")
        risco_ml = prever_risco_ml(modelo, mm_h_local, SENSOR_COD_ESTACAO)
        print(f"Predição do modelo ML (dados locais simulados): Nível de risco = {risco_ml}")
        risco_final = determinar_risco_final(risco_ml, intensidade_local, mm_h_local)
        exibir_alerta(risco_final)
        time.sleep(1)  # Pausa para simular tempo real

import sys

def testar_simulador():
    print("=== Teste de Verificação do Simulador FloodGuard ===")
    modelo = carregar_modelo("ml_model/cemaden_flood_risk_model_pipeline.joblib")
    # Teste 1: Intensidade Leve (espera-se risco baixo)
    intensidade, mm_h = "Leve", 2.0
    risco_ml = prever_risco_ml(modelo, mm_h, SENSOR_COD_ESTACAO)
    risco_final = determinar_risco_final(risco_ml, intensidade, mm_h)
    print(f"Teste 1 - Intensidade: {intensidade}, mm/h: {mm_h}")
    print(f"Risco ML: {risco_ml}, Risco Final: {risco_final}")
    assert risco_final in [0, 1], "Risco final inesperado para chuva leve"

    # Teste 2: Intensidade Extrema (espera-se risco alto)
    intensidade, mm_h = "Extrema", 50.0
    risco_ml = prever_risco_ml(modelo, mm_h, SENSOR_COD_ESTACAO)
    risco_final = determinar_risco_final(risco_ml, intensidade, mm_h)
    print(f"Teste 2 - Intensidade: {intensidade}, mm/h: {mm_h}")
    print(f"Risco ML: {risco_ml}, Risco Final: {risco_final}")
    assert risco_final == 2, "Risco final deveria ser máximo para chuva extrema"

    print("Todos os testes passaram com sucesso.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        testar_simulador()
    else:
        main()

# FloodGuard: Sistema Inteligente de Monitoramento e Alerta de Enchentes

## 🏛️ Instituição
FIAP - Faculdade de Informática e Administração Paulista

## 👨‍🎓 Integrantes
- Bruno Castro - RM558359
- Hugo Mariano - RM560688
- Matheus Castro - RM559293

---

## 📜 Descrição do Projeto

O **FloodGuard** é um sistema distribuído de monitoramento inteligente que combina análise preditiva baseada em machine learning com simulação de sensoriamento IoT para detectar condições de risco de enchentes. Utiliza dados pluviométricos oficiais do CEMADEN, processados para treinar um modelo de ML capaz de prever riscos regionais, os quais são refinados por dados simulados de sensores locais (ESP32 emulado em Python) para gerar alertas mais contextualizados. O foco é entregar um MVP funcional para a Global Solution da FIAP, abordando o tema de mitigação de impactos de eventos naturais extremos.

---

## 🚀 Guia Rápido de Uso

### 1. **Configuração do Ambiente**

- Recomenda-se o uso de ambiente virtual Python.
- Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```

### 2. **Processamento dos Dados Oficiais**

- Os dados brutos do CEMADEN (Janeiro a Maio de 2025) estão em `/data`.
- Execute o script de processamento:
  ```bash
  python src/1_process_official_data.py
  ```
- O arquivo processado será salvo em `data/cemaden_official_processed_hourly.csv`.

### 3. **Análise Exploratória (EDA)**

- Abra o notebook `notebooks/EDA_Cemaden.ipynb` no Jupyter Notebook/Lab.
- O notebook utiliza os dados processados de `/data`.

### 4. **Treinamento do Modelo**

- Execute o script:
  ```bash
  python src/2_train_model.py
  ```
- O modelo treinado será salvo em `/ml_model/cemaden_flood_risk_model_pipeline.joblib`.

### 5. **Simulação com Sensor Local**

- Execute o script:
  ```bash
  python src/3_run_simulation_with_local_sensor.py
  ```
- O script utiliza o modelo treinado e simula leituras de sensores locais para gerar alertas.

---

## 📁 Estrutura do Repositório

```
/
├── src/                # Scripts Python principais
│   ├── 1_process_official_data.py
│   ├── 2_train_model.py
│   └── 3_run_simulation_with_local_sensor.py
├── data/               # Dados brutos e processados
│   ├── cemaden_SP_jan_25.csv
│   ├── cemaden_sp_fev_25.csv
│   ├── cemaden_sp_marco_25.csv
│   ├── cemaden_sp_abril_25.csv
│   ├── cemaden_sp_maio_25.csv
│   ├── cemaden_official_processed_hourly.csv
│   └── eventos_enchentes_sp_2025.csv
├── ml_model/           # Modelos treinados e relatórios
│   ├── cemaden_flood_risk_model_pipeline.joblib
│   └── model_validation_report.txt
├── notebooks/          # Notebooks Jupyter
│   ├── EDA_Cemaden.ipynb
│   └── Train_Evaluate_Models.ipynb
├── docs/               # Documentação e arquivos de apoio
│   ├── FloodGuard.pdf
├── requirements.txt
└── README.md
```

---

## 📚 Documentação e Referências

- Toda a documentação detalhada está em `/docs`:
  - Floofguard.pdf
- Os scripts Python e notebooks possuem comentários explicativos.

---

## 🧪 Testes e Validação

- O script de simulação (`3_run_simulation_with_local_sensor.py`) inclui testes de verificação do modelo.
- O notebook `Train_Evaluate_Models.ipynb` apresenta validação dos modelos treinados.

---

## 🗃 Histórico de Versões

- **v1.0.0 (06/06/2025):**
  - Estrutura reorganizada do projeto.
  - Scripts, dados, modelos e documentação organizados em pastas temáticas.
  - README.md atualizado com instruções completas.

---

## 📋 Licença

Este projeto segue o modelo educacional FIAP e está licenciado sob Creative Commons Attribution 4.0 International.

---

## 👣 Recomendações Finais

- Consulte sempre o README e a pasta `/docs` para entender o fluxo e as decisões do projeto.
- Para dúvidas ou sugestões, utilize os comentários nos scripts e notebooks.

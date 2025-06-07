import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Embora treinaremos com todos os dados, pode ser útil para consistência
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

print("Iniciando o script de treinamento do modelo...")

# Carregar os dados processados
print("Carregando dados de data/cemaden_official_processed_hourly.csv...")
try:
    df = pd.read_csv('data/cemaden_official_processed_hourly.csv')
    print(f"Dados carregados com sucesso. Formato: {df.shape}")
except FileNotFoundError:
    print("Erro: Arquivo data/cemaden_official_processed_hourly.csv não encontrado.")
    print("Certifique-se de que o script 1_process_official_data.py foi executado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")
    exit()

# Verificar se o DataFrame tem dados
if df.empty:
    print("Erro: O DataFrame carregado está vazio.")
    exit()

# Seleção de Features (X) e Target (y)
print("Preparando features (X) e target (y)...")
X = df[['acumulado_chuva_1_h_mm', 'cod_estacao']]
y = df['nivel_risco']
print(f"Formato de X: {X.shape}, Formato de y: {y.shape}")

# Definir o pré-processador para One-Hot Encoding da coluna 'cod_estacao'
# remainder='passthrough' mantém as outras colunas (acumulado_chuva_1_h_mm)
# sparse_output=False para compatibilidade com alguns estimadores ou para inspeção mais fácil
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['cod_estacao'])
    ],
    remainder='passthrough'
)
print("Pré-processador definido.")

# Criar o pipeline para Random Forest com pré-processamento
# Usaremos os mesmos hiperparâmetros do notebook para consistência
pipeline_rf_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])
print("Pipeline do Random Forest criado.")

# Treinar o modelo com todos os dados disponíveis
print("Treinando o pipeline do Random Forest com todos os dados...")
try:
    pipeline_rf_clf.fit(X, y)
    print("Pipeline treinado com sucesso.")
except Exception as e:
    print(f"Erro durante o treinamento do pipeline: {e}")
    exit()

# Salvar o pipeline treinado
model_dir = 'ml_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Diretório '{model_dir}' criado.")

model_filename = os.path.join(model_dir, 'cemaden_flood_risk_model_pipeline.joblib')
print(f"Salvando o pipeline treinado em {model_filename}...")
try:
    joblib.dump(pipeline_rf_clf, model_filename)
    print(f"Pipeline salvo com sucesso em {model_filename}")
except Exception as e:
    print(f"Erro ao salvar o pipeline: {e}")
    exit()

# --- Seção de Geração de Relatório de Validação ---
print("\nGerando relatório de validação...")

# Dividir dados para avaliação (não afeta o modelo principal já treinado com todos os dados)
# Usamos as mesmas features X e target y globais
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Criar e treinar um pipeline temporário para validação
# (poderíamos também carregar o modelo salvo e prever, mas treinar um novo garante que as métricas são de um modelo treinado em um subset)
print("Treinando um pipeline temporário para gerar métricas de validação...")
pipeline_val = Pipeline(steps=[
    ('preprocessor', preprocessor), # Reutiliza o preprocessor definido anteriormente
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

try:
    pipeline_val.fit(X_train_val, y_train_val)
    y_pred_val = pipeline_val.predict(X_test_val)

    # Gerar métricas
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    accuracy_val = accuracy_score(y_test_val, y_pred_val)
    conf_matrix_val = confusion_matrix(y_test_val, y_pred_val)
    class_report_val = classification_report(y_test_val, y_pred_val)

    # Preparar conteúdo do relatório
    report_content = f"Relatório de Validação do Modelo (Random Forest com Features de Localidade)\n"
    report_content += f"-----------------------------------------------------------------------\n"
    report_content += f"Data da Geração: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content += f"Modelo Avaliado em um Split de Teste (20% dos dados totais, estratificado)\n\n"
    report_content += f"Acurácia no conjunto de teste de validação: {accuracy_val:.4f}\n\n"
    report_content += f"Matriz de Confusão (Teste de Validação):\n{str(conf_matrix_val)}\n\n"
    report_content += f"Relatório de Classificação (Teste de Validação):\n{class_report_val}\n"

    # Salvar relatório
    report_filename = os.path.join(model_dir, 'model_validation_report.txt')
    with open(report_filename, 'w') as f:
        f.write(report_content)
    print(f"Relatório de validação salvo em {report_filename}")

except Exception as e:
    print(f"Erro ao gerar ou salvar o relatório de validação: {e}")

# --- Seção de Re-treinamento (Exemplo) ---
def retrain_model_with_new_data(existing_model_path, new_data_path, output_model_path):
    """
    Carrega um modelo existente, adiciona novos dados, re-treina e salva o modelo atualizado.
    Assume que new_data_path aponta para um CSV com o mesmo formato dos dados originais.
    """
    print(f"\nIniciando processo de re-treinamento...")
    print(f"Carregando modelo existente de: {existing_model_path}")
    try:
        pipeline_to_retrain = joblib.load(existing_model_path)
    except FileNotFoundError:
        print(f"Erro: Modelo existente não encontrado em {existing_model_path}. Abortando re-treinamento.")
        return
    except Exception as e:
        print(f"Erro ao carregar o modelo existente: {e}. Abortando re-treinamento.")
        return

    print(f"Carregando novos dados de: {new_data_path}")
    try:
        new_df = pd.read_csv(new_data_path)
        if new_df.empty:
            print("Aviso: Arquivo de novos dados está vazio. Nenhum re-treinamento será feito.")
            return
        # Aqui, idealmente, você também processaria os novos dados da mesma forma que os originais
        # (ex: aplicar os mesmos limiares para 'nivel_risco' se não estiver presente)
        # Para este exemplo, assumimos que new_df já está no formato esperado para X_new e y_new.
        X_new = new_df[['acumulado_chuva_1_h_mm', 'cod_estacao']]
        y_new = new_df['nivel_risco']
        print(f"Novos dados carregados. Formato: {new_df.shape}")
    except FileNotFoundError:
        print(f"Erro: Arquivo de novos dados não encontrado em {new_data_path}. Abortando re-treinamento.")
        return
    except Exception as e:
        print(f"Erro ao carregar ou processar novos dados: {e}. Abortando re-treinamento.")
        return

    # Para re-treinar, precisamos dos dados originais com os quais o pipeline foi treinado
    # ou treinar do zero com dados combinados.
    # A abordagem mais simples para este exemplo é treinar um NOVO pipeline com os dados combinados.
    # Se o objetivo fosse ATUALIZAR o modelo existente de forma incremental, seria mais complexo
    # e dependeria do modelo suportar 'warm_start' ou similar, e o pré-processador também.

    print("Combinando dados existentes (usados no último treino completo) com novos dados...")
    # Recarregando os dados originais para combinar
    # NOTA: Esta é uma simplificação. Em um cenário real, você gerenciaria o dataset de treino de forma mais robusta.
    try:
        original_df_for_retrain = pd.read_csv('data/cemaden_official_processed_hourly.csv')
        combined_df = pd.concat([original_df_for_retrain, new_df], ignore_index=True)
        
        X_combined = combined_df[['acumulado_chuva_1_h_mm', 'cod_estacao']]
        y_combined = combined_df['nivel_risco']
        print(f"Dados combinados. Formato total: {combined_df.shape}")
    except Exception as e:
        print(f"Erro ao carregar ou combinar dados originais para re-treinamento: {e}")
        return

    print(f"Re-treinando o pipeline com dados combinados...")
    # Reutiliza a definição do pipeline_rf_clf, mas treina com os dados combinados
    # É importante que o preprocessor seja 'fitado' nos dados combinados também.
    pipeline_retrained = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['cod_estacao'])],
            remainder='passthrough')),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    try:
        pipeline_retrained.fit(X_combined, y_combined)
        print("Pipeline re-treinado com sucesso com dados combinados.")
    except Exception as e:
        print(f"Erro durante o re-treinamento do pipeline: {e}")
        return

    print(f"Salvando o pipeline re-treinado em: {output_model_path}")
    try:
        joblib.dump(pipeline_retrained, output_model_path)
        print(f"Pipeline re-treinado salvo com sucesso em {output_model_path}")
    except Exception as e:
        print(f"Erro ao salvar o pipeline re-treinado: {e}")

if __name__ == "__main__":
    print("\n--- Teste da Função de Re-treinamento ---")
    
    # Criar dados dummy para o teste de re-treinamento
    dummy_data_for_retrain = {
        'acumulado_chuva_1_h_mm': [10.0, 20.0, 3.0, 30.0],
        'cod_estacao': ['355030801A', '355030802A', '355030801A', '355030803A'], # Usar códigos de estação existentes ou novos
        'nivel_risco': [1, 2, 0, 2]
    }
    dummy_df = pd.DataFrame(dummy_data_for_retrain)
    
    dummy_data_dir = 'data'
    if not os.path.exists(dummy_data_dir):
        os.makedirs(dummy_data_dir)
        
    dummy_data_filename = os.path.join(dummy_data_dir, 'dummy_new_data_for_retrain.csv')
    
    try:
        dummy_df.to_csv(dummy_data_filename, index=False)
        print(f"Dados dummy para re-treinamento salvos em: {dummy_data_filename}")

        # Caminho para o modelo principal treinado
        main_model_path = os.path.join(model_dir, 'cemaden_flood_risk_model_pipeline.joblib')
        
        # Caminho para o modelo re-treinado (com nome diferente para não sobrescrever o principal durante o teste)
        retrained_model_output_path = os.path.join(model_dir, 'cemaden_flood_risk_model_pipeline_retrained_dummy_test.joblib')

        if os.path.exists(main_model_path):
            retrain_model_with_new_data(
                existing_model_path=main_model_path, # Usa o modelo treinado com todos os dados
                new_data_path=dummy_data_filename,
                output_model_path=retrained_model_output_path
            )
            
            # Verificar se o modelo re-treinado foi salvo
            if os.path.exists(retrained_model_output_path):
                print(f"Teste de re-treinamento: Modelo re-treinado salvo em {retrained_model_output_path}")
            else:
                print(f"Teste de re-treinamento: Falha ao salvar o modelo re-treinado.")
        else:
            print(f"Modelo principal {main_model_path} não encontrado. Teste de re-treinamento não pode ser executado.")

    except Exception as e:
        print(f"Erro durante o teste de re-treinamento: {e}")
    finally:
        # Limpar o arquivo dummy
        if os.path.exists(dummy_data_filename):
            try:
                os.remove(dummy_data_filename)
                print(f"Arquivo dummy {dummy_data_filename} removido.")
            except Exception as e:
                print(f"Erro ao remover arquivo dummy {dummy_data_filename}: {e}")
    
    print("\n--- Fim do Teste de Re-treinamento ---")

print("\nScript de treinamento principal, validação e teste de re-treinamento concluído.")

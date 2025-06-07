import pandas as pd
import glob
import os

def processar_dados_cemaden_oficiais():
    """
    Lê os arquivos CSV mensais do CEMADEN, unifica, padroniza colunas,
    agrega para dados horários e salva o resultado.
    """
    # Ajustar o padrão se os nomes dos arquivos variarem muito ou estiverem em outra pasta
    arquivos_mensais = [
        "data/cemaden_SP_jan_25.csv",  # Nome diferente (SP maiúsculo)
        "data/cemaden_sp_fev_25.csv",
        "data/cemaden_sp_marco_25.csv",
        "data/cemaden_sp_abril_25.csv",
        "data/cemaden_sp_maio_25.csv"
    ]
    
    # Verificar se os arquivos existem
    arquivos_encontrados = []
    for f_nome in arquivos_mensais:
        if os.path.exists(f_nome):
            arquivos_encontrados.append(f_nome)
        else:
            print(f"Aviso: Arquivo {f_nome} não encontrado e será ignorado.")

    if not arquivos_encontrados:
        print("Nenhum arquivo de dados mensal encontrado. Encerrando o script.")
        return

    print(f"Arquivos encontrados para processamento: {arquivos_encontrados}")

    lista_dfs_mensais = []

    for arquivo_csv in arquivos_encontrados:
        try:
            # Usar sep=';' e decimal=',' diretamente, pois foi o que funcionou.
            df_mes = pd.read_csv(arquivo_csv, encoding='utf-8', sep=';', decimal=',', engine='python')
            print(f"Lido {arquivo_csv} com sep=';' e decimal=',' ({len(df_mes)} linhas).")
            
            # Renomear colunas aqui para inspecionar 'chuva_10min_mm' com nome padronizado
            mapa_renomear_leitura = {
                'municipio': 'municipio', 'codEstacao': 'cod_estacao', 'uf': 'uf',
                'nomeEstacao': 'nome_estacao', 'latitude': 'latitude', 'longitude': 'longitude',
                'datahora': 'datahora_utc', 'valorMedida': 'chuva_10min_mm'
            }
            colunas_existentes_leitura = {k: v for k, v in mapa_renomear_leitura.items() if k in df_mes.columns}
            df_mes.rename(columns=colunas_existentes_leitura, inplace=True)

            if 'datahora_utc' in df_mes.columns and 'chuva_10min_mm' in df_mes.columns:
                print("Amostra das colunas 'datahora_utc' e 'chuva_10min_mm' após leitura e renomeação:")
                print(df_mes[['datahora_utc', 'chuva_10min_mm']].head())
            else:
                print("Aviso: Colunas 'datahora_utc' ou 'chuva_10min_mm' não encontradas após renomeação preliminar.")
                print(f"Colunas disponíveis no df_mes: {df_mes.columns.tolist()}")

            lista_dfs_mensais.append(df_mes)

        except Exception as e:
            print(f"Erro ao ler o arquivo {arquivo_csv} com sep=';' e decimal=',': {e}. Pulando este arquivo.")
            # Se quiser manter uma segunda tentativa com r'\s+', pode adicionar aqui,
            # mas como ';' funcionou, vamos simplificar por enquanto.
            # Exemplo de raw string para regex: sep=r'\s+'
            pass # Pula o arquivo se a leitura principal falhar


    if not lista_dfs_mensais:
        print("Nenhum DataFrame mensal foi carregado. Encerrando.")
        return

    df_completo = pd.concat(lista_dfs_mensais, ignore_index=True)
    print(f"Total de {len(df_completo)} linhas após concatenação.")

    # Lidar com possível BOM (Byte Order Mark) na primeira coluna
    if df_completo.columns[0].startswith('\ufeff'):
        print(f"Detectado BOM na primeira coluna: {df_completo.columns[0]}")
        df_completo.rename(columns={df_completo.columns[0]: df_completo.columns[0].replace('\ufeff', '')}, inplace=True)
        print(f"Primeira coluna renomeada para: {df_completo.columns[0]}")

    # Padronização de nomes de colunas já foi feita parcialmente dentro do loop de leitura para inspeção.
    # Aqui, garantimos que todas as colunas esperadas no df_completo tenham os nomes corretos,
    # caso alguma não tenha sido renomeada no loop (ex: se a leitura falhou e pulou a renomeação interna).
    mapa_renomear_final = {
        # 'municipio' já deve estar correto devido ao tratamento de BOM e renomeação no loop
        'codEstacao': 'cod_estacao', 'uf': 'uf', 'nomeEstacao': 'nome_estacao',
        'latitude': 'latitude', 'longitude': 'longitude',
        'datahora': 'datahora_utc', 'valorMedida': 'chuva_10min_mm'
    }
    # Aplicar renomeação para colunas que ainda podem ter nomes antigos
    colunas_para_renomear_final = {k: v for k, v in mapa_renomear_final.items() if k in df_completo.columns and k !=v}
    if colunas_para_renomear_final:
         df_completo.rename(columns=colunas_para_renomear_final, inplace=True)

    print(f"Colunas após concatenação e renomeação final: {df_completo.columns.tolist()}")

    # Verificar se as colunas essenciais existem
    colunas_essenciais = ['datahora_utc', 'chuva_10min_mm', 'cod_estacao']
    for col in colunas_essenciais:
        if col not in df_completo.columns:
            print(f"Erro: Coluna essencial '{col}' não encontrada após renomeação. Verifique os nomes das colunas nos arquivos CSV.")
            print(f"Colunas disponíveis: {df_completo.columns.tolist()}")
            return

    # Conversão de tipos
    df_completo['datahora_utc'] = pd.to_datetime(df_completo['datahora_utc'])
    df_completo['chuva_10min_mm'] = pd.to_numeric(df_completo['chuva_10min_mm'], errors='coerce').fillna(0)

    # Criar datahora_brasilia (UTC-3)
    df_completo['datahora_brasilia'] = df_completo['datahora_utc'] - pd.Timedelta(hours=3)

    # Agregação para dados horários
    # Arredondar datahora_utc para a hora cheia para agrupar
    df_completo['hora_utc_agrupada'] = df_completo['datahora_utc'].dt.floor('h') # Corrigido de 'H' para 'h'

    df_horario = df_completo.groupby(['cod_estacao', 'hora_utc_agrupada']).agg(
        acumulado_chuva_1_h_mm=('chuva_10min_mm', 'sum'),
        # Preservar outras informações da primeira ocorrência na hora (ou da mais relevante)
        municipio=('municipio', 'first'),
        uf=('uf', 'first'),
        nome_estacao=('nome_estacao', 'first'),
        latitude=('latitude', 'first'),
        longitude=('longitude', 'first'),
        datahora_brasilia_ref=('datahora_brasilia', 'first') # Referência para features temporais
    ).reset_index()
    
    df_horario.rename(columns={'hora_utc_agrupada': 'datahora_utc_hora'}, inplace=True)

    # Adicionar features temporais baseadas em datahora_brasilia_ref (que é a primeira ocorrência na hora UTC)
    # Para features como 'hora', usar a hora de Brasília correspondente ao início da janela horária UTC
    # Se datahora_brasilia_ref não existir (caso 'municipio', etc. não estejam nos CSVs), usar datahora_utc_hora
    if 'datahora_brasilia_ref' in df_horario.columns:
        ref_dt_col = df_horario['datahora_brasilia_ref']
    else: # Fallback se colunas como municipio não existirem e datahora_brasilia_ref não for criada
        df_horario['datahora_brasilia_ref_fallback'] = df_horario['datahora_utc_hora'] - pd.Timedelta(hours=3)
        ref_dt_col = df_horario['datahora_brasilia_ref_fallback']

    df_horario['ano'] = ref_dt_col.dt.year
    df_horario['mes'] = ref_dt_col.dt.month
    df_horario['dia'] = ref_dt_col.dt.day
    df_horario['hora_brasilia'] = ref_dt_col.dt.hour
    df_horario['dia_semana_brasilia'] = ref_dt_col.dt.dayofweek # Segunda=0, Domingo=6

    # Adicionar coluna nivel_risco com base em limiares ajustados após EDA (3ª rodada)
    # Baixo=0: <5.5mm/h; Moderado=1: 5.5mm <= chuva < 18mm/h; Alto=2: chuva >= 18mm/h
    limiares_risco = {
        'baixo_max': 5.5,  # Chuva < 5.5mm/h é Baixo Risco
        'moderado_max': 18  # Chuva >= 5.5mm/h e < 18mm/h é Moderado Risco
                            # Chuva >= 18mm/h é Alto Risco
    }

    def classificar_risco(chuva_h):
        if chuva_h < limiares_risco['baixo_max']:
            return 0 # Baixo
        elif chuva_h < limiares_risco['moderado_max']:
            return 1 # Moderado
        else:
            return 2 # Alto

    df_horario['nivel_risco'] = df_horario['acumulado_chuva_1_h_mm'].apply(classificar_risco)
    print(f"Coluna 'nivel_risco' adicionada com limiares ajustados (3ª rodada): Baixo (<{limiares_risco['baixo_max']}mm), Moderado (<{limiares_risco['moderado_max']}mm), Alto (>= {limiares_risco['moderado_max']}mm).")
    print(df_horario['nivel_risco'].value_counts(normalize=True).sort_index().map('{:.2%}'.format))


    print(f"Total de {len(df_horario)} linhas após agregação horária e adição de nivel_risco.")

    # Salvar o arquivo processado
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "cemaden_official_processed_hourly.csv")
    df_horario.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dados processados e agregados salvos em: {output_path}")

if __name__ == "__main__":
    processar_dados_cemaden_oficiais()
    print("Processamento concluído.")

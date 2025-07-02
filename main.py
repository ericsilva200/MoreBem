import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Função de Limpeza de Dados
def clean_numeric_data(df):
    """
    Limpa os campos numéricos do dataframe, substituindo valores inválidos por None.
    """
    try:
        cleaned_df = df.copy()
        
        def extract_number(value):
            if pd.isna(value) or value == '':
                return None
            
            num_str = re.sub(r'[^\d,.]', '', str(value))
            num_str = num_str.replace(',', '.')
            
            if '.' in num_str and ',' in num_str:
                num_str = num_str.replace('.', '')
            
            try:
                return float(num_str) if num_str else None
            except (ValueError, TypeError):
                return None
        
        # Limpeza dos campos numéricos
        for col in ['CONDO', 'TAX', 'AREA', 'PRICE']:
            cleaned_df[col] = cleaned_df[col].apply(extract_number)
        
        # Campos inteiros
        for col in ['ROOMS_NO', 'BATH_NO', 'PARKING_SPOTS']:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df[col] = cleaned_df[col].apply(lambda x: int(x) if pd.notnull(x) else None)
        
        return cleaned_df
    
    except Exception as e:
        print(f"Erro na limpeza de dados: {str(e)}")
        raise

# 2. Classificação por Tamanho
def classificar_tamanho(area):
    try:
        if area is None:
            return 'Não informado'
        
        area = float(area)
        if area < 20 or area > 500:
            return 'Área inválida'
        if 20 <= area <= 60:
            return 'Pequeno'
        elif 60 < area <= 90:
            return 'Médio'
        else:
            return 'Grande'
    except (TypeError, ValueError) as e:
        print(f"Erro ao classificar tamanho: {str(e)}")
        return 'Não informado'

# 3. Filtragem por Preferências
def filtrar_imoveis(df, preferencias):
    try:
        df_filtrado = df.copy()
        
        # Filtros
        if preferencias.get('bairro'):
            df_filtrado = df_filtrado[
                df_filtrado['NEIGHBORHOOD'].str.contains(
                    preferencias['bairro'], case=False, na=False)]
        
        if preferencias.get('quartos_min'):
            df_filtrado = df_filtrado[
                (df_filtrado['ROOMS_NO'] >= preferencias['quartos_min'])]
        
        if preferencias.get('banheiros_min'):
            df_filtrado = df_filtrado[
                (df_filtrado['BATH_NO'] >= preferencias['banheiros_min'])]
        
        if preferencias.get('vagas_min'):
            df_filtrado = df_filtrado[
                (df_filtrado['PARKING_SPOTS'] >= preferencias['vagas_min'])]
        
        if preferencias.get('tamanho'):
            if preferencias['tamanho'] == 'Pequeno':
                df_filtrado = df_filtrado[(df_filtrado['AREA'] <= 60)]
            elif preferencias['tamanho'] == 'Médio':
                df_filtrado = df_filtrado[
                    (df_filtrado['AREA'] > 60) & (df_filtrado['AREA'] <= 90)]
            elif preferencias['tamanho'] == 'Grande':
                df_filtrado = df_filtrado[(df_filtrado['AREA'] > 90)]
        
        if preferencias.get('preco_max'):
            df_filtrado = df_filtrado[
                (df_filtrado['PRICE'] <= preferencias['preco_max']) & 
                (df_filtrado['PRICE'] >= 50000)]
        
        if preferencias.get('condominio_max'):
            df_filtrado = df_filtrado[
                (df_filtrado['CONDO'] <= preferencias['condominio_max'])]
        
        if preferencias.get('iptu_max'):
            df_filtrado = df_filtrado[
                (df_filtrado['TAX'] <= preferencias['iptu_max'])]
        
        return df_filtrado.reset_index(drop=True)
    
    except Exception as e:
        print(f"Erro na filtragem de imóveis: {str(e)}")
        raise

# 4. Regressão e Classificação
def analisar_oportunidades(df):
    """
    Utiliza a regressão para encontrar o valor predito dos imóveis com base na sua área, quartos e banheiros.
    
    Após a regressão é utilizada a classificação para classificar os imóveis em bons negócios ou não com base no percentual da diferençã do preço original com o preço predito.
    """
    try:
        # Preparar dados para regressão
        df_reg = df.dropna(subset=['AREA', 'ROOMS_NO', 'BATH_NO', 'PRICE'])
        
        if len(df_reg) < 10:
            raise ValueError("Dados insuficientes para modelagem")
        
        X = df_reg[['AREA', 'ROOMS_NO', 'BATH_NO']]
        y = df_reg['PRICE']
        
        # Modelagem
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Adicionar preço predito
        df['PRECO_PREDITO'] = model.predict(df[['AREA', 'ROOMS_NO', 'BATH_NO']])
        
        # Calcular diferença percentual
        df['DIFERENCA_PERCENTUAL'] = (
            (df['PRECO_PREDITO'] - df['PRICE']) / df['PRECO_PREDITO'] * 100)
        
        # Classificar recomendações
        df['RECOMENDACAO'] = np.where(df['DIFERENCA_PERCENTUAL'] >= 10, 1, 0)
        
        return df.sort_values('DIFERENCA_PERCENTUAL', ascending=False)
    
    except Exception as e:
        print(f"Erro na análise de oportunidades: {str(e)}")
        raise

# 5. Clusterização e Visualização
def clusterizar_imoveis(df):
    try:
        # Preparar dados
        df_cluster = df.dropna(subset=['AREA', 'ROOMS_NO', 'BATH_NO', 'PRICE', 'PRECO_PREDITO'])
        
        if len(df_cluster) < 3:
            raise ValueError("Dados insuficientes para clusterização")
        
        # Feature engineering
        df_cluster['PRECO_M2'] = df_cluster['PRICE'] / df_cluster['AREA']
        df_cluster['DIFERENCA_ABSOLUTA'] = df_cluster['PRECO_PREDITO'] - df_cluster['PRICE']
        
        # Clusterização
        features = ['AREA', 'ROOMS_NO', 'BATH_NO', 'PRECO_M2', 'DIFERENCA_ABSOLUTA']
        X = StandardScaler().fit_transform(df_cluster[features])
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_cluster['CLUSTER'] = kmeans.fit_predict(X)
        
        # Nomear clusters
        cluster_names = {
            0: 'Imóveis padrão',
            1: 'Imóveis espaçosos e de maior investimento', 
            2: 'Imóveis compactos e econômicos'
        }
        df_cluster['CLUSTER_NAME'] = df_cluster['CLUSTER'].map(cluster_names)
        
        # Visualização
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_cluster, x='AREA', y='PRICE', 
                        hue='CLUSTER_NAME', style='RECOMENDACAO',
                        palette='viridis', size='DIFERENCA_PERCENTUAL')
        plt.title('Distribuição de Imóveis por Cluster')
        plt.xlabel('Área (m²)')
        plt.ylabel('Preço (R$)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return df_cluster
    
    except Exception as e:
        print(f"Erro na clusterização: {str(e)}")
        raise

# 6. Exibição de Resultados
def mostrar_resultados(df_cluster):
    try:
        print("\n💎 Melhores Negócios por Cluster 💎\n")
        
        cluster_names = df_cluster['CLUSTER_NAME'].unique()
        
        for name in cluster_names:
            cluster_df = df_cluster[df_cluster['CLUSTER_NAME'] == name]
            print(f"\n🏘️ Cluster {name} (Total: {len(cluster_df)}, Bons Negócios: {sum(cluster_df['RECOMENDACAO'])})")
            print("-"*50)
            
            # Mostrar melhores negócios ou os mais próximos
            deals = cluster_df[cluster_df['RECOMENDACAO'] == 1].sort_values(
                'DIFERENCA_PERCENTUAL', ascending=False).head(3)
            
            if deals.empty:
                deals = cluster_df.sort_values('DIFERENCA_PERCENTUAL', ascending=False).head(3)
                print("⚠️ Nenhum 'Bom Negócio' encontrado. Mostrando os melhores disponíveis:")
            
            for _, row in deals.iterrows():
                print(f"📍 {row['NEIGHBORHOOD']}")
                print(f"📏 Área: {row['AREA']:.0f}m² | 🛏️ Quartos: {row['ROOMS_NO']} | 🚽 Banheiros: {row['BATH_NO']}")
                print(f"💰 Preço: R${row['PRICE']:,.2f} | Predito: R${row['PRECO_PREDITO']:,.2f}")
                print(f"📉 Economia: {row['DIFERENCA_PERCENTUAL']:.2f}%")
                print(f"🏷️ Condomínio: R${row['CONDO'] if pd.notnull(row['CONDO']) else 'N/A'} | IPTU: R${row['TAX'] if pd.notnull(row['TAX']) else 'N/A'}")
                print("-"*50)
    
    except Exception as e:
        print(f"Erro ao exibir resultados: {str(e)}")
        raise

# Função Principal
def main():
    try:
        # 1. Carregar dados
        print("🚀 Iniciando processo de recomendação de imóveis...")
        df = pd.read_csv('/content/sample_data/final_dataframe.csv')
        
        # 2. Limpeza dos dados
        print("🧹 Limpando dados...")
        df_limpo = clean_numeric_data(df)
        df_limpo['Tamanho'] = df_limpo['AREA'].apply(classificar_tamanho)
        
        # 3. Exemplo de preferências do usuário
        preferencias = {
            'bairro': 'Barreiro',
            'quartos_min': 2,
            'banheiros_min': 1,
            'vagas_min': 1,
            'tamanho': 'Médio',
            'preco_max': 300000,
            'condominio_max': 400,
            'iptu_max': 200
        }
        
        # 4. Filtrar imóveis
        print("🔎 Filtrando imóveis pelas preferências...")
        df_filtrado = filtrar_imoveis(df_limpo, preferencias)
        print(f"✅ Encontrados {len(df_filtrado)} imóveis que atendem aos critérios")
        
        # 5. Analisar oportunidades
        print("📊 Analisando oportunidades...")
        df_recomendacao = analisar_oportunidades(df_filtrado)
        
        # 6. Clusterizar resultados
        print("🧩 Clusterizando imóveis...")
        df_cluster = clusterizar_imoveis(df_recomendacao)
        
        # 7. Mostrar resultados
        mostrar_resultados(df_cluster)
        print("\n🎉 Processo concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro no processo principal: {str(e)}")
        return None

# Executar o pipeline completo
if __name__ == "__main__":
    main()
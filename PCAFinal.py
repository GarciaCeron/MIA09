import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(page_title="Análisis Avanzado de Cáncer de Mama", page_icon="breastCancer.ico", layout="wide")
st.title("Análisis Avanzado de Cáncer de Mama con Clustering")


# Cargar datos
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target_name'] = df['target'].map({0: 'Maligno', 1: 'Benigno'})
    return df, data

df, data = load_data()

# Mostrar DataFrame original
st.header("1. DataFrame Original")
st.dataframe(df.head())

# Escalar datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[data.feature_names])
df_scaled = pd.DataFrame(scaled_data, columns=data.feature_names)

# Mostrar DataFrame escalado
st.header("2. DataFrame Escalado")
st.dataframe(df_scaled.head())

# Gráficos combinados y de dispersión
st.header("3. Visualizaciones antes de Clusterización")

# Selección de características
col1, col2 = st.columns(2)
with col1:
    feature_x = st.selectbox("Selecciona característica para eje X", data.feature_names, index=0)
with col2:
    feature_y = st.selectbox("Selecciona característica para eje Y", data.feature_names, index=1)

# Gráfico combinado de barras y dispersión
st.subheader("Gráfico Combinado (Barras y Dispersión)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='target_name', y=feature_x, ax=ax, alpha=0.6, palette=['salmon', 'lightblue'])
ax2 = ax.twinx()
sns.scatterplot(data=df, x='target_name', y=feature_y, ax=ax2, color='green', s=100)
plt.title(f"Distribución de {feature_x} (barras) vs {feature_y} (puntos)")
st.pyplot(fig)

# Gráficos de dispersión
st.subheader("Gráficos de Dispersión")

cols = st.columns(2)
with cols[0]:
    st.write("**Primeras dos características:**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=data.feature_names[0], y=data.feature_names[1], 
                    hue='target_name', palette=['salmon', 'lightblue'])
    st.pyplot(fig)

with cols[1]:
    st.write("**Últimas dos características:**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=data.feature_names[-2], y=data.feature_names[-1], 
                    hue='target_name', palette=['salmon', 'lightblue'])
    st.pyplot(fig)

# Método del codo para determinar k óptimo
st.header("4. Método del Codo para Determinar Número Óptimo de Clústers")

# Reducción de dimensionalidad con PCA para visualización
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Calcular WCSS para diferentes valores de k
wcss = []
max_clusters = 10
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Gráfico del método del codo
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='--')
ax.set_title('Método del Codo')
ax.set_xlabel('Número de clústers')
ax.set_ylabel('WCSS')  # Within-cluster sum of squares
ax.axvline(x=3, color='red', linestyle=':')
st.pyplot(fig)

# Aplicar K-Means con el número óptimo de clústers
st.header("5. Aplicación de K-Means y Visualización de Clústers")

optimal_k = st.slider("Selecciona el número de clústers", 2, 10, 3)

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df_pca['Cluster'] = clusters.astype(str)

# Visualización de clústers en espacio PCA
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', 
                palette=sns.color_palette("hls", optimal_k), s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', label='Centroides', marker='X')
plt.title(f'Clústers de Pacientes (k={optimal_k})')
plt.legend()
st.pyplot(fig)

# Interpretación de resultados
st.header("6. Interpretación de Resultados")
st.write(f"""
- **Método del codo:** La gráfica muestra que el punto de inflexión (codo) está alrededor de k=3
- **Clústeres:** Hemos dividido los pacientes en {optimal_k} grupos basados en similitud de características
- **Visualización PCA:** Hemos reducido las 30 dimensiones a 2 componentes principales para visualización
""")

# Sidebar con información
st.sidebar.header("Opciones Avanzadas")
show_full_data = st.sidebar.checkbox("Mostrar datos completos")
if show_full_data:
    st.dataframe(df)

st.sidebar.header("Acerca de")
st.sidebar.info("""
Esta aplicación realiza:
1. Análisis exploratorio de datos
2. Visualizaciones avanzadas
3. Clusterización con K-Means
4. Reducción de dimensionalidad con PCA
""")

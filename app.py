import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
@st.cache_data
def load_data():
    data = pd.read_csv("exams.csv")
    return data

# Préparer les données pour le clustering
def prepare_data(data):
    X = data.drop(columns=['gender'])
    X = X.iloc[:, [4, 5, 6]]
    X = np.array(X)
    return X

# Calculer la méthode Elbow
def calculate_elbow(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    return K, distortions

# Entraîner le modèle K-means
def train_kmeans(X, n_clusters=3):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_predict = kmeans_model.fit_predict(X)
    return kmeans_model, kmeans_predict

# Mapper les niveaux de performance
def map_performance(cluster_means):
    performance_labels = []
    for mean in cluster_means:
        if mean >= 80:
            performance_labels.append('Elevée')
        elif mean >= 60:
            performance_labels.append('Moyenne')
        else:
            performance_labels.append('Faible')
    return performance_labels

# Fonction principale pour l'application Streamlit
def main():
    st.set_page_config(page_title="Clustering K-means App", page_icon=":bar_chart:")

    st.title("Application de Clustering K-means")

    # Charger les données
    data = load_data()
    st.write("Aperçu des données:")
    st.write(data.head())

    # Préparer les données
    X = prepare_data(data)

    # Calculer la méthode Elbow
    st.header("Méthode Elbow pour déterminer le nombre optimal de clusters")
    K, distortions = calculate_elbow(X)
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Méthode Elbow montrant le nombre optimal de clusters')
    st.pyplot(plt)
    st.markdown("---")

    # Sélection du nombre de clusters
    n_clusters = st.slider("Sélectionner le nombre de clusters", 1, 10, 3)

    # Entraîner le modèle K-means
    kmeans_model, kmeans_predict = train_kmeans(X, n_clusters)

    # Ajouter les étiquettes de clusters au DataFrame original
    data['Cluster'] = kmeans_predict

    # Calculer les moyennes des scores pour chaque cluster
    cluster_means = data.groupby('Cluster')[['math score', 'reading score', 'writing score']].mean().mean(axis=1)

    # Mapper les niveaux de performance
    performance_labels = map_performance(cluster_means)
    performance_dict = {i: performance_labels[i] for i in range(len(performance_labels))}

    # Ajouter la colonne de performance
    data['Performance'] = data['Cluster'].map(performance_dict)

    # Afficher les centres de clusters
    st.write("Centres de clusters:")
    st.write(pd.DataFrame(kmeans_model.cluster_centers_, columns=['math score', 'reading score', 'writing score']))

    # Afficher le DataFrame avec les clusters et les niveaux de performance
    st.write("Données avec les clusters et les niveaux de performance:")
    st.write(data)

    # Calculer les métriques de clustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    silhouette_avg = silhouette_score(X, kmeans_predict)
    calinski_harabasz = calinski_harabasz_score(X, kmeans_predict)
    davies_bouldin = davies_bouldin_score(X, kmeans_predict)

    st.header("Métriques de clustering")
    st.write(f"Silhouette Score: {silhouette_avg}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin}")

    # Affichage du pourcentage de chaque niveau de performance dans le dataset
    performance_counts = data['Performance'].value_counts(normalize=True)
    st.header("Répartition des niveaux de performance dans le dataset")
    fig, ax = plt.subplots()
    ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Entrée des notes d'un étudiant dans la sidebar
    st.sidebar.title("Entrer les notes de l'étudiant")
    math_score = st.sidebar.number_input("Note en mathématiques", min_value=0, max_value=100, value=75)
    reading_score = st.sidebar.number_input("Note en lecture", min_value=0, max_value=100, value=75)
    writing_score = st.sidebar.number_input("Note en écriture", min_value=0, max_value=100, value=75)

    # Prédire le cluster de l'étudiant
    if st.sidebar.button("Prédire le cluster"):
        student_scores = np.array([[math_score, reading_score, writing_score]])
        student_cluster = kmeans_model.predict(student_scores)[0]
        student_performance = performance_dict[student_cluster]

        st.sidebar.write(f"L'étudiant appartient au cluster: {student_cluster}")
        st.sidebar.write(f"Niveau de performance du cluster: {student_performance}")

if __name__ == '__main__':
    main()

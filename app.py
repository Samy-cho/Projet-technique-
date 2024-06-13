import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
import streamlit as st
import gc

# Charger les datasetsx
@st.cache_data
def load_data():
    anime_df = pd.read_csv('/content/anime.csv')
    manga_df = pd.read_csv('/content/manga.csv')
    return anime_df, manga_df

# Préparer les données
def prepare_data(anime_df, manga_df):
    data = pd.concat([anime_df, manga_df], ignore_index=True)
    titles = data[['Title', 'Score']].copy()

    # Remplir les valeurs manquantes pour les colonnes numériques
    data[['Score', 'Vote', 'Ranked', 'Popularity']] = data[['Score', 'Vote', 'Ranked', 'Popularity']].fillna(0)

    # Remplir les valeurs manquantes pour les colonnes catégorielles
    categorical_columns = ['Genres', 'Status', 'Aired', 'Premiered', 'Producers', 'Licensors', 'Studios', 'Source', 'Duration', 'Rating', 'Published', 'Themes', 'Demographics', 'Serialization', 'Author']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')

    # Nettoyage des données : suppression des valeurs non numériques dans les colonnes numériques
    columns_to_clean = ['Score', 'Vote', 'Ranked', 'Popularity', 'Episodes', 'Members', 'Favorite', 'Volumes', 'Chapters']
    for col in columns_to_clean:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Encodage des colonnes catégorielles
    encoder = OneHotEncoder(sparse_output=False)
    for col in categorical_columns:
        if col in data.columns:
            encoded = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
            data = pd.concat([data, encoded_df], axis=1)

    # Retirer les colonnes non nécessaires ou catégorielles
    columns_to_remove = [col for col in categorical_columns if col in data.columns] + ['Title']
    data = data.drop(columns=columns_to_remove)

    # Normalisation des caractéristiques numériques
    scaler = StandardScaler()
    numeric_columns = ['Vote', 'Ranked', 'Popularity', 'Episodes', 'Members', 'Favorite', 'Volumes', 'Chapters']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data, titles, scaler, encoder

# Charger les données
anime_df, manga_df = load_data()
data, titles, scaler, encoder = prepare_data(anime_df, manga_df)

# Vérifier s'il reste des colonnes non numériques
non_numeric_cols = data.select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) > 0:
    print(f"Colonnes non numériques restantes: {non_numeric_cols}")
    raise ValueError(f"Colonnes non numériques trouvées: {non_numeric_cols}")

# Échantillonner une partie des données pour réduire l'utilisation de la mémoire
data_sample = data.sample(frac=0.05, random_state=42)  # Utiliser 5% des données pour le test

# Séparation des données pour le modèle de prédiction de succès
X = data_sample.drop(columns=['Score'])
y = data_sample['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Prétraitement terminé.")

# Entraînement du modèle de prédiction de succès
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prédiction et validation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Libérer de la mémoire
del X_train, X_test, y_train, y_test, y_pred
gc.collect()

# Préparation des données pour le modèle de recommandation
if 'Score' in titles.columns:
    # Utiliser l'index comme identifiant d'utilisateur simulé
    ratings_dict = {
        'item_id': titles['Title'],
        'user_id': titles.index,  # Utilisation de l'index comme identifiant d'utilisateur simulé
        'rating': titles['Score']
    }
    df = pd.DataFrame(ratings_dict)

    # Préparer les données pour surprise
    reader = Reader(rating_scale=(1, 10))
    data_surprise = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    # Entraînement du modèle de recommandation
    algo = KNNBasic(k=5, min_k=1, sim_options={'user_based': False})
    cross_validate(algo, data_surprise, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    trainset = data_surprise.build_full_trainset()
    algo.fit(trainset)
else:
    print("Les colonnes nécessaires pour la recommandation ne sont pas présentes.")

# Interface utilisateur avec Streamlit
st.title('Anime/Manga Success Prediction and Recommendation System')

tabs = st.tabs(["Prédiction du Succès", "Recommandation"])

with tabs[0]:
    st.header('Prédiction du Succès')
    st.write(f'Mean Squared Error for Success Prediction: {mse}')

    # Input fields for the prediction (vous devez ajouter les champs d'entrée correspondant à vos caractéristiques)
    st.write("Entrez les détails pour prédire le succès :")
    vote = st.number_input('Vote', min_value=0, step=1)
    ranked = st.number_input('Ranked', min_value=0, step=1)
    popularity = st.number_input('Popularity', min_value=0, step=1)
    episodes = st.number_input('Episodes', min_value=0, step=1)
    members = st.number_input('Members', min_value=0, step=1)
    favorite = st.number_input('Favorite', min_value=0, step=1)
    volumes = st.number_input('Volumes', min_value=0, step=1)
    chapters = st.number_input('Chapters', min_value=0, step=1)

    if st.button('Prédire'):
        input_data = pd.DataFrame({
            'Vote': [vote],
            'Ranked': [ranked],
            'Popularity': [popularity],
            'Episodes': [episodes],
            'Members': [members],
            'Favorite': [favorite],
            'Volumes': [volumes],
            'Chapters': [chapters]
        })

        # Appliquer l'encodage OneHotEncoding sur les données d'entrée
        for col in encoder.categories_:
            if col in input_data.columns:
                encoded = encoder.transform(input_data[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                input_data = pd.concat([input_data, encoded_df], axis=1)
                input_data = input_data.drop(columns=[col])

        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        st.write(f'Score Prédit: {prediction[0]}')

with tabs[1]:
    st.header('Recommandation')
    selected_title = st.selectbox('Select an anime/manga you like:', titles['Title'].unique())
    if st.button('Recommend'):
        inner_id = algo.trainset.to_inner_iid(selected_title)
        neighbors = algo.get_neighbors(inner_id, k=5)
        recommendations = [algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors]
        st.write('Recommendations:')
        for rec in recommendations:
            st.write(rec)

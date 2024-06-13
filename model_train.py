import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import gc  # Garbage Collector to free up memory

# Charger les datasets
anime_df = pd.read_csv('/content/anime.csv')
manga_df = pd.read_csv('/content/manga.csv')

# Fusionner les datasets
data = pd.concat([anime_df, manga_df], ignore_index=True)

# Sauvegarder la colonne 'Title' et 'Score' pour le modèle de recommandation
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
encoded_cols = []
for col in categorical_columns:
    if col in data.columns:
        encoded = encoder.fit_transform(data[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
        data = pd.concat([data, encoded_df], axis=1)
        encoded_cols.extend(encoder.get_feature_names_out([col]))

# Retirer les colonnes non nécessaires ou catégorielles
columns_to_remove = [col for col in categorical_columns if col in data.columns] + ['Title']
data = data.drop(columns=columns_to_remove)

# Normalisation des caractéristiques numériques
scaler = StandardScaler()
numeric_columns = ['Vote', 'Ranked', 'Popularity', 'Episodes', 'Members', 'Favorite', 'Volumes', 'Chapters']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Vérifier s'il reste des colonnes non numériques
non_numeric_cols = data.select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) > 0:
    print(f"Colonnes non numériques restantes: {non_numeric_cols}")
    raise ValueError(f"Colonnes non numériques trouvées: {non_numeric_cols}")

# Échantillonner une partie des données pour réduire l'utilisation de la mémoire
data_sample = data.sample(frac=0.1, random_state=42)  # Utiliser 10% des données pour le test

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
    algo = SVD()
    cross_validate(algo, data_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)
else:
    print("Les colonnes nécessaires pour la recommandation ne sont pas présentes.")


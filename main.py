import pandas as pd
from sklearn.decomposition import TruncatedSVD
from datetime import datetime, date
import streamlit as st


df= pd.read_csv("datasetreco.csv", sep=";")

def rating(df):
    coef_view,coef_cart,coef_purchase= 1, 5, 10
    df["rating"]=df["p_views"]*coef_view + df["p_carts"]*coef_cart+df["p_purchases"]*coef_purchase
    return df

df=rating(df)

df_group= df[df["p_purchases"]>= 1]
df_reduct= df_group[["user_id","product_id","subsubcategory","rating"]]
df_reduct= df_reduct.drop_duplicates()
max_ratings = df_reduct.groupby('subsubcategory')['rating'].transform('max')
df_reduct['normalized_rating'] = df_reduct['rating'] / max_ratings

df_pivot = df_reduct.pivot(index='product_id', columns='user_id', values='rating')
df_pivot= df_pivot.fillna(0)

dimention=10
svd=TruncatedSVD(n_components=dimention)
matrix_pivot_trans=svd.fit_transform(df_pivot)

matrix_pivot_trans
matrix_pivot_trans.shape
import numpy as np
correlation=np.corrcoef(matrix_pivot_trans)

df_pivot_trans=pd.DataFrame(correlation,index=df_pivot.index,columns=df_pivot.index)

def recom(id_product, nbr_index):
    if id_product not in df_pivot_trans.index:
        print("Le produit demandé n'existe pas dans la matrice de corrélation")
    else:
        product_correlations = df_pivot_trans.loc[id_product]
        most_similar_products = product_correlations.sort_values(ascending=False).drop(id_product)
        top_similar=most_similar_products[:nbr_index]
    return top_similar.index.tolist()

df_detail=df[["product_id","price","category","subcategory","subsubcategory"]]
df_detail.set_index('product_id', inplace= True)
#df_detail= df_detail.drop(columns=df_detail["product_id"])
#df_detail.drop_duplicates(subset=['product_id'], inplace=True, keep='first')



df_detail_user=df[["user_id","product_id","Date","Time","price","category","subcategory","subsubcategory"]]
df_detail_user['datetime']= pd.to_datetime(df_detail_user["Date"]+' '+df_detail_user['Time'])
df_detail_user.drop(columns=["Date", "Time"], inplace=True)
df_detail_user.set_index("user_id", inplace=True)


def predict_by_user(user_id, nbr_index, nbr_hist):
    # Assurer que 'datetime' est au bon format
    df_detail_user['datetime'] = pd.to_datetime(df_detail_user['datetime'])
    
    # Filtrer les données pour l'utilisateur spécifique
    user_data = df_detail_user.loc[user_id]
    
    # Sélectionner les nbr_hist dernières consultations par date et heure
    user_data = user_data.nlargest(nbr_hist, 'datetime', keep='first')
    
    # Garder uniquement les premières occurrences de chaque produit
    user_data.drop_duplicates(inplace=True)
    #user_data.drop_duplicates(inplace=True)
    #print(user_data)
    # Créer un DataFrame pour les dernières consultations
    df_last_nbr_hist = user_data[["product_id", "datetime"]]
    df_last_nbr_hist.set_index("product_id", inplace=True)
    
    #print(df_last_nbr_hist.head())
    
    # Initialiser une liste pour stocker les recommandations
    recommandations = []
    
    # Boucle pour obtenir les recommandations pour chaque produit dans l'historique
    for product_id in df_last_nbr_hist.index.to_list():
        recommandations=recommandations + recom(product_id,nbr_index)
        #print(recommandations)
    recommandations = list(set(recommandations))
    df_detail_recom = df_detail.loc[recommandations]
    df_detail_recom.drop_duplicates(inplace=True)
    # Filtrer les détails des produits recommandés
    
    
    return df_detail_recom


# Charger les données
@st.cache
def load_data():
    df = pd.read_csv("datasetreco.csv", sep=";")
    
    return df

# Fonction de recommandation
def recommandation(df, user_id, nbr_index, nbr_hist):
    # Votre code de recommandation ici
    return df

# Chargement des données
df = load_data()

# Titre de l'application
st.title("Système de Recommandation")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres")
user_id = st.sidebar.text_input("ID Utilisateur", value="123")
nbr_index = st.sidebar.slider("Nombre de Recommandations", min_value=1, max_value=5, value=3)
nbr_hist = st.sidebar.slider("Nombre d'Historique", min_value=1, max_value=10, value=3)

# Affichage des recommandations
st.subheader("Produits Recommandés")
if st.button("Rechercher"):
    recommendations = recommandation(df, user_id, nbr_index, nbr_hist)
    st.dataframe(predict_by_user)

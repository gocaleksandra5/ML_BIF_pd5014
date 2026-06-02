#To zainstalowałam w terminalu
#pip install tensorflow keras numpy pandas matplotlib seaborn umap-learn scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split

#Wczytanie pliku, tu nie wiedziałam czy mam udostępnić dane pliku csv z mojego github czy mam założyć z góry, że plik najduje się już u mnie na komputerz więc przedstawiam dwie wersie:  
#df = pd.read_csv("dane_projekt1.csv")
df = pd.read_csv("https://raw.githubusercontent.com/gocaleksandra5/ML_BIF_pd5014/main/ML_BIF_pd5014_ML1/dane_projekt1.csv")

#Podstawowych statystyk
print("Podstawowe statystyki")
print(df.describe())

#Pierwsze 5 wierszy
print("\nPierwsze pięć wierszy")
print(df.head())

#Podział na cechy (X) i etykiety (y)
X = df.drop(columns=["Gene_Function"])
y = df["Gene_Function"]

#80% dane treningowo-walidacyjne, 20% dane testowe
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#60% treningowe, 20% walidacyjne, 20% testowe
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

#Informacje o rozmiarach zbiorów
print("\nZbiory:")
print("Treningowy:", X_train.shape)
print("Walidacyjny:", X_val.shape)
print("Testowy:", X_test.shape)

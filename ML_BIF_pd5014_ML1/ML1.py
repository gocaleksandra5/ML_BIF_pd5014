#To zainstalowałam w terminalu
#pip install tensorflow keras numpy pandas matplotlib seaborn umap-learn scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split

#Na tym etapie nie było dla mnie jasne, czy dane powinny być wczytywane bezpośrednio z pliku CSV udostępnionego w repozytorium GitHub, czy też można założyć, 
#że plik znajduje się lokalnie na komputerze użytkownika. Z tego względu przygotowałam dwie wersje rozwiązania: 
#pierwszą wykorzystującą plik CSV pobierany z GitHuba, a drugą zakładającą, że plik jest już zapisany lokalnie i dostępny do wczytania z dysku.

#df = pd.read_csv("dane_projekt1.csv")  - pobrane lokalnie 
df = pd.read_csv("https://raw.githubusercontent.com/gocaleksandra5/ML_BIF_pd5014/main/ML_BIF_pd5014_ML1/dane_projekt1.csv") # dane pobierane z Github

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

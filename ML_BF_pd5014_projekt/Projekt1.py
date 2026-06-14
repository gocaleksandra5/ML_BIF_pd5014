import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


#Dane

url = "https://raw.githubusercontent.com/gocaleksandra5/ML_BIF_pd5014/refs/heads/main/ML_BF_pd5014_projekt/dane_projekt1.csv"

df = pd.read_csv(url)

#Info o danych: 

print("Pierwsze 5 wierszy danych:")
print(df.head())

print("\nOpis statystyczny danych:")
print(df.describe())

print("\nLiczba genów w poszczególnych klasach:")
print(df["Gene_Function"].value_counts())



df = df.drop(columns=["Gene_ID"]) #Usunełam identyfikacje genu

X = df.drop(columns=["Gene_Function"]) #Rodzieliłam cechy  i etykiety
y = df["Gene_Function"]

#Zminiłam etykiety tekstowe na liczby 
encoder = LabelEncoder()
y = encoder.fit_transform(y)



print("\nZakodowane klasy:")
for klasa, numer in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"{klasa} -> {numer}")

#Skalowanie danych
scaler = StandardScaler()
X = scaler.fit_transform(X)


#Podział na zbiór treningowy i testowy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\nLiczba próbek treningowych:", len(X_train))
print("Liczba próbek testowych:", len(X_test))

#Sieć neuronowa

mlp = MLPClassifier(hidden_layer_sizes=(8, 16), activation="relu", max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

#Obliczenie metryk
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score( y_test, y_pred, average="weighted")
f1 = f1_score( y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)


print("\nWyniki modelu MLP")

print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1, 2))

print("\n Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)


#Macierz pomyłek

plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_)

plt.xlabel("Przewidziana klasa")
plt.ylabel("Rzeczywista klasa")
plt.title("Funkcji genów (MLP)")
plt.tight_layout()
plt.show()



#Interpretacja wyników:

#W obecnej konfiguracji model osiągnął najlepsze wyniki, uzyskując skuteczność na poziomie 50%. Oznacza to, że poprawnie sklasyfikował 5 z 10 genów znajdujących się w zbiorze testowym.
#Dane zostały podzielone na 40 próbek treningowych oraz 10 próbek testowych.
#Model najlepiej poradził sobie z genami należącymi do grupy structural_protein, dla której uzyskano: precision = 0,67, recall = 1,00 oraz F1-score = 0,80. Wartość recall równa 1 oznacza, że wszystkie geny z tej klasy zostały poprawnie rozpoznane przez model.
#Dla klasy transcription_factor model osiągnął średnie wyniki: precision = 0,5, recall = 0,67 oraz F1-score = 0,57. Oznacza to, że większość genów została sklasyfikowana poprawnie, jednak część przewidywań była błędna.
#Najsłabsze wyniki model uzyskał dla klas enzymów. Wszystkie kategorie enzymatyczne otrzymały wartości równe zero, co oznacza, że model nie przypisał poprawnie żadnego genu do odpowiedniej klasy.
#Podsumowując, model wykazuje umiarkowaną skuteczność w przewidywaniu funkcji genów na podstawie dostępnych danych. Dobrze rozpoznaje niektóre klasy, jednak w przypadku klas enzymatycznych jego działanie jest niezadowalające.
#Niska skuteczność modelu może wynikać z niewielkiej liczby próbek dostępnych do treningu i testowania, co ogranicza możliwość nauczenia się charakterystycznych cech poszczególnych klas.


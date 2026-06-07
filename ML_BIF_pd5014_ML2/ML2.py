import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Wczytywanie danych
data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
    'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3],
    'KRAS': [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['TP53_expr', 'BRCA1_expr', 'TF_motifs', 'KRAS']]
y = df['Cancer_status']

#Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

#Drzewo
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("Drzewo - Accuracy :", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall :", recall_score(y_test, y_pred_tree))
print("F1-score :", f1_score(y_test, y_pred_tree))

plt.figure(figsize=(10, 6))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["Healthy", "Cancer"],
    filled=True
)
plt.title("Drzewo decyzyjne")
plt.show()

#Las losowy
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


print("Las losowy - Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#Macierz błędu
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Healthy", "Cancer"])

plt.title("Random Forest - Confusion Matrix")
plt.show()

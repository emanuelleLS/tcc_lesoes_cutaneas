import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === 1. Carregar a base de treinamento ===
caminho_dados = r"C:\Users\DettCloud2\Downloads\tcc\base_treinamento.csv"
df = pd.read_csv(caminho_dados)

# === 2. Selecionar as colunas de entrada (features) ===
X = df[["area", "perimetro", "circularidade", "aspect_ratio", "solidez"]]

# === 3. Selecionar o rótulo (target) ===
y = df["diagnostico_real"]

# === 4. Dividir em treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Treinar o modelo Random Forest com balanceamento ===
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced'
)

model.fit(X_train, y_train)

# === 6. Avaliar o modelo ===
y_pred = model.predict(X_test)
print("✅ Relatório de desempenho:")
print(classification_report(y_test, y_pred))

print("📊 Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# === 7. Salvar o modelo treinado ===
caminho_modelo = r"C:\Users\DettCloud2\Downloads\tcc\modelo_random_forest.pkl"
joblib.dump(model, caminho_modelo)
print(f"✅ Modelo salvo com sucesso em: {caminho_modelo}")

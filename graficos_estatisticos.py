import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from matplotlib.backends.backend_pdf import PdfPages

# ==================== Configurações ====================

csv_resultados = r"C:\Users\DettCloud2\Downloads\tcc\results_100\relatorio_lote.csv"
csv_metadata = (
    r"C:\Users\DettCloud2\Downloads\tcc\ham10000\metadata\HAM10000_metadata.csv"
)
output_folder = r"C:\Users\DettCloud2\Downloads\tcc\results2406\graficos"

os.makedirs(output_folder, exist_ok=True)

# ==================== Carregar dados ====================

resultados = pd.read_csv(csv_resultados)
metadata = pd.read_csv(csv_metadata)

resultados["imagem"] = resultados["imagem"].str.replace(".jpg", "", regex=False)
df = pd.merge(
    resultados, metadata[["image_id", "dx"]], left_on="imagem", right_on="image_id"
)

mapa = {
    "akiec": "SUSPEITA",
    "bcc": "SUSPEITA",
    "mel": "SUSPEITA",
    "bkl": "PROVAVELMENTE BENIGNA",
    "df": "PROVAVELMENTE BENIGNA",
    "nv": "PROVAVELMENTE BENIGNA",
    "vasc": "PROVAVELMENTE BENIGNA",
}
df["diagnostico_real"] = df["dx"].map(mapa)

y_true = df["diagnostico_real"]
y_pred = df["classificacao"]

# ==================== 📊 Métricas ====================

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label="SUSPEITA")
rec = recall_score(y_true, y_pred, pos_label="SUSPEITA")
f1 = f1_score(y_true, y_pred, pos_label="SUSPEITA")
cm = confusion_matrix(y_true, y_pred, labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"])

# ========== 🔥 Mostrar no terminal ==========

print("📊 Avaliação do Desempenho:")
print(f"Acurácia: {acc*100:.2f}%")
print(f"Precisão (SUSPEITA): {prec*100:.2f}%")
print(f"Recall (SUSPEITA): {rec*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")

# ========== 🔍 Mostrar matriz de confusão na tela ==========

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()

# ==================== 📝 PDF ====================

pdf_path = os.path.join(output_folder, "relatorio_geral.pdf")
pdf = PdfPages(pdf_path)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# ==================== 📑 CAPA ====================

plt.figure(figsize=(8.5, 11))
plt.axis("off")
plt.title(
    "Relatório de Análise e Validação\nSoftware de Análise de Lesões Cutâneas",
    fontsize=18,
    pad=20,
    weight="bold",
)
plt.text(
    0,
    0.7,
    "📅 Projeto TCC\n👩‍💻 Desenvolvido por Emanuelle\n\nEste relatório contém:\n\n"
    "• Análise estatística das características morfológicas\n"
    "• Avaliação do desempenho do software frente aos diagnósticos reais\n"
    "• Métricas quantitativas e matriz de confusão",
    fontsize=13,
)
pdf.savefig()
plt.close()

# ==================== 🔍 Análise Estatística ====================

# 1. Distribuição das Classificações
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="classificacao",
    palette="pastel",  # Usando uma paleta suave
    edgecolor="black",  # Bordas discretas
    linewidth=0.8
)
plt.title("Distribuição das Classificações (Software)", fontsize=14, weight="bold")
plt.xlabel("Classificação", fontsize=12)
plt.ylabel("Quantidade", fontsize=12)

# Adicionando rótulos nas barras
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')

plt.tight_layout()
pdf.savefig()
plt.close()

# 2. Distribuição dos Diagnósticos Reais
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="diagnostico_real",
    palette="muted",  # Paleta mais neutra
    edgecolor="black",  # Bordas discretas
    linewidth=0.8
)
plt.title("Distribuição dos Diagnósticos Reais", fontsize=14, weight="bold")
plt.xlabel("Diagnóstico Real", fontsize=12)
plt.ylabel("Quantidade", fontsize=12)

# Adicionando rótulos nas barras
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')

plt.tight_layout()
pdf.savefig()
plt.close()

# 3. Histograma de Área
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="area",
    hue="classificacao",
    bins=20,  # Ajustando o número de bins
    kde=True,  # Linha de densidade
    palette="pastel",  # Paleta suave
    multiple="stack",
    edgecolor="black",
    linewidth=0.8
)
plt.title("Distribuição da Área das Lesões por Classificação", fontsize=14, weight="bold")
plt.xlabel("Área", fontsize=12)
plt.ylabel("Frequência", fontsize=12)

plt.tight_layout()
pdf.savefig()
plt.close()

# 4. Scatterplot Área vs Circularidade
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="area",
    y="circularidade",
    hue="classificacao",
    palette="muted",  # Paleta suave
    edgecolor="black",  # Bordas discretas
    linewidth=0.6,
    s=50  # Ajuste no tamanho dos pontos
)
plt.title("Relação: Área vs Circularidade", fontsize=14, weight="bold")
plt.xlabel("Área", fontsize=12)
plt.ylabel("Circularidade", fontsize=12)

plt.tight_layout()
pdf.savefig()
plt.close()

# 5. Boxplot Circularidade
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="classificacao",
    y="circularidade",
    palette="coolwarm",  # Paleta suave
    linewidth=1.2
)
plt.title("Boxplot da Circularidade por Classificação", fontsize=14, weight="bold")
plt.xlabel("Classificação", fontsize=12)
plt.ylabel("Circularidade", fontsize=12)

plt.tight_layout()
pdf.savefig()
plt.close()

# ==================== 🧠 Matriz de Confusão no PDF ====================

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"]
)
disp.plot(
    cmap="Blues",  # Colormap suave
    values_format="d",
    ax=ax,
    colorbar=False,
)

plt.title("Matriz de Confusão", fontsize=16, weight="bold")
plt.grid(False)
plt.xlabel("Classe Predita", fontsize=12)
plt.ylabel("Classe Real", fontsize=12)
plt.subplots_adjust(left=0.436, bottom=0.367, right=0.998, top=0.938, wspace=0.2, hspace=0.2)
plt.tight_layout()
pdf.savefig()
plt.close()

# ==================== 🔥 Página de Métricas ====================

plt.figure(figsize=(8.5, 11))
plt.axis("off")
plt.title("Relatório de Desempenho do Software", fontsize=16, pad=20, weight="bold")

texto = (
    f"📊 Avaliação Quantitativa\n\n"
    f"- Acurácia: {acc*100:.2f}%\n"
    f"- Precisão (SUSPEITA): {prec*100:.2f}%\n"
    f"- Recall (SUSPEITA): {rec*100:.2f}%\n"
    f"- F1-Score: {f1*100:.2f}%\n\n"
    f"🔍 Interpretação:\n"
    f"- Acurácia geral reflete a taxa de acertos do software.\n"
    f"- Precisão indica quantos dos classificados como SUSPEITA realmente são.\n"
    f"- Recall mede a capacidade de detectar todas as lesões suspeitas.\n"
    f"- F1-Score é o equilíbrio entre precisão e recall.\n\n"
    f"📝 Observações:\n"
    f"- A matriz de confusão mostra erros e acertos claramente.\n"
    f"- O recall alto é importante para não deixar lesões suspeitas passarem.\n"
)
plt.text(0, 1, texto, fontsize=12, va="top")

pdf.savefig()
plt.close()

# ====================  Fechar PDF ====================

pdf.close()

# ====================  CSV de Comparação ====================

df[["imagem", "diagnostico_real", "classificacao"]].to_csv(
    os.path.join(output_folder, "comparacao.csv"), index=False
)

print(f"✅ PDF salvo em: {pdf_path}")
print(f"✅ CSV salvo em: {os.path.join(output_folder, 'comparacao.csv')}")

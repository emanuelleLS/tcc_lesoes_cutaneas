import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# Carregar dados
csv_path = "results_teste100/relatorio_lote.csv"
df = pd.read_csv(csv_path)

# Criar pasta para gráficos
output_folder = "results_teste100/plots"
os.makedirs(output_folder, exist_ok=True)

# Começar a gerar PDF
pdf_path = "results_teste100/relatorio_visual.pdf"
pdf = PdfPages(pdf_path)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Gráfico 1: Classificação
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="classificacao", palette=["green", "red"])
plt.title("Distribuição das Classificações")
plt.xlabel("Classificação")
plt.ylabel("Quantidade")
pdf.savefig(); plt.close()

# Gráfico 2: Histograma de Área por Classificação
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="area", hue="classificacao", bins=20, kde=True, palette="Set2", multiple="stack")
plt.title("Distribuição da Área das Lesões por Classificação")
plt.xlabel("Área")
plt.ylabel("Frequência")
pdf.savefig(); plt.close()

# Gráfico 3: Scatterplot Área x Circularidade
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="area", y="circularidade", hue="classificacao", palette="Set1")
plt.title("Área vs Circularidade")
plt.xlabel("Área")
plt.ylabel("Circularidade")
pdf.savefig(); plt.close()

# Gráfico 4: Boxplot Circularidade por Classificação
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="classificacao", y="circularidade", palette="coolwarm")
plt.title("Boxplot da Circularidade por Classificação")
plt.xlabel("Classificação")
plt.ylabel("Circularidade")
pdf.savefig(); plt.close()

# Gráfico 5: Pairplot geral (salvar só como imagem extra)
sns.pairplot(df, hue="classificacao", corner=True, palette="husl")
plt.savefig(f"{output_folder}/pairplot.png")
plt.close()

# Fechar o PDF
pdf.close()
print(f"✅ Relatório visual salvo em: {pdf_path}")
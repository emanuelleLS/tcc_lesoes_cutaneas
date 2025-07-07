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
csv_resultados = r"C:\Users\DettCloud2\Downloads\tcc\results_lote_0707_ml\relatorio_lote.csv"
csv_metadata = r"C:\Users\DettCloud2\Downloads\tcc\ham10000\metadata\HAM10000_metadata.csv"
output_folder = r"C:\Users\DettCloud2\Downloads\tcc\results_lote_0707_ml\graficos"

os.makedirs(output_folder, exist_ok=True)

# ==================== Funções Utilitárias ====================

def carregar_dados(csv_resultados, csv_metadata):
    """Carregar dados de resultados e metadados."""
    resultados = pd.read_csv(csv_resultados)
    metadata = pd.read_csv(csv_metadata)
    resultados["imagem"] = resultados["imagem"].str.replace(".jpg", "", regex=False)
    df = pd.merge(resultados, metadata[["image_id", "dx"]], left_on="imagem", right_on="image_id")

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
    
    return df

def calcular_metricas(y_true, y_pred):
    """Calcular as métricas de desempenho do modelo."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="SUSPEITA")
    rec = recall_score(y_true, y_pred, pos_label="SUSPEITA")
    f1 = f1_score(y_true, y_pred, pos_label="SUSPEITA")
    return acc, prec, rec, f1

def gerar_matriz_confusao(y_true, y_pred):
    """Gerar e exibir a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred, labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.show()
    return cm

def gerar_relatorio_pdf(acc, prec, rec, f1, cm, output_folder):
    """Gerar um relatório em PDF com as métricas e a matriz de confusão."""
    pdf_path = os.path.join(output_folder, "relatorio_desempenho.pdf")
    pdf = PdfPages(pdf_path)

    # Página de Métricas
    plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.title("Relatório de Desempenho do Modelo", fontsize=16, pad=20, weight="bold")
    
    texto = (
        f"📊 Avaliação Quantitativa\n\n"
        f"- Acurácia: {acc*100:.2f}%\n"
        f"- Precisão (SUSPEITA): {prec*100:.2f}%\n"
        f"- Recall/Sensibilidade (SUSPEITA): {rec*100:.2f}%\n"
        f"- F1-Score: {f1*100:.2f}%\n\n"
        f"🔍 Interpretação:\n"
        f"- Acurácia geral reflete a taxa de acertos do modelo.\n"
        f"- Precisão indica quantos dos classificados como SUSPEITA realmente são.\n"
        f"- Recall mede a capacidade de detectar todas as lesões SUSPEITAS.\n"
        f"- F1-Score é o equilíbrio entre precisão e recall.\n\n"
        f"📝 Observações:\n"
        f"- A matriz de confusão mostra erros e acertos claramente.\n"
        f"- O recall alto é importante para não deixar lesões SUSPEITAS passarem.\n"
    )
    plt.text(0, 1, texto, fontsize=12, va="top")
    
    pdf.savefig()
    plt.close()

    # Página da Matriz de Confusão
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"])
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)

    plt.title("Matriz de Confusão", fontsize=16, weight="bold")
    plt.grid(False)
    plt.xlabel("Classe Predita", fontsize=12)
    plt.ylabel("Classe Real", fontsize=12)
    plt.subplots_adjust(left=0.436, bottom=0.367, right=0.998, top=0.938, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close()
    print(f"✅ Relatório PDF salvo em: {pdf_path}")

def salvar_comparacao(df, output_folder):
    """Salvar o CSV de comparação entre diagnóstico real e classificação do modelo."""
    output_csv = os.path.join(output_folder, "comparacao.csv")
    df[['imagem', 'diagnostico_real', 'classificacao']].to_csv(output_csv, index=False)
    print(f"✅ CSV de comparação salvo em: {output_csv}")
    
def gerar_pairplot_morfologia(df, output_folder):
    """Gerar gráfico de dispersão em pares (pairplot) dos atributos morfológicos"""
    import seaborn as sns

    features = ["area", "perimetro", "circularidade", "aspect_ratio", "solidez"]
    df_plot = df[features + ["classificacao"]].copy()

    sns.set(style="whitegrid", font_scale=1.1)
    g = sns.pairplot(
        df_plot,
        hue="classificacao",
        palette="Set1",
        diag_kind="kde",
        markers=["o", "s"],
        plot_kws={'alpha': 0.6, 's': 35}
    )

    g.fig.suptitle("Gráfico de Dispersão dos Atributos Morfológicos", fontsize=16, y=1.03)
    output_path = os.path.join(output_folder, "pairplot_morfologia.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Pairplot salvo em: {output_path}")


# ==================== Análise Estatística e Geração de Gráficos ====================

def gerar_graficos(df, acc, prec, rec, f1, cm, output_folder):
    # Gerar gráficos para análise estatística
    pdf_path = os.path.join(output_folder, "relatorio_geral.pdf")
    pdf = PdfPages(pdf_path)

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

    # Fechar PDF
    pdf.close()

# ==================== Execução Principal ====================

def main():
    # Carregar dados
    df = carregar_dados(csv_resultados, csv_metadata)

    # Labels reais e preditos
    y_true = df['diagnostico_real']
    y_pred = df['classificacao']

    # Calcular métricas
    acc, prec, rec, f1 = calcular_metricas(y_true, y_pred)

    # Mostrar no terminal
    print("📊 Avaliação do Desempenho:")
    print(f"Acurácia: {acc*100:.2f}%")
    print(f"Precisão (SUSPEITA): {prec*100:.2f}%")
    print(f"Recall/Sensibilidade (SUSPEITA): {rec*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")

    # Gerar matriz de confusão
    cm = gerar_matriz_confusao(y_true, y_pred)

    # Gerar relatórios e gráficos
    gerar_relatorio_pdf(acc, prec, rec, f1, cm, output_folder)
    gerar_graficos(df, acc, prec, rec, f1, cm, output_folder)
    salvar_comparacao(df, output_folder)
    gerar_pairplot_morfologia(df, output_folder)
if __name__ == "__main__":
    main()

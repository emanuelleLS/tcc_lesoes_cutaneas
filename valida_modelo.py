import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from matplotlib.backends.backend_pdf import PdfPages
import os

# =================== CONFIGURAÇÕES ===================

# Caminho dos arquivos (modifique conforme necessário)
csv_resultados = r"C:\Users\DettCloud2\Downloads\tcc\results_lote\relatorio_lote.csv"
csv_metadata = r"C:\Users\DettCloud2\Downloads\tcc\ham10000\HAM10000_metadata.csv"
output_folder = r"C:\Users\DettCloud2\Downloads\tcc\results_lote"

os.makedirs(output_folder, exist_ok=True)

# =================== CARREGAR DADOS ===================

def carregar_dados(csv_resultados, csv_metadata):
    """Carregar os dados dos arquivos CSV."""
    resultados = pd.read_csv(csv_resultados)
    metadata = pd.read_csv(csv_metadata)

    # Remover extensão '.jpg' das imagens no resultados
    resultados['imagem'] = resultados['imagem'].str.replace('.jpg', '', regex=False)

    # Merge entre resultados e metadados
    df = pd.merge(resultados, metadata[['image_id', 'dx']], left_on='imagem', right_on='image_id')
    
    # Mapeamento para diagnóstico real (SUSPEITA ou BENIGNA)
    mapa = {
        'akiec': 'SUSPEITA',
        'bcc': 'SUSPEITA',
        'mel': 'SUSPEITA',
        'bkl': 'PROVAVELMENTE BENIGNA',
        'df': 'PROVAVELMENTE BENIGNA',
        'nv': 'PROVAVELMENTE BENIGNA',
        'vasc': 'PROVAVELMENTE BENIGNA'
    }
    df['diagnostico_real'] = df['dx'].map(mapa)

    return df

# =================== CÁLCULO DE MÉTRICAS ===================

def calcular_metricas(y_true, y_pred):
    """Calcular as métricas de desempenho do modelo."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="SUSPEITA")
    rec = recall_score(y_true, y_pred, pos_label="SUSPEITA")
    f1 = f1_score(y_true, y_pred, pos_label="SUSPEITA")
    
    return acc, prec, rec, f1

# =================== GERAR MATRIZ DE CONFUSÃO ===================

def gerar_matriz_confusao(y_true, y_pred):
    """Gerar e exibir a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred, labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SUSPEITA", "PROVAVELMENTE BENIGNA"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.show()

    return cm

# =================== GERAR RELATÓRIO EM PDF ===================

def gerar_relatorio_pdf(acc, prec, rec, f1, cm, output_folder):
    """Gerar um relatório em PDF com as métricas e matriz de confusão."""
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

    # Fechar PDF
    pdf.close()

    print(f"✅ Relatório PDF salvo em: {pdf_path}")

# =================== GERAR CSV DE COMPARAÇÃO ===================

def salvar_comparacao(df, output_folder):
    """Salvar o CSV de comparação entre diagnóstico real e classificação do modelo."""
    output_csv = os.path.join(output_folder, "comparacao.csv")
    df[['imagem', 'diagnostico_real', 'classificacao']].to_csv(output_csv, index=False)
    print(f"✅ CSV de comparação salvo em: {output_csv}")

# =================== EXECUÇÃO PRINCIPAL ===================

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

    # Gerar relatório PDF
    gerar_relatorio_pdf(acc, prec, rec, f1, cm, output_folder)

    # Salvar comparação em CSV
    salvar_comparacao(df, output_folder)

if __name__ == "__main__":
    main()

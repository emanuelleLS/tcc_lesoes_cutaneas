from lesion_analyzer import SkinLesionAnalyzer
import os
import csv
import shutil
import time

# =================== CONFIGURAÇÕES ===================

# Definir os caminhos conforme a necessidade:
# ✔️ Basta mudar esses dois caminhos para usar em qualquer lote

# caminho_imgs = r"C:\Users\DettCloud2\Downloads\tcc\ham10000\images"  # Caminho das imagens
caminho_imgs = r"C:\Users\DettCloud2\Downloads\tcc\ham10000\teste100" 
saida = r"C:\Users\DettCloud2\Downloads\tcc\results_lote_0707_ml"             # Pasta de saída
# saida = r"C:\Users\DettCloud2\Downloads\tcc\results2406"   

# Criar pasta de saída se não existir
os.makedirs(saida, exist_ok=True)

# ==========================================================

# Função robusta para mover arquivos (sobrescreve se existir)
def mover_com_sobrescrita(origem, destino):
    if os.path.exists(destino):
        os.remove(destino)
    shutil.move(origem, destino)

# Listar imagens válidas
imagens = [img for img in os.listdir(caminho_imgs) if img.lower().endswith(".jpg")]
dados_csv = []

print(f"🔬 Total de imagens encontradas: {len(imagens)}")

# Loop principal de processamento
for i, nome_img in enumerate(imagens, 1):
    caminho = os.path.join(caminho_imgs, nome_img)
    print(f"[{i}/{len(imagens)}] Processando: {nome_img}")
    try:
        analyzer = SkinLesionAnalyzer(caminho)
        analyzer.analyze()

        base = nome_img.replace(".jpg", "")
        mover_com_sobrescrita("results/lesion_analysis.png", os.path.join(saida, f"{base}_analysis.png"))
        mover_com_sobrescrita("results/lesion_report.txt", os.path.join(saida, f"{base}_report.txt"))

        time.sleep(0.1)  # Pequena pausa para evitar conflitos de I/O

        try:
            with open(os.path.join(saida, f"{base}_report.txt"), encoding="utf-8") as f:
                linhas = f.readlines()
                area = float(linhas[3].split(":")[1])
                perimetro = float(linhas[4].split(":")[1])
                circ = float(linhas[5].split(":")[1])
                ar = float(linhas[6].split(":")[1])
                solidez = float(linhas[7].split(":")[1])
                classificacao = linhas[9].split(":")[1].strip()
                dados_csv.append([base, area, perimetro, circ, ar, solidez, classificacao])
        except Exception as e:
            print(f"⚠️ Erro ao ler relatório de {base}: {e}")
            continue

    except Exception as e:
        print(f"⚠️ Erro ao processar {nome_img}: {e}")
        continue

# Gerar CSV
csv_path = os.path.join(saida, "relatorio_lote.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["imagem", "area", "perimetro", "circularidade", "aspect_ratio", "solidez", "classificacao"])
    writer.writerows(dados_csv)

print(f"✅ Relatório CSV salvo em: {csv_path}")

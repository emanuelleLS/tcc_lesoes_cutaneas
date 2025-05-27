from lesion_analyzer import SkinLesionAnalyzer
import os
import csv
import shutil
import time

# Caminho das 100 imagens de teste
caminho_imgs = "ham10000/teste100"
saida = "results_teste100"
os.makedirs(saida, exist_ok=True)

# Função segura para mover arquivos com sobrescrita
def mover_com_sobrescrita(origem, destino):
    if os.path.exists(destino):
        os.remove(destino)
    shutil.move(origem, destino)

imagens = [img for img in os.listdir(caminho_imgs) if img.endswith(".jpg")]
dados_csv = []

print(f"🔬 Total de imagens de teste: {len(imagens)}")

for i, nome_img in enumerate(imagens, 1):
    caminho = os.path.join(caminho_imgs, nome_img)
    print(f"[{i}/{len(imagens)}] Processando: {nome_img}")
    try:
        analyzer = SkinLesionAnalyzer(caminho)
        analyzer.analyze()

        base = nome_img.replace(".jpg", "")
        mover_com_sobrescrita("results/lesion_analysis.png", f"{saida}/{base}_analysis.png")
        mover_com_sobrescrita("results/lesion_report.txt", f"{saida}/{base}_report.txt")

        time.sleep(0.1)  # pequena pausa para evitar conflito de leitura

        try:
            with open(f"{saida}/{base}_report.txt", encoding="utf-8") as f:
                linhas = f.readlines()
                area = float(linhas[3].split(":")[1])
                perimetro = float(linhas[4].split(":")[1])
                circ = float(linhas[5].split(":")[1])
                ar = float(linhas[6].split(":")[1])
                solidez = float(linhas[7].split(":")[1])
                classificacao = linhas[9].split(":")[1].strip()
                dados_csv.append([base, area, perimetro, circ, ar, solidez, classificacao])
        except UnicodeDecodeError as e:
            print(f"⚠️ Erro ao ler {base}_report.txt: {e}")
            continue

    except Exception as e:
        print(f"⚠️ Erro em {nome_img}: {e}")

# Gerar relatorio_lote.csv
csv_path = os.path.join(saida, "relatorio_lote.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["nome_imagem", "area", "perimetro", "circularidade", "aspect_ratio", "solidez", "classificacao"])
    writer.writerows(dados_csv)

print(f"✅ Relatório salvo em {csv_path}")
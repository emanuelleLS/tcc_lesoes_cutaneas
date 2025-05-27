from lesion_analyzer import SkinLesionAnalyzer
import os

# Caminho unificado para as imagens da base
caminho_imgs = "ham10000/images"
saida = "results_lote"
os.makedirs(saida, exist_ok=True)

# Lista de imagens
imagens = [img for img in os.listdir(caminho_imgs) if img.endswith('.jpg')]

print(f"🔍 Total de imagens encontradas: {len(imagens)}")

for i, nome_img in enumerate(imagens, 1):
    caminho = os.path.join(caminho_imgs, nome_img)
    print(f"\n[{i}/{len(imagens)}] Processando: {nome_img}")
    try:
        analyzer = SkinLesionAnalyzer(caminho)
        analyzer.analyze()

        base = nome_img.replace(".jpg", "")
        os.rename("results/lesion_analysis.png", f"{saida}/{base}_analysis.png")
        os.rename("results/lesion_report.txt", f"{saida}/{base}_report.txt")

    except Exception as e:
        print(f"⚠️ Erro ao processar {nome_img}: {e}")

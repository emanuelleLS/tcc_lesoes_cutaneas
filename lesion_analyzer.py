import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.segmentation import watershed
import joblib
import pandas as pd

class SkinLesionAnalyzer:
    def __init__(self, image_path, circularity_threshold=0.4, aspect_ratio_threshold=0.5, area_threshold=10000):
        """Inicializa o analisador com uma imagem específica e parâmetros ajustáveis"""
        os.makedirs('results', exist_ok=True)
        self.image_path = image_path
        self.modelo_path = r"C:\Users\DettCloud2\Downloads\tcc\modelo_random_forest.pkl"
        self.modelo = joblib.load(self.modelo_path)

        
        # Parâmetros de classificação
        self.circularity_threshold = circularity_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.area_threshold = area_threshold

    def load_image(self):
        """Carrega a imagem local"""
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem em {self.image_path}")
            return image
        except Exception as e:
            print(f"Erro ao carregar imagem: {str(e)}")
            return None

    def preprocess_image(self, image):
        """Pré-processamento da imagem"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))

        # Remoção de ruído local (pontual)
        denoised = cv2.medianBlur(resized, 5)

        # Equalização de histograma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(denoised)

        # Suavização global (bordas e fundo)
        smoothed = cv2.GaussianBlur(equalized, (5, 5), 0)

        return smoothed

    def segment_lesion(self, image):
        """Segmenta a lesão usando binarização, morfologia e watershed"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)

        # Abertura para remover ruídos
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Fechamento para preencher buracos
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Bordas para visualização (não influencia na segmentação final)
        edges = cv2.Canny(image, 100, 200)

        try:
            dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            sure_bg = cv2.dilate(closing, kernel, iterations=3)
            unknown = cv2.subtract(sure_bg, sure_fg)

            _, markers = cv2.connectedComponents(sure_fg)
            markers += 1
            markers[unknown == 255] = 0

            markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[markers > 1] = 255

            return mask, edges
        except Exception as e:
            print(f"Erro no watershed: {str(e)}")
            return closing, edges

    def extract_features(self, mask):
        """Extrai características da lesão segmentada"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        }

    def classify_lesion(self, features):
        """Classifica usando o modelo treinado, com nomes consistentes"""
        entrada = pd.DataFrame([{
            'area': features['area'],
            'perimetro': features['perimeter'],
            'circularidade': features['circularity'],
            'aspect_ratio': features['aspect_ratio'],
            'solidez': features['solidity'],
        }])
        return self.modelo.predict(entrada)[0]


    def generate_report(self, original, processed, mask, edges, features, classification):
        """Gera relatório visual e textual"""
        plt.figure(figsize=(15, 10))

        # Imagem original
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        # Pré-processada
        plt.subplot(2, 3, 2)
        plt.imshow(processed, cmap='gray')
        plt.title('Pré-processada')
        plt.axis('off')

        # Máscara segmentada
        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Segmentação')
        plt.axis('off')

        # Bordas
        plt.subplot(2, 3, 4)
        plt.imshow(edges, cmap='gray')
        plt.title('Bordas (Canny)')
        plt.axis('off')

        # Sobreposição
        plt.subplot(2, 3, 5)
        overlay = cv2.addWeighted(
            cv2.cvtColor(cv2.resize(original, (256, 256)), cv2.COLOR_BGR2RGB), 0.7,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0
        )
        plt.imshow(overlay)
        plt.title('Sobreposição')
        plt.axis('off')

        # Texto das características
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.5,
                 f"Área: {features['area']:.2f}\n"
                 f"Perímetro: {features['perimeter']:.2f}\n"
                 f"Circularidade: {features['circularity']:.2f}\n"
                 f"Aspect Ratio: {features['aspect_ratio']:.2f}\n"
                 f"Solidez: {features['solidity']:.2f}\n\n"
                 f"Classificação:\n{classification}",
                 fontsize=10)
        plt.axis('off')

        plt.suptitle("Análise da Lesão Cutânea", fontsize=16)
        plt.savefig("results/lesion_analysis.png", bbox_inches='tight')
        plt.close()

        # Relatório em texto
        with open("results/lesion_report.txt", 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE LESÃO CUTÂNEA\n")
            f.write("=" * 50 + "\n")
            f.write("CARACTERÍSTICAS:\n")
            for k, v in features.items():
                f.write(f"- {k}: {v:.2f}\n")
            f.write(f"\nCLASSIFICAÇÃO: {classification}\n")
            f.write("\nOBSERVAÇÕES:\n")
            f.write("Recomenda-se avaliação dermatológica.\n" if classification == "SUSPEITA"
                    else "Lesão com características benignas.\n")
            f.write("\nPARAMETROS DE CLASSIFICACAO:\n")
            f.write(f"- Circularidade < {self.circularity_threshold}\n")
            f.write(f"- Aspect Ratio > {self.aspect_ratio_threshold}\n")
            f.write(f"- Área > {self.area_threshold}\n")

    def analyze(self):
        """Executa o pipeline completo de análise"""
        print("\nIniciando análise da imagem...")

        original = self.load_image()
        if original is None:
            return

        processed = self.preprocess_image(original)
        mask, edges = self.segment_lesion(processed)
        features = self.extract_features(mask)

        if not features:
            print("Falha na extração de características")
            return

        classification = self.classify_lesion(features)
        self.generate_report(original, processed, mask, edges, features, classification)
        print("Análise concluída! Verifique a pasta 'results'.")

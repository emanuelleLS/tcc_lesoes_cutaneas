import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.segmentation import watershed

class SkinLesionAnalyzer:
    def __init__(self, image_path):
        """Inicializa o analisador com uma imagem específica"""
        os.makedirs('results', exist_ok=True)
        self.image_path = image_path
        
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
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar
        resized = cv2.resize(gray, (256, 256))
        
        # Remoção de ruído
        denoised = cv2.medianBlur(resized, 5)
        
        # Equalização de histograma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(denoised)
        
        return equalized

    def segment_lesion(self, image):
        """Segmenta a lesão usando múltiplas técnicas"""
        # Binarização com Otsu
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operações morfológicas
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Detecção de bordas com Canny
        edges = cv2.Canny(image, 100, 200)
        
        # Watershed para segmentação avançada
        try:
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
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
            return opening, edges

    def extract_features(self, mask):
        """Extrai características da lesão segmentada"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        cnt = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        _, _, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h > 0 else 0
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        }

    def classify_lesion(self, features):
        """Classifica a lesão com base nas características"""
        irregular = features['circularity'] < 0.6
        asymmetric = abs(features['aspect_ratio'] - 1.0) > 0.3
        large = features['area'] > 5000
        
        return "SUSPEITA" if (irregular or asymmetric or large) else "PROVAVELMENTE BENIGNA"

    def generate_report(self, original, processed, mask, edges, features, classification):
        """Gera relatório completo"""
        # Configurar figura
        plt.figure(figsize=(15, 10))
        
        # Imagem original
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        
        # Imagem processada
        plt.subplot(2, 3, 2)
        plt.imshow(processed, cmap='gray')
        plt.title('Pré-processada')
        plt.axis('off')
        
        # Máscara
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
        overlay = cv2.addWeighted(cv2.cvtColor(cv2.resize(original, (256, 256)), cv2.COLOR_BGR2RGB), 0.7,
                                 cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
        plt.imshow(overlay)
        plt.title('Sobreposição')
        plt.axis('off')
        
        # Características
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
        
        # Salvar resultados
        img_path = "results/lesion_analysis.png"
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        
        txt_path = "results/lesion_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE LESÃO CUTÂNEA\n")
            f.write("="*50 + "\n")
            f.write("CARACTERÍSTICAS:\n")
            for k, v in features.items():
                f.write(f"- {k}: {v:.2f}\n")
            f.write(f"\nCLASSIFICAÇÃO: {classification}\n")
            f.write("\nOBSERVAÇÕES:\n")
            f.write("Recomenda-se avaliação dermatológica.\n" if classification == "SUSPEITA" 
                   else "Lesão com características benignas.\n")

    def analyze(self):
        """Executa o pipeline completo de análise"""
        print("\nIniciando análise da imagem...")
        
        # Carregar imagem
        original = self.load_image()
        if original is None:
            return
            
        # Pré-processamento
        processed = self.preprocess_image(original)
        
        # Segmentação
        mask, edges = self.segment_lesion(processed)
        
        # Extração de características
        features = self.extract_features(mask)
        if not features:
            print("Falha na extração de características")
            return
            
        # Classificação
        classification = self.classify_lesion(features)
        
        # Relatório
        self.generate_report(original, processed, mask, edges, features, classification)
        print("Análise concluída! Verifique a pasta 'results'.")

from lesion_analyzer import SkinLesionAnalyzer

if __name__ == "__main__":
    image_path = r"samples/lpele5.jpg"  # substitua pelo nome do arquivo real
    analyzer = SkinLesionAnalyzer(image_path)
    analyzer.analyze()

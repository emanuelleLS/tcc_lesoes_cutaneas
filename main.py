from lesion_analyzer import SkinLesionAnalyzer

if __name__ == "__main__":
    image_path = r"samples/lpele5.jpg" 
    analyzer = SkinLesionAnalyzer(image_path)
    analyzer.analyze()

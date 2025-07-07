import pandas as pd

# === 1. Caminhos dos arquivos ===
csv_resultados = r"C:\Users\DettCloud2\Downloads\tcc\results_lote_2506\relatorio_lote.csv"
csv_metadata = r"C:\Users\DettCloud2\Downloads\tcc\ham10000\metadata\HAM10000_metadata.csv"

# === 2. Carregar os CSVs ===
print("🔄 Lendo arquivos...")
resultados = pd.read_csv(csv_resultados)
metadata = pd.read_csv(csv_metadata)

# === 3. Limpar nomes de imagem para bater com metadata ===
resultados["imagem"] = resultados["imagem"].str.replace(".jpg", "", regex=False)

# === 4. Mapear dx → rotulo binário ===
mapa_diagnostico = {
    "akiec": "SUSPEITA",
    "bcc": "SUSPEITA",
    "mel": "SUSPEITA",
    "bkl": "PROVAVELMENTE BENIGNA",
    "df": "PROVAVELMENTE BENIGNA",
    "nv": "PROVAVELMENTE BENIGNA",
    "vasc": "PROVAVELMENTE BENIGNA"
}
metadata["diagnostico_real"] = metadata["dx"].map(mapa_diagnostico)

# === 5. Juntar os dois datasets ===
df_merged = pd.merge(resultados, metadata[["image_id", "diagnostico_real"]],
                     left_on="imagem", right_on="image_id")

# === 6. Verificar as primeiras linhas ===
print("✅ Exemplo de dados preparados:")
print(df_merged.head())

# === 7. Salvar CSV preparado ===
caminho_saida = r"C:\Users\DettCloud2\Downloads\tcc\base_treinamento.csv"
df_merged.to_csv(caminho_saida, index=False)
print(f"✅ Arquivo final salvo em: {caminho_saida}")

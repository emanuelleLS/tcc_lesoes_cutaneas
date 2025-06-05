# Sistema de Detecção de Lesões Cutâneas com Processamento de Imagens

Este projeto realiza análise de imagens de lesões cutâneas utilizando técnicas de processamento de imagens e classificação com base em características morfológicas.

## ⚙️ Requisitos

Instale as dependências com:

```
pip install -r requirements.txt
```

## 📂 Estrutura do Projeto

```
tcc_lesoes_cutaneas/
├── main.py
├── lesion_analyzer.py
├── analisar_lote.py
├── requirements.txt
├── results/
├── results_lote/
├── results_teste100/
├── samples/
└── ham10000/
    ├── metadata/
    │   └── HAM10000_metadata.csv
    └── images/   ← Imagens da base (não incluídas)
```
A pasta `results_teste100/` guarda as saídas e relatórios de testes.

## 📥 Como obter a base de imagens

As imagens da base **HAM10000** não estão incluídas neste repositório devido ao seu tamanho.

Você pode baixá-las manualmente em:

🔗 [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Após baixar, coloque as imagens extraídas em:

```
ham10000/images/
```

E o metadata em:

```
ham10000/metadata/HAM10000_metadata.csv
```

## 🚀 Execução

Para processar uma única imagem de teste:

```
py main.py
```

Para rodar em lote em toda a base HAM10000:

```
py analisar_lote.py
```

Todos os relatórios e imagens segmentadas serão salvos na pasta `results_lote/`.

## 🧠 Observação

Este projeto é acadêmico e não substitui diagnóstico médico. Sempre consulte um especialista.

# Projeto CP5-IA

## Integrantes

### Pedro Henrique dos Santos RM 559064

Este repositório contém os exercícios do checkpoint e a parte de Visão Computacional do trabalho.

Sumário
- Exercício 1 — Classificação (Keras) ✅
- Exercício 2 — Regressão (Keras) ✅
- Parte 02 — Visão Computacional (YOLOv8 + MediaPipe)
- Arquivos e estrutura do repositório

---

## ATIVIDADE – TREINAMENTO DE REDES NEURAIS COM KERAS (DADOS TABULARES)

Esta atividade corresponde a 40% da nota do Checkpoint 2. Apenas um membro do grupo deve submeter o repositório. O repositório deve conter os códigos organizados e um `README.md` explicando cada exercício.

Os arquivos principais para os exercícios tabulares são os notebooks:

- `exercicio1_classificacao.ipynb` — Classificação multiclasses (Wine Dataset UCI)
- `exercicio2_regressao.ipynb` — Regressão (California Housing dataset)

### EXERCÍCIO 1 — CLASSIFICAÇÃO MULTICLASSE (Wine Dataset)

Requisitos implementados:

1. Treinar uma rede neural em Keras para classificar vinhos em 3 classes.
   - Arquivo: `exercicio1_classificacao.ipynb`
   - Arquitetura mínima: 2 camadas ocultas com 32 neurônios cada, ReLU.
   - Saída: 3 neurônios com Softmax.
   - Loss: categorical_crossentropy.
   - Otimizador: Adam.

2. Comparar resultados com um modelo do scikit-learn (RandomForestClassifier ou LogisticRegression).

3. Registrar métricas de acurácia e discutir qual modelo teve melhor desempenho (consulte a seção final do notebook com tabelas e gráficos).

### EXERCÍCIO 2 — REGRESSÃO (California Housing)

Requisitos implementados:

1. Treinar uma rede neural em Keras para prever o valor médio das casas.
   - Arquivo: `exercicio2_regressao.ipynb`
   - Arquitetura mínima: 3 camadas ocultas com 64, 32 e 16 neurônios (ReLU).
   - Saída: 1 neurônio (Linear).
   - Loss: MSE.
   - Otimizador: Adam.

2. Comparar resultados com um modelo scikit-learn (LinearRegression ou RandomForestRegressor).

3. Registrar métricas de erro (RMSE ou MAE) e discutir qual modelo teve melhor desempenho (resultados no notebook).

Como executar os notebooks localmente (recomendado):

```powershell
# criar e ativar ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# instalar dependências comuns (exemplos)
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Se os notebooks usam TensorFlow ou PyTorch, instale conforme necessário, por exemplo:
python -m pip install tensorflow
# ou
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# abrir jupyter
jupyter notebook
```

---

## Parte 02 — Visão Computacional (YOLOv8 + MediaPipe)

Este diretório contém um script de inferência (`visao-computacional/inference.py`) que combina duas ferramentas:

- YOLOv8 (biblioteca `ultralytics`) para detecção de objetos (modelo `yolov8n.pt`).
- MediaPipe (`holistic`) para detecção de landmarks e detecção simples de gesto (mão erguida).

O objetivo é demonstrar o uso combinado das duas ferramentas (detecção + análise de mão) em uma imagem.

### Pré-requisitos

- Windows ou outro sistema com Python 3.8+ instalado.
- PowerShell (instruções abaixo usam PowerShell).
- Arquivo de imagem de teste em `visao-computacional/data/teste.jpg`.
- O peso do YOLO `yolov8n.pt` deve estar na raiz do projeto (ou no diretório corrente ao executar o script).

### Instalação das dependências (packages principais)

Exemplos de instalação via PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics
python -m pip install opencv-python
python -m pip install mediapipe

# Possível necessidade de instalar PyTorch se o ultralytics requerer:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Como executar o `inference.py`

```powershell
cd visao-computacional
python inference.py
```

O script gera `output_result.jpg` no diretório atual e abre uma janela com o resultado via OpenCV.

---

## Estrutura do repositório

- `exercicio1_classificacao.ipynb` — notebook do exercício 1
- `exercicio2_regressao.ipynb` — notebook do exercício 2
- `visao-computacional/` — código e dados da parte de visão computacional
  - `inference.py` — script de inferência combinando YOLOv8 + MediaPipe
  - `data/teste.jpg` — imagem de exemplo
  - `yolov8n.pt` — peso YOLOv8 (se presente na raiz)

---



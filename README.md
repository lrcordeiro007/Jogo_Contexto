# 🎮 Recriação do Jogo Contexto com NLP

Este projeto é uma implementação *open-source* da lógica do jogo [Contexto.me](https://contexto.me/), desenvolvida em Python. O objetivo foi aplicar conceitos de **Processamento de Linguagem Natural (NLP)** estudados na UFG, explorando como máquinas "entendem" a semântica das palavras.

## 📋 Sobre o Projeto

O jogo consiste em descobrir uma palavra secreta através de tentativas. A cada palavra chutada, o algoritmo retorna um número indicando a proximidade semântica em relação à palavra secreta (quanto menor o número, mais próximo).

Diferente de jogos de forca ou palavras cruzadas que analisam letras, este projeto analisa **significado** usando vetores densos (embeddings).

## 🛠 Tecnologias Utilizadas

* **Python 3.x**
* **[Sentence-Transformers](https://www.sbert.net/):** Framework para geração de embeddings de sentenças e textos.
* **Modelo:** `distiluse-base-multilingual-cased-v1` (Modelo multilingual leve e eficiente).
* **Scikit-Learn / SciPy:** (Implícito) Para cálculo de Similaridade de Cosseno.
* **Requests:** Para baixar o dicionário de palavras em PT-BR.

## 🚀 Como Executar

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU-USUARIO/SEU-REPO.git](https://github.com/SEU-USUARIO/SEU-REPO.git)
   cd SEU-REPO
   ```
2. **Instale as dependências:**
   ```bash
   pip install sentence-transformers requests
   ```
3. **Rode o jogo:**
   ```bash
   python main.py
   ```
Nota: Na primeira execução, o script fará o download do dicionário e do modelo, o que pode levar alguns minutos. As execuções seguintes usarão o cache local.

## 🧠 Como Funciona (Deep Dive)

1. **Coleta de Dados**: Baixa uma lista de palavras em português e aplica filtros (remove palavras curtas, hífens, etc).
2. **Engenharia de Prompt**: Ao invés de vetorizar apenas a palavra (ex: "banco"), vetorizamos a sentença "o significado da palavra banco". Isso ajuda o modelo a focar na semântica da palavra em um contexto neutro.
3. **Geração de Embeddings**: O modelo transforma cada sentença em um vetor numérico de alta dimensionalidade.
4. **Cálculo de Distância**: Utilizamos a Similaridade de Cosseno para calcular o ângulo entre o vetor da palavra secreta e os vetores de todas as outras palavras do dicionário, gerando um ranking de proximidade.

## 🧪 Aprendizados e Limitações

Durante o desenvolvimento, realizei testes com scripts auxiliares `(ver_vizinhos.py)` e observei desafios interessantes na modelagem de linguagem atual:

- **Sintaxe vs. Semântica**: Mesmo utilizando modelos multilinguais robustos, o algoritmo ainda tende a aproximar palavras pela grafia (sintaxe) e não apenas pelo significado puro. Por exemplo, palavras com sufixos iguais tendem a ficar próximas, mesmo que não sejam sinônimos.
- **Importância do Contexto**: Modelos menores (distil) têm dificuldade em capturar relações óbvias (como "Banana" e "Fruta") sem um prompt auxiliar. A técnica de adicionar "o significado da palavra..." melhorou substancialmente a precisão.

## 📂 Estrutura do Projeto

- `main.py`: Código principal contendo a lógica do jogo e download de dados.
- `ver_vizinhos.py`: Script de análise para listar as N palavras mais próximas de um termo alvo (debug).
- `dados_contexto.pkl`: Arquivo de cache (gerado automaticamente) para acelerar a inicialização.
- `palavras.txt`: Arquivo texto com as palavras escolhidas para o jogo.

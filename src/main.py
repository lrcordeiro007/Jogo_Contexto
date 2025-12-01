import os
import pickle
import random
import requests
import time
from sentence_transformers import SentenceTransformer, util

ARQUIVO_CACHE = "dados_contexto.pkl"
URL_DICIONARIO = "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
MODELO_NOME = 'distiluse-base-multilingual-cased-v1'
ARQUIVO_ALVOS = "palavras.txt"  

def carregar_palavras_alvo():
    if not os.path.exists(ARQUIVO_ALVOS):
        print(f"Arquivo '{ARQUIVO_ALVOS}' não encontrado.")
        return []

    with open(ARQUIVO_ALVOS, 'r', encoding='utf-8') as f:
        palavras = [linha.strip().lower() for linha in f.readlines()]
    
    palavras = [p for p in palavras if p]
    return palavras

def baixar_e_processar_dados():
    print("--- INICIANDO ---")
    
    # 1. Baixar Dicionário
    try:
        r = requests.get(URL_DICIONARIO)
        todas_palavras = r.text.splitlines()
    except Exception as e:
        print(f"Erro ao baixar dicionário: {e}")
        return None, None

    # 2. Limpar Dicionário
    dicionario = []
    for p in todas_palavras:
        p = p.lower().strip()
        if len(p) > 2 and '-' not in p and p.isalpha():
            dicionario.append(p)
    
    dicionario = sorted(list(set(dicionario)))
    print(f"{len(dicionario)} palavras.")

    # 3. Gerar Embeddings
    model = SentenceTransformer(MODELO_NOME)
    start_time = time.time()
    dicionario_com_contexto = [f"o significado da palavra {p}" for p in dicionario]
    embeddings = model.encode(dicionario_com_contexto, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    
    tempo = time.time() - start_time
    print(f" concluída em {tempo:.2f} segundos.")

    # 4. Salvar Cache
    dados = {
        'dicionario': dicionario, 
        'embeddings': embeddings 
    }
    with open(ARQUIVO_CACHE, 'wb') as f:
        pickle.dump(dados, f)
    print(f" -> Dados salvos em '{ARQUIVO_CACHE}'.")
    
    return dicionario, embeddings

def carregar_dados():
    if os.path.exists(ARQUIVO_CACHE):
        with open(ARQUIVO_CACHE, 'rb') as f:
            dados = pickle.load(f)
        return dados['dicionario'], dados['embeddings']
    else:
        return baixar_e_processar_dados()

def jogar():
    dicionario, matriz_embeddings = carregar_dados()
    if not dicionario: return

    alvos_possiveis = carregar_palavras_alvo()
    
    candidatos_validos = [p for p in alvos_possiveis if p in dicionario]
    
    if not candidatos_validos:
        print(f"Erro: Nenhuma palavra do arquivo {ARQUIVO_ALVOS} existe no dicionário baixado.")
        return

    #palavra_secreta = random.choice(candidatos_validos)
    palavra_secreta = "banana"
    print(palavra_secreta)
    # Pegar o vetor da palavra secreta
    idx_secreta = dicionario.index(palavra_secreta)
    vetor_secreta = matriz_embeddings[idx_secreta]

    print("\n" + "="*40)
    print(f"🎮 JOGO CONTEXTO")
    print(f"Objetivo: Adivinhe a palavra secreta.")
    print(f"Dica: O número indica a distância. #1 é a vitória.")
    print("="*40)
    print("Calculando distâncias do dia...")
    
    todos_scores = util.cos_sim(vetor_secreta, matriz_embeddings)[0]
    
    ranking_global = []
    for i in range(len(dicionario)):
        score = todos_scores[i].item()
        ranking_global.append({'palavra': dicionario[i], 'score': score})
    
    ranking_global.sort(key=lambda x: x['score'], reverse=True)
    
    mapa_posicoes = {item['palavra']: i+1 for i, item in enumerate(ranking_global)}

    tentativas = 0
    historico_usuario = []

    while True:
        chute = input("\nDigite uma palavra: ").strip().lower()

        if chute == 'sair': break
        if chute == 'desisto': 
            print(f"A palavra era: {palavra_secreta}")
            break

        if chute not in mapa_posicoes:
            print(f" A palavra '{chute}' não está no dicionário.")
            continue

        tentativas += 1
        posicao = mapa_posicoes[chute]

        historico_usuario.append({'palavra': chute, 'posicao': posicao})
        historico_usuario.sort(key=lambda x: x['posicao'])
        
        print(f"\n--- Histórico ({tentativas}) ---")
        print(f"{'RANK':<10} {'PALAVRA':<20} {'STATUS'}")
        
        for item in historico_usuario:
            p = item['posicao']
            
            if p == 1:
                cor = "🏆"
                barra = "🟩🟩🟩🟩🟩 (Você acertou!)"
            elif p <= 300:
                cor = "🟢"
                barra = "🟩🟩🟩🟩⬜"
            elif p <= 1500:
                cor = "🟡"
                barra = "🟨🟨⬜⬜⬜"
            else:
                cor = "🔴"
                barra = "🟥⬜⬜⬜⬜"
            
            print(f"{cor} #{p:<8} {item['palavra']:<20} {barra}")

        if posicao == 1:
            print(f"\nPARABÉNS! A palavra era {palavra_secreta.upper()}")
            print(f"Você acertou em {tentativas} tentativas.")
            break

if __name__ == "__main__":
    jogar()
from sentence_transformers import util
import pickle

# 1. Carregar os dados que já processamos
print("Carregando dados...")
with open('dados_contexto.pkl', 'rb') as f:
    dados = pickle.load(f)
    dicionario = dados['dicionario']
    embeddings = dados['embeddings']

def ver_vizinhos(palavra_alvo, top_n=10):
    palavra_alvo = palavra_alvo.lower()
    
    if palavra_alvo not in dicionario:
        print(f"A palavra '{palavra_alvo}' não está no dicionário.")
        return

    # 2. Achar o vetor da palavra
    idx = dicionario.index(palavra_alvo)
    vetor_alvo = embeddings[idx]

    # 3. Calcular similaridade com TODO O RESTO
    scores = util.cos_sim(vetor_alvo, embeddings)[0]

    # 4. Criar lista de pares (score, palavra)
    resultados = []
    for i in range(len(dicionario)):
        # Ignora a própria palavra (score 1.0)
        if dicionario[i] == palavra_alvo:
            continue
            
        resultados.append({
            'palavra': dicionario[i], 
            'score': scores[i].item()
        })

    # 5. Ordenar e pegar os Top N
    resultados.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n--- Vizinhos de '{palavra_alvo.upper()}' ---")
    for i in range(top_n):
        item = resultados[i]
        print(f"#{i+1} {item['palavra']:<15} (Similaridade: {item['score']:.4f})")

# --- TESTE ---
ver_vizinhos("centro")
ver_vizinhos("computador")
"""
Word Mover's Distance (WMD) para português europeu.

Instalação:
    pip install gensim POT spacy nltk
    python -m spacy download pt_core_news_md

Embeddings (fastText, Common Crawl, 300 dimensões):
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz
    (descarregar e descomprimir para a pasta do script, ou deixar o
     código descarregar automaticamente com download_embeddings())
"""

import math
import os
import urllib.request
from pathlib import Path

import nltk
import spacy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# 0. Diretório de trabalho (auxiliares: downloads e caches)
# ---------------------------------------------------------------------------
# Tudo o que é ficheiro auxiliar (embeddings .vec.gz e cache .kv) fica aqui,
# em vez da raiz do projeto. O caminho é relativo a este script.
AUX_DIR = Path(__file__).resolve().parent / "aux_fold"
AUX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Recursos linguísticos
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)

# spaCy: modelo de português (tokenização + lematização)
nlp = spacy.load("pt_core_news_md", disable=["parser", "ner"])

# Stopwords: usar APENAS a lista do NLTK.
# A lista nlp.Defaults.stop_words inclui numerais ("sete", "seis", ...) e
# verbos muito comuns, o que degrada a WMD (frases diferentes ficavam iguais).
STOPWORDS_PT = set(stopwords.words("portuguese"))

# Palavras de negação/quantidade que convém manter para análise semântica.
# Removidas da lista de stopwords para contarem para a distância.
STOPWORDS_PT -= {"não", "nunca", "mais", "menos", "sem"}


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------
def preprocess(sentences, kv=None):
    """
    Minúsculas, remoção de pontuação e stopwords, e lematização.

    Se `kv` (KeyedVectors) for passado, usa o lema apenas quando este existe
    no vocabulário do modelo; caso contrário, recorre à forma original.
    Isto evita lemas inválidos gerados pelo spaCy (ex.: "contém" -> "contémr").
    """
    new_sentences = []
    for doc in nlp.pipe(s.lower() for s in sentences):
        tokens = []
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue

            form = tok.text.lower()
            lem = tok.lemma_.lower()

            # Stopwords: comparar forma E lema
            if form in STOPWORDS_PT or lem in STOPWORDS_PT:
                continue

            # Preferir o lema se existir no vocabulário; senão, a forma original.
            if kv is not None and lem in kv.key_to_index:
                tokens.append(lem)
            else:
                tokens.append(form)

        new_sentences.append(tokens)
    return new_sentences


# ---------------------------------------------------------------------------
# 3. Embeddings pré-treinados em português
# ---------------------------------------------------------------------------
EMB_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"
EMB_FILE = AUX_DIR / "cc.pt.300.vec.gz"
KV_FILE = AUX_DIR / "cc.pt.300.kv"   # cache nativo do gensim (mais rápido)


def download_embeddings():
    if not EMB_FILE.exists() and not KV_FILE.exists():
        print(f"A descarregar embeddings (~1 GB) para {EMB_FILE} ...")
        urllib.request.urlretrieve(EMB_URL, EMB_FILE)


# download_embeddings()

if KV_FILE.exists():
    print(f"A carregar embeddings do cache ({KV_FILE.name})...")
    model = KeyedVectors.load(str(KV_FILE))
else:
    print(f"A carregar embeddings de {EMB_FILE.name} (pode demorar)...")
    # limit=500_000 carrega só as palavras mais frequentes (poupa RAM e tempo)
    model = KeyedVectors.load_word2vec_format(
        str(EMB_FILE), binary=False, limit=500_000
    )
    print(f"A guardar cache em {KV_FILE.name} para próximas execuções...")
    model.save(str(KV_FILE))


# ---------------------------------------------------------------------------
# 4. Cálculo da distância
# ---------------------------------------------------------------------------
def wmd(s1, s2):
    t1, t2 = preprocess([s1, s2], kv=model)
    print(t1, t2)

    # Sem tokens (frases só com stopwords/pontuação): distância indefinida.
    if not t1 or not t2:
        return float("nan")

    d = model.wmdistance(t1, t2)

    # wmdistance devolve inf se todos os tokens forem OOV.
    if math.isinf(d):
        return float("inf")

    return d


# ---------------------------------------------------------------------------
# Conjunto de pares organizado por tipo de relação semântica.
# ---------------------------------------------------------------------------
pares = {
    # -----------------------------------------------------------------
    # 1. QUASE IGUAIS — difere exatamente 1 token (léxico controlado)
    # -----------------------------------------------------------------
    "quase_iguais": [
        (
            "Esta caixa contém sete cores diferentes",
            "Esta caixa contém seis cores diferentes",
        ),
        (
            "O menino comeu uma maçã vermelha ao pequeno-almoço",
            "O menino comeu uma maçã verde ao pequeno-almoço",
        ),
        (
            "A reunião foi marcada para as três da tarde na sala grande",
            "A reunião foi marcada para as quatro da tarde na sala grande",
        ),
        (
            "O restaurante italiano fica na rua principal e abre às sete",
            "O restaurante italiano fica na rua principal e abre às oito",
        ),
        (
            "O curso de matemática começa em setembro e dura seis meses",
            "O curso de matemática começa em outubro e dura seis meses",
        ),
        (
            "Havia vinte pessoas na sala de aula durante a apresentação",
            "Havia trinta pessoas na sala de aula durante a apresentação",
        ),
        (
            "O projeto custou cerca de quinhentos mil euros aos contribuintes",
            "O projeto custou cerca de oitocentos mil euros aos contribuintes",
        ),
    ],

    # -----------------------------------------------------------------
    # 2. PARÁFRASES — mesmo significado, léxico diferente (sinónimos)
    # -----------------------------------------------------------------
    "parafrases": [
        (
            "O carro do João avariou na estrada nacional",
            "O automóvel do João teve uma avaria na estrada nacional",
        ),
        (
            "A Maria comprou um livro novo sobre história medieval",
            "A Maria adquiriu uma obra recente acerca da história da Idade Média",
        ),
        (
            "Choveu intensamente durante toda a noite na cidade",
            "A chuva foi muito forte ao longo da noite na cidade",
        ),
        (
            "A biblioteca municipal da cidade vai reabrir na próxima segunda-feira "
            "depois de vários meses de obras de renovação",
            "Depois de meses de trabalhos de renovação, a biblioteca da cidade "
            "reabre na próxima segunda-feira",
        ),
    ],

    # -----------------------------------------------------------------
    # 3a. CONTRADIÇÕES POR NEGAÇÃO EXPLÍCITA
    #     (diferença = token "não"; a WMD deve dar distância > 0)
    # -----------------------------------------------------------------
    "contradicoes_negacao": [
        (
            "O gato está em cima da mesa da cozinha",
            "O gato não está em cima da mesa da cozinha",
        ),
        (
            "O João foi à reunião de manhã cedo",
            "O João não foi à reunião de manhã cedo",
        ),
        (
            "A Maria gosta de ler romances ao fim de semana",
            "A Maria não gosta de ler romances ao fim de semana",
        ),
    ],

    # -----------------------------------------------------------------
    # 3b. CONTRADIÇÕES POR ANTÓNIMO LEXICAL
    #     (limitação conhecida da WMD: antónimos ficam próximos no
    #      espaço distribucional, logo distâncias tendem a ser baixas)
    # -----------------------------------------------------------------
    "contradicoes_antonimo": [
        (
            "O preço do bilhete subiu bastante este ano",
            "O preço do bilhete desceu bastante este ano",
        ),
        (
            "O aluno respondeu corretamente a todas as perguntas do exame",
            "O aluno respondeu erradamente a todas as perguntas do exame",
        ),
        (
            "A temperatura da sala aumentou durante a tarde",
            "A temperatura da sala diminuiu durante a tarde",
        ),
    ],

    # -----------------------------------------------------------------
    # 4. MESMO TÓPICO, DETALHES DIFERENTES
    #     (partilham o assunto, mas não são paráfrases nem contradições)
    # -----------------------------------------------------------------
    "mesmo_topico": [
        (
            "A empresa portuguesa exporta vinho para a Alemanha e França",
            "A empresa portuguesa exporta vinho para a Espanha e Itália",
        ),
        (
            "O arco-íris tem sete cores",
            "Há um arco-íris lá fora",
        ),
        (
            "O restaurante novo da esquina serve comida italiana muito boa",
            "O restaurante novo da esquina serve comida italiana barata",
        ),
    ],

    # -----------------------------------------------------------------
    # 5. TÓPICOS DIFERENTES — devem dar as distâncias mais altas
    # -----------------------------------------------------------------
    "topicos_diferentes": [
        (
            "O arco-íris tem sete cores",
            "Uma maçã por dia mantém o médico afastado",
        ),
        (
            "A equipa de futebol ganhou o campeonato nacional este ano",
            "O cientista publicou um artigo sobre buracos negros na revista",
        ),
        (
            "O restaurante novo da esquina serve comida italiana muito boa",
            "A empresa de tecnologia lançou um novo telefone no mercado asiático",
        ),
        (
            "O governo anunciou um novo pacote de medidas para apoiar as famílias "
            "com dificuldades em pagar a renda da casa",
            "A seleção nacional de hóquei em patins venceu ontem o campeonato "
            "europeu numa final muito disputada",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Execução: imprimir as distâncias por grupo
# ---------------------------------------------------------------------------
for nome, lista in pares.items():
    print(f"\n=== {nome} ===")
    for a, b in lista:
        d = wmd(a, b)
        print(f"  distância = {d:.4f}")
        print("\n")
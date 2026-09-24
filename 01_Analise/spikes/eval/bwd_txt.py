"""
Word Mover's Distance (WMD) para português europeu.

Instalação:
    pip install gensim POT spacy nltk
    python -m spacy download pt_core_news_md

Embeddings (fastText, Common Crawl, 300 dimensões):
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz

Estrutura esperada de ficheiros:
    <script_dir>/samples/txt_<topico>_<indice>.txt
    Ex.: txt_00_001.txt, txt_00_002.txt, txt_01_001.txt, ...

Ficheiros auxiliares (downloads e caches) ficam em:
    <script_dir>/aux_fold/
"""

import math
import re
import urllib.request
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import nltk
import spacy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# 0. Diretórios de trabalho
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
AUX_DIR = SCRIPT_DIR / "aux_fold"
SAMPLES_DIR = SCRIPT_DIR / "samples"

AUX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Recursos linguísticos
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)

nlp = spacy.load("pt_core_news_md", disable=["parser", "ner"])

STOPWORDS_PT = set(stopwords.words("portuguese"))
# Manter negação/quantidade (contam para a distância).
STOPWORDS_PT -= {"não", "nunca", "mais", "menos", "sem"}


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------
def preprocess(sentences, kv=None):
    """
    Minúsculas, remoção de pontuação e stopwords, e lematização.

    Se `kv` for passado, usa o lema apenas quando este existe no vocabulário
    do modelo; caso contrário, recorre à forma original.
    """
    new_sentences = []
    for doc in nlp.pipe(s.lower() for s in sentences):
        tokens = []
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue

            form = tok.text.lower()
            lem = tok.lemma_.lower()

            if form in STOPWORDS_PT or lem in STOPWORDS_PT:
                continue

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
KV_FILE = AUX_DIR / "cc.pt.300.kv"


def download_embeddings():
    if not EMB_FILE.exists() and not KV_FILE.exists():
        print(f"A descarregar embeddings (~1 GB) para {EMB_FILE} ...")
        urllib.request.urlretrieve(EMB_URL, EMB_FILE)


download_embeddings()

if KV_FILE.exists():
    print(f"A carregar embeddings do cache ({KV_FILE.name})...")
    model = KeyedVectors.load(str(KV_FILE))
else:
    print(f"A carregar embeddings de {EMB_FILE.name} (pode demorar)...")
    model = KeyedVectors.load_word2vec_format(
        str(EMB_FILE), binary=False, limit=500_000
    )
    print(f"A guardar cache em {KV_FILE.name} para próximas execuções...")
    model.save(str(KV_FILE))


# ---------------------------------------------------------------------------
# 4. Cálculo da distância
# ---------------------------------------------------------------------------
def wmd_tokens(t1, t2):
    """WMD a partir de listas de tokens já pré-processados."""
    if not t1 or not t2:
        return float("nan")
    d = model.wmdistance(t1, t2)
    if math.isinf(d):
        return float("inf")
    return d


def wmd(s1, s2, verbose=False):
    """WMD a partir de duas frases em texto livre."""
    t1, t2 = preprocess([s1, s2], kv=model)
    if verbose:
        print(t1, t2)
    return wmd_tokens(t1, t2)


# ---------------------------------------------------------------------------
# 5. Carregar e agrupar ficheiros .txt por tópico
# ---------------------------------------------------------------------------
# Padrão: txt_<topico>_<indice>.txt
PATTERN = re.compile(r"^txt_(\d+)_(\d+)\.txt$", re.IGNORECASE)


def carregar_amostras(samples_dir: Path):
    """
    Lê todos os .txt com nome txt_<topico>_<indice>.txt.

    Devolve:
        documentos: dict[(topico, indice)] -> {"path": Path, "texto": str, "tokens": list}
        por_topico: dict[topico] -> list[(indice, chave)]
    """
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"Pasta de amostras não encontrada: {samples_dir}"
        )

    documentos = {}
    por_topico = defaultdict(list)

    for path in sorted(samples_dir.glob("*.txt")):
        m = PATTERN.match(path.name)
        if not m:
            print(f"  [aviso] nome fora do padrão, ignorado: {path.name}")
            continue

        topico = int(m.group(1))
        indice = int(m.group(2))
        texto = path.read_text(encoding="utf-8").strip()

        if not texto:
            print(f"  [aviso] ficheiro vazio, ignorado: {path.name}")
            continue

        chave = (topico, indice)
        documentos[chave] = {"path": path, "texto": texto, "tokens": None}
        por_topico[topico].append(chave)

    # Pré-processar em lote (mais rápido que um a um)
    chaves = list(documentos.keys())
    textos = [documentos[k]["texto"] for k in chaves]
    tokens_todos = preprocess(textos, kv=model)
    for k, toks in zip(chaves, tokens_todos):
        documentos[k]["tokens"] = toks

    return documentos, por_topico


# ---------------------------------------------------------------------------
# 6. Execução
# ---------------------------------------------------------------------------
def main():
    documentos, por_topico = carregar_amostras(SAMPLES_DIR)

    n_docs = len(documentos)
    n_topicos = len(por_topico)
    print(f"\nCarregados {n_docs} documentos em {n_topicos} tópicos.")
    for topico in sorted(por_topico):
        idxs = sorted(i for (t, i) in por_topico[topico])
        print(f"  tópico {topico:02d}: {len(idxs)} docs (índices {idxs})")

    # -----------------------------------------------------------------
    # 6a. Pares INTRA-tópico (mesmo tópico, devem ser próximos)
    # -----------------------------------------------------------------
    resultados_intra = []   # lista de (topico, chave1, chave2, distancia)
    for topico, chaves in sorted(por_topico.items()):
        for k1, k2 in combinations(sorted(chaves), 2):
            d = wmd_tokens(documentos[k1]["tokens"], documentos[k2]["tokens"])
            resultados_intra.append((topico, k1, k2, d))

    # -----------------------------------------------------------------
    # 6b. Pares INTER-tópico (tópicos diferentes, devem ser distantes)
    # -----------------------------------------------------------------
    resultados_inter = []   # lista de (topico1, topico2, chave1, chave2, distancia)
    topicos = sorted(por_topico.keys())
    for t1, t2 in combinations(topicos, 2):
        for k1 in sorted(por_topico[t1]):
            for k2 in sorted(por_topico[t2]):
                d = wmd_tokens(documentos[k1]["tokens"], documentos[k2]["tokens"])
                resultados_inter.append((t1, t2, k1, k2, d))

    # -----------------------------------------------------------------
    # 6c. Impressão detalhada
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PARES INTRA-TÓPICO (mesmo tópico)")
    print("=" * 70)
    for topico, k1, k2, d in resultados_intra:
        nome1, nome2 = documentos[k1]["path"].name, documentos[k2]["path"].name
        print(f"  [T{topico:02d}] {nome1}  vs  {nome2}  ->  {d:.4f}")

    print("\n" + "=" * 70)
    print("PARES INTER-TÓPICO (tópicos diferentes)")
    print("=" * 70)
    for t1, t2, k1, k2, d in resultados_inter:
        nome1, nome2 = documentos[k1]["path"].name, documentos[k2]["path"].name
        print(f"  [T{t1:02d} vs T{t2:02d}] {nome1}  vs  {nome2}  ->  {d:.4f}")

    # -----------------------------------------------------------------
    # 6d. Sumário
    # -----------------------------------------------------------------
    def _stats(valores):
        vals = [v for v in valores if not math.isnan(v) and not math.isinf(v)]
        if not vals:
            return None
        n = len(vals)
        media = sum(vals) / n
        return {
            "n": n,
            "min": min(vals),
            "max": max(vals),
            "media": media,
        }

    s_intra = _stats([d for _, _, _, d in resultados_intra])
    s_inter = _stats([d for _, _, _, _, d in resultados_inter])

    print("\n" + "=" * 70)
    print("SUMÁRIO")
    print("=" * 70)

    def _print_stats(nome, s):
        if s is None:
            print(f"  {nome}: sem pares válidos")
            return
        print(
            f"  {nome}: n={s['n']}  min={s['min']:.4f}  "
            f"média={s['media']:.4f}  max={s['max']:.4f}"
        )

    _print_stats("intra-tópico", s_intra)
    _print_stats("inter-tópico", s_inter)

    if s_intra and s_inter:
        gap = s_inter["media"] - s_intra["media"]
        print(f"\n  Diferença (inter - intra) = {gap:.4f}")
        if gap > 0:
            print("  [pass] A WMD separa os dois regimes (inter > intra).")
        else:
            print("  [warning]  A WMD NÃO está a separar os dois regimes.")


if __name__ == "__main__":
    main()
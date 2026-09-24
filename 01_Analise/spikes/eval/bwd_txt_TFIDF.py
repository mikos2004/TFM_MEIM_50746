"""
Word Mover's Distance (WMD) para português europeu, com filtragem por IDF.

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

import argparse
import math
import re
import urllib.request
from collections import Counter, defaultdict
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
STOPWORDS_PT -= {"não", "nunca", "mais", "menos", "sem"}

# Tokens que NUNCA devem ser removidos pelo filtro IDF, mesmo que sejam
# muito frequentes no corpus. São marcadores semânticos importantes
# (negação, quantificação) cuja perda degrada tarefas como deteção de
# contradições ou análise de sentimento.
TOKENS_PROTEGIDOS = frozenset({
    # negação
    "não", "nunca", "jamais", "sem", "nenhum", "nenhuma", "nada", "nem",
    # quantidade / grau
    "mais", "menos", "muito", "pouco", "bastante", "demasiado",
    # contraste
    "mas", "porém", "contudo", "todavia", "embora",
})


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------
def preprocess(sentences, kv=None):
    """Minúsculas, remoção de pontuação/stopwords e lematização."""
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
# 4. IDF a partir do corpus de amostras
# ---------------------------------------------------------------------------
class IDF:
    """
    Calcula IDF a partir de um conjunto de documentos já tokenizados.

    IDF(t) = log((N + 1) / (df(t) + 1)) + 1      (suavizado, à la sklearn)
    """

    def __init__(self, docs_tokens):
        self.n_docs = len(docs_tokens)
        df = Counter()
        for toks in docs_tokens:
            for t in set(toks):
                df[t] += 1
        self.df = df
        self.idf = {
            t: math.log((self.n_docs + 1) / (d + 1)) + 1.0
            for t, d in df.items()
        }

    def __getitem__(self, token):
        return self.idf.get(token, math.log(self.n_docs + 1) + 1.0)

    def filtrar(self, tokens, min_idf, protegidos=frozenset()):
        """
        Remove tokens com IDF abaixo de `min_idf`, EXCETO os que estão em
        `protegidos` (ex.: negação, quantificação), que são sempre mantidos.
        """
        return [
            t for t in tokens
            if self[t] >= min_idf or t in protegidos
        ]


# ---------------------------------------------------------------------------
# 5. Cálculo da distância
# ---------------------------------------------------------------------------
def wmd_tokens(t1, t2):
    if not t1 or not t2:
        return float("nan")
    d = model.wmdistance(t1, t2)
    if math.isinf(d):
        return float("inf")
    return d


# ---------------------------------------------------------------------------
# 6. Carregar e agrupar ficheiros .txt por tópico
# ---------------------------------------------------------------------------
PATTERN = re.compile(r"^txt_(\d+)_(\d+)\.txt$", re.IGNORECASE)


def carregar_amostras(samples_dir: Path):
    if not samples_dir.exists():
        raise FileNotFoundError(f"Pasta de amostras não encontrada: {samples_dir}")

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

    chaves = list(documentos.keys())
    textos = [documentos[k]["texto"] for k in chaves]
    tokens_todos = preprocess(textos, kv=model)
    for k, toks in zip(chaves, tokens_todos):
        documentos[k]["tokens"] = toks

    return documentos, por_topico


# ---------------------------------------------------------------------------
# 7. Execução
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="WMD com filtragem IDF e proteção de tokens semânticos."
    )
    p.add_argument(
        "--min-idf", type=float, default=1.5,
        help="Limiar de IDF (default: 1.5). Tokens abaixo são removidos, "
             "exceto se estiverem na lista de protegidos.",
    )
    p.add_argument(
        "--samples-dir", type=Path, default=SAMPLES_DIR,
        help=f"Pasta com os .txt (default: {SAMPLES_DIR})",
    )
    p.add_argument(
        "--no-protecao", action="store_true",
        help="Desativa a proteção de tokens de negação/quantificação.",
    )
    return p.parse_args()


def main(min_idf=1.5, samples_dir=SAMPLES_DIR, usar_protecao=True):
    protegidos = TOKENS_PROTEGIDOS if usar_protecao else frozenset()

    documentos, por_topico = carregar_amostras(samples_dir)

    n_docs = len(documentos)
    n_topicos = len(por_topico)
    print(f"\nCarregados {n_docs} documentos em {n_topicos} tópicos.")
    for topico in sorted(por_topico):
        idxs = sorted(i for (t, i) in por_topico[topico])
        print(f"  tópico {topico:02d}: {len(idxs)} docs (índices {idxs})")

    # -----------------------------------------------------------------
    # 7a. IDF
    # -----------------------------------------------------------------
    todos_tokens = [documentos[k]["tokens"] for k in documentos]
    idf = IDF(todos_tokens)

    print(f"\nIDF calculado sobre {idf.n_docs} documentos "
          f"({len(idf.df)} tokens únicos).")

    # Mostrar tokens que SERIAM filtrados mas estão protegidos
    protegidos_presentes = sorted(
        t for t in idf.df if t in protegidos and idf[t] < min_idf
    )
    if protegidos_presentes:
        print("\n  Tokens protegidos que seriam filtrados pelo IDF:")
        for t in protegidos_presentes:
            print(f"    {t:<20} df={idf.df[t]:>2}  idf={idf[t]:.3f}")

    # Diagnóstico: 20 tokens mais comuns
    mais_comuns = sorted(idf.idf.items(), key=lambda x: x[1])[:20]
    print("\n  Tokens mais comuns no corpus (IDF baixo):")
    for t, v in mais_comuns:
        marca = "  [PROTEGIDO]" if t in protegidos else ""
        print(f"    {t:<20} df={idf.df[t]:>2}  idf={v:.3f}{marca}")

    # Aplicar filtro com proteção
    tokens_antes = sum(len(d["tokens"]) for d in documentos.values())
    for k in documentos:
        documentos[k]["tokens_filtrados"] = idf.filtrar(
            documentos[k]["tokens"], min_idf, protegidos=protegidos
        )
    tokens_depois = sum(len(d["tokens_filtrados"]) for d in documentos.values())

    print(f"\n  MIN_IDF = {min_idf}  |  proteção = {'ON' if usar_protecao else 'OFF'}")
    print(f"  Tokens totais: {tokens_antes} -> {tokens_depois} "
          f"({100 * tokens_depois / tokens_antes:.1f}%)")

    # -----------------------------------------------------------------
    # 7b. Pares intra-tópico
    # -----------------------------------------------------------------
    resultados_intra = []
    for topico, chaves in sorted(por_topico.items()):
        for k1, k2 in combinations(sorted(chaves), 2):
            d = wmd_tokens(
                documentos[k1]["tokens_filtrados"],
                documentos[k2]["tokens_filtrados"],
            )
            resultados_intra.append((topico, k1, k2, d))

    # -----------------------------------------------------------------
    # 7c. Pares inter-tópico
    # -----------------------------------------------------------------
    resultados_inter = []
    topicos = sorted(por_topico.keys())
    for t1, t2 in combinations(topicos, 2):
        for k1 in sorted(por_topico[t1]):
            for k2 in sorted(por_topico[t2]):
                d = wmd_tokens(
                    documentos[k1]["tokens_filtrados"],
                    documentos[k2]["tokens_filtrados"],
                )
                resultados_inter.append((t1, t2, k1, k2, d))

    # -----------------------------------------------------------------
    # 7d. Impressão
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
    # 7e. Sumário
    # -----------------------------------------------------------------
    def _stats(valores):
        vals = [v for v in valores if not math.isnan(v) and not math.isinf(v)]
        if not vals:
            return None
        n = len(vals)
        return {
            "n": n,
            "min": min(vals),
            "max": max(vals),
            "media": sum(vals) / n,
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

        # Verificação de sobreposição
        sobreposicao = s_intra["max"] >= s_inter["min"]
        if sobreposicao:
            print(f"  [warning] Sobreposição: max(intra)={s_intra['max']:.4f} "
                  f">= min(inter)={s_inter['min']:.4f}")
        else:
            print(f"  [pass] Sem sobreposição: max(intra)={s_intra['max']:.4f} "
                  f"< min(inter)={s_inter['min']:.4f}")

        if gap > 0:
            print("  [pass] A WMD separa os dois regimes (inter > intra).")
        else:
            print("  [warning] A WMD NÃO está a separar os dois regimes.")


if __name__ == "__main__":
    args = parse_args()
    main(
        min_idf=args.min_idf,
        samples_dir=args.samples_dir,
        usar_protecao=not args.no_protecao,
    )
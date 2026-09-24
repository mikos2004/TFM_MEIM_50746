"""
Comparação de abordagens WMD para português europeu.

Corre três variantes sobre o mesmo corpus de amostras:
    A) WMD pura (sem IDF, sem proteção)
    B) WMD + IDF (sem proteção)
    C) WMD + IDF + proteção de negação/quantidade

Produz:
    - Tabela comparativa no terminal
    - Boxplot intra vs inter para cada variante
    - Scatter das médias por tópico
    - CSV com todos os pares de cada variante

Uso:
    python compare_wmd.py
    python compare_wmd.py --min-idf 1.5 --out-dir resultados
"""

import argparse
import csv
import math
import re
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend não interativo (guarda em ficheiro)
import matplotlib.pyplot as plt
import nltk
import numpy as np
import spacy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# 0. Diretórios
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
AUX_DIR = SCRIPT_DIR / "aux_fold"
SAMPLES_DIR = SCRIPT_DIR / "samples"
OUT_DIR = SCRIPT_DIR / "resultados"

AUX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Recursos linguísticos
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)

nlp = spacy.load("pt_core_news_md", disable=["parser", "ner"])

STOPWORDS_PT = set(stopwords.words("portuguese"))
STOPWORDS_PT -= {"não", "nunca", "mais", "menos", "sem"}

TOKENS_PROTEGIDOS = frozenset({
    "não", "nunca", "jamais", "sem", "nenhum", "nenhuma", "nada", "nem",
    "mais", "menos", "muito", "pouco", "bastante", "demasiado",
    "mas", "porém", "contudo", "todavia", "embora",
})

# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------
def preprocess(sentences, kv=None):
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
# 3. Embeddings
# ---------------------------------------------------------------------------
EMB_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"
EMB_FILE = AUX_DIR / "cc.pt.300.vec.gz"
KV_FILE = AUX_DIR / "cc.pt.300.kv"


def carregar_modelo():
    if not EMB_FILE.exists() and not KV_FILE.exists():
        print(f"A descarregar embeddings (~1 GB) para {EMB_FILE} ...")
        urllib.request.urlretrieve(EMB_URL, EMB_FILE)

    if KV_FILE.exists():
        print(f"A carregar embeddings do cache ({KV_FILE.name})...")
        return KeyedVectors.load(str(KV_FILE))
    print(f"A carregar embeddings de {EMB_FILE.name} (pode demorar)...")
    model = KeyedVectors.load_word2vec_format(
        str(EMB_FILE), binary=False, limit=500_000
    )
    print(f"A guardar cache em {KV_FILE.name} ...")
    model.save(str(KV_FILE))
    return model


# ---------------------------------------------------------------------------
# 4. IDF
# ---------------------------------------------------------------------------
class IDF:
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
        return [t for t in tokens if self[t] >= min_idf or t in protegidos]


# ---------------------------------------------------------------------------
# 5. WMD
# ---------------------------------------------------------------------------
def wmd_tokens(model, t1, t2):
    if not t1 or not t2:
        return float("nan")
    d = model.wmdistance(t1, t2)
    return float("inf") if math.isinf(d) else d


# ---------------------------------------------------------------------------
# 6. Carregar amostras
# ---------------------------------------------------------------------------
PATTERN = re.compile(r"^txt_(\d+)_(\d+)\.txt$", re.IGNORECASE)


def carregar_amostras(model, samples_dir: Path):
    if not samples_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {samples_dir}")

    documentos = {}
    por_topico = defaultdict(list)

    for path in sorted(samples_dir.glob("*.txt")):
        m = PATTERN.match(path.name)
        if not m:
            print(f"  [aviso] ignorado: {path.name}")
            continue
        topico, indice = int(m.group(1)), int(m.group(2))
        texto = path.read_text(encoding="utf-8").strip()
        if not texto:
            continue
        chave = (topico, indice)
        documentos[chave] = {"path": path, "texto": texto, "tokens": None}
        por_topico[topico].append(chave)

    chaves = list(documentos.keys())
    textos = [documentos[k]["texto"] for k in chaves]
    for k, toks in zip(chaves, preprocess(textos, kv=model)):
        documentos[k]["tokens"] = toks

    return documentos, por_topico


# ---------------------------------------------------------------------------
# 7. Variantes
# ---------------------------------------------------------------------------
def aplicar_variante(nome, documentos, idf, min_idf):
    """
    Devolve um dict chave -> tokens, de acordo com a variante:
        "A_wmd_pura"   : tokens originais (sem IDF, sem proteção)
        "B_wmd_idf"    : tokens filtrados por IDF (sem proteção)
        "C_wmd_idf_prot": tokens filtrados por IDF com proteção
    """
    resultado = {}
    for k, doc in documentos.items():
        if nome == "A_wmd_pura":
            resultado[k] = doc["tokens"]
        elif nome == "B_wmd_idf":
            resultado[k] = idf.filtrar(doc["tokens"], min_idf, frozenset())
        elif nome == "C_wmd_idf_prot":
            resultado[k] = idf.filtrar(doc["tokens"], min_idf, TOKENS_PROTEGIDOS)
        else:
            raise ValueError(nome)
    return resultado


def calcular_pares(model, tokens_por_doc, por_topico):
    """Devolve (intra, inter) como listas de dicts."""
    intra, inter = [], []

    for topico, chaves in sorted(por_topico.items()):
        for k1, k2 in combinations(sorted(chaves), 2):
            d = wmd_tokens(model, tokens_por_doc[k1], tokens_por_doc[k2])
            intra.append({
                "tipo": "intra", "t1": topico, "t2": topico,
                "doc1": k1, "doc2": k2, "dist": d,
            })

    topicos = sorted(por_topico.keys())
    for t1, t2 in combinations(topicos, 2):
        for k1 in sorted(por_topico[t1]):
            for k2 in sorted(por_topico[t2]):
                d = wmd_tokens(model, tokens_por_doc[k1], tokens_por_doc[k2])
                inter.append({
                    "tipo": "inter", "t1": t1, "t2": t2,
                    "doc1": k1, "doc2": k2, "dist": d,
                })

    return intra, inter


def stats(valores):
    vals = [v for v in valores if not math.isnan(v) and not math.isinf(v)]
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "media": sum(vals) / len(vals),
    }


# ---------------------------------------------------------------------------
# 8. Visualização
# ---------------------------------------------------------------------------
def plot_boxplot(por_variante, out_path):
    """
    por_variante: dict[nome_variante] -> {"intra": [..], "inter": [..]}
    """
    nomes = list(por_variante.keys())
    fig, axes = plt.subplots(1, len(nomes), figsize=(5 * len(nomes), 5),
                             sharey=True)

    if len(nomes) == 1:
        axes = [axes]

    for ax, nome in zip(axes, nomes):
        dados = por_variante[nome]
        ax.boxplot(
            [dados["intra"], dados["inter"]],
            tick_labels=["intra", "inter"],
            patch_artist=True,
            boxprops=dict(facecolor="#cfe2f3"),
            medianprops=dict(color="black"),
        )
        ax.set_title(nome.replace("_", " "), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Linhas de referência: médias
        mi = np.mean(dados["intra"])
        mr = np.mean(dados["inter"])
        ax.axhline(mi, color="C0", ls="--", lw=1, alpha=0.7)
        ax.axhline(mr, color="C1", ls="--", lw=1, alpha=0.7)
        ax.set_ylabel("Distância WMD")

    fig.suptitle("WMD intra vs inter-tópico por variante", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> guardado: {out_path}")


def plot_gap_barras(por_variante, out_path):
    """Barra com o gap (inter média - intra média) por variante."""
    nomes = list(por_variante.keys())
    gaps = [
        np.mean(por_variante[n]["inter"]) - np.mean(por_variante[n]["intra"])
        for n in nomes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    cores = ["#6aa84f" if g > 0 else "#cc0000" for g in gaps]
    ax.bar(range(len(nomes)), gaps, color=cores)
    ax.set_xticks(range(len(nomes)))
    ax.set_xticklabels([n.replace("_", "\n") for n in nomes], fontsize=9)
    ax.set_ylabel("Gap (inter média - intra média)")
    ax.set_title("Separação intra/inter por variante")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3)

    for i, g in enumerate(gaps):
        ax.text(i, g + 0.005, f"{g:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> guardado: {out_path}")


def plot_sobreposicao(por_variante, out_path):
    """
    Mostra visualmente se há sobreposição: para cada variante, desenha
    a gama [min, max] de intra e inter como linhas horizontais.
    """
    nomes = list(por_variante.keys())
    fig, ax = plt.subplots(figsize=(10, 4 + 0.4 * len(nomes)))

    for i, nome in enumerate(nomes):
        d = por_variante[nome]
        y = len(nomes) - 1 - i

        # Gama intra
        ax.plot([min(d["intra"]), max(d["intra"])], [y + 0.15, y + 0.15],
                color="C0", lw=6, solid_capstyle="butt",
                label="intra" if i == 0 else None)
        # Gama inter
        ax.plot([min(d["inter"]), max(d["inter"])], [y - 0.15, y - 0.15],
                color="C1", lw=6, solid_capstyle="butt",
                label="inter" if i == 0 else None)

        # Marcar sobreposição
        sobrepos = max(d["intra"]) >= min(d["inter"])
        ax.text(max(max(d["intra"]), max(d["inter"])) + 0.02,
                y, "SOBREPOSIÇÃO" if sobrepos else "OK",
                color="red" if sobrepos else "green", fontsize=9,
                va="center", fontweight="bold")

    ax.set_yticks(range(len(nomes)))
    ax.set_yticklabels(list(reversed([n.replace("_", " ") for n in nomes])))
    ax.set_xlabel("Distância WMD")
    ax.set_title("Gama intra vs inter por variante (sobreposição?)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> guardado: {out_path}")


def plot_media_por_topico(por_variante, por_topico, out_path):
    """
    Para cada variante, barra da média intra-tópico por tópico.
    """
    nomes = list(por_variante.keys())
    topicos = sorted(por_topico.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(topicos))
    largura = 0.8 / len(nomes)

    for i, nome in enumerate(nomes):
        medias = []
        for t in topicos:
            vals = [
                d["dist"] for d in por_variante[nome]["intra_raw"]
                if d["t1"] == t
            ]
            medias.append(np.mean(vals) if vals else np.nan)
        ax.bar(x + i * largura - 0.4 + largura / 2, medias, largura,
               label=nome.replace("_", " "))

    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t:02d}" for t in topicos])
    ax.set_ylabel("WMD média intra-tópico")
    ax.set_title("Coesão intra-tópico por tópico e variante")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> guardado: {out_path}")


# ---------------------------------------------------------------------------
# 9. CSV
# ---------------------------------------------------------------------------
def exportar_csv(por_variante, out_path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variante", "tipo", "topico_a", "topico_b",
                    "doc_a", "doc_b", "distancia"])
        for nome, dados in por_variante.items():
            for d in dados["intra_raw"]:
                w.writerow([
                    nome, "intra", d["t1"], d["t2"],
                    f"txt_{d['doc1'][0]:02d}_{d['doc1'][1]:03d}",
                    f"txt_{d['doc2'][0]:02d}_{d['doc2'][1]:03d}",
                    f"{d['dist']:.4f}",
                ])
            for d in dados["inter_raw"]:
                w.writerow([
                    nome, "inter", d["t1"], d["t2"],
                    f"txt_{d['doc1'][0]:02d}_{d['doc1'][1]:03d}",
                    f"txt_{d['doc2'][0]:02d}_{d['doc2'][1]:03d}",
                    f"{d['dist']:.4f}",
                ])
    print(f"  -> guardado: {out_path}")


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-idf", type=float, default=1.5)
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main(min_idf, samples_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    model = carregar_modelo()
    documentos, por_topico = carregar_amostras(model, samples_dir)

    n_docs = len(documentos)
    n_topicos = len(por_topico)
    print(f"\nCarregados {n_docs} documentos em {n_topicos} tópicos.")

    # IDF
    idf = IDF([documentos[k]["tokens"] for k in documentos])

    # Variantes
    variantes = ["A_wmd_pura", "B_wmd_idf", "C_wmd_idf_prot"]
    resultados = {}

    for nome in variantes:
        print(f"\n>>> A calcular {nome} ...")
        tokens_por_doc = aplicar_variante(nome, documentos, idf, min_idf)
        intra, inter = calcular_pares(model, tokens_por_doc, por_topico)
        resultados[nome] = {
            "intra": [d["dist"] for d in intra],
            "inter": [d["dist"] for d in inter],
            "intra_raw": intra,
            "inter_raw": inter,
        }

    # -----------------------------------------------------------------
    # Tabela comparativa
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABELA COMPARATIVA")
    print("=" * 78)
    header = f"{'variante':<18} {'intra média':>12} {'inter média':>12} " \
             f"{'gap':>8} {'max(intra)':>12} {'min(inter)':>12} {'sobrep.?':>10}"
    print(header)
    print("-" * len(header))

    for nome in variantes:
        d = resultados[nome]
        si = stats(d["intra"])
        sr = stats(d["inter"])
        gap = sr["media"] - si["media"]
        sobrep = "SIM" if si["max"] >= sr["min"] else "não"
        print(f"{nome:<18} {si['media']:>12.4f} {sr['media']:>12.4f} "
              f"{gap:>8.4f} {si['max']:>12.4f} {sr['min']:>12.4f} "
              f"{sobrep:>10}")

    # -----------------------------------------------------------------
    # Gráficos
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("A GERAR GRÁFICOS")
    print("=" * 78)

    plot_boxplot(resultados, out_dir / "boxplot_intra_inter.png")
    plot_gap_barras(resultados, out_dir / "gap_por_variante.png")
    plot_sobreposicao(resultados, out_dir / "sobreposicao.png")
    plot_media_por_topico(resultados, por_topico,
                          out_dir / "media_intra_por_topico.png")

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("A EXPORTAR CSV")
    print("=" * 78)
    exportar_csv(resultados, out_dir / "pares_wmd.csv")

    print("\nFeito. Resultados em:", out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args.min_idf, args.samples_dir, args.out_dir)
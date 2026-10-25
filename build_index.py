import re
import shutil
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR   = Path(__file__).parent
CSV_PATH   = BASE_DIR / "data" / "flipkart_com-ecommerce_sample.csv"
INDEX_PATH = BASE_DIR / "flipkart_faiss_index"

# ── 1. Chargement et nettoyage ──────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[["product_name", "description", "product_specifications",
         "retail_price", "discounted_price", "brand",
         "product_rating", "product_category_tree"]]

df = df.dropna(subset=["product_name", "description"])
df["brand"] = df["brand"].fillna("Unknown")

df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
df["retail_price"]     = pd.to_numeric(df["retail_price"], errors="coerce")

def clean_specs(txt):
    txt = str(txt).replace("=>", ":")
    txt = re.sub(r'[\{\}\[\]"]', "", txt)
    return txt.strip()

df["specs"] = df["product_specifications"].apply(clean_specs)

def simplifier_categorie(tree):
    parts = [p.strip(' ["]') for p in str(tree).split(">>")]
    return " > ".join(parts[:3])

df["categorie"] = df["product_category_tree"].apply(simplifier_categorie)
df = df.drop_duplicates(subset=["product_name", "description"])
print(f"Produits : {len(df)}")

# ── 2. UN document par produit ──────────────────────────────────
documents = []
for _, r in df.iterrows():
    texte = (
        f"Product: {r['product_name']}\n"
        f"Brand: {r['brand']}\n"
        f"Category: {r['categorie']}\n"
        f"Price: {r['discounted_price']}\n"
        f"Description: {str(r['description'])[:400]}\n"
        f"Specs: {r['specs'][:250]}"
    )
    documents.append(Document(
        page_content=texte,
        metadata={
            "product_name": str(r["product_name"]),
            "brand": str(r["brand"]),
            "price": None if pd.isna(r["discounted_price"]) else float(r["discounted_price"]),
            "retail_price": None if pd.isna(r["retail_price"]) else float(r["retail_price"]),
            "rating": str(r["product_rating"]),
            "categorie": r["categorie"],
        },
    ))
print(f"Documents : {len(documents)}")

# ── 3. Embeddings + FAISS ───────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

if INDEX_PATH.exists():
    shutil.rmtree(INDEX_PATH)

print("Création de l'index (quelques minutes)...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(str(INDEX_PATH))
print("Index sauvegardé ✅")
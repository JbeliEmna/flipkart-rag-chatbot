import os
import re
import json
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR   = Path(__file__).parent
INDEX_PATH = BASE_DIR / "flipkart_faiss_index"

# ── Configuration ───────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env", override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY introuvable (fichier .env ou secrets)")
if not INDEX_PATH.exists():
    raise FileNotFoundError("Index introuvable. Lance d'abord : python build_index.py")

# ── Chargement de l'index ───────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
)
TOUS_LES_DOCS = list(vectorstore.docstore._dict.values())

# ── LLM ─────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.1,
    max_tokens=2000,
)

NOT_FOUND = "I could not find a matching product in our catalog."
MSG_ACCUEIL = ("Hello! I'm the Flipkart shopping assistant. Ask me about a product, "
               "a budget or a brand, e.g. 'wrist watch for men under 1500'.")

# ── Prompts ─────────────────────────────────────────────────────
PROMPT_SEARCH = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Flipkart shopping assistant.
Answer ONLY using the products listed below. Never invent products, prices or specs.
Reply in the same language as the customer's message.
Start with one short sentence, then list the relevant products as bullets:
- **Product name** — Rs. price
If a price is "not listed", say so.
If NONE of the products is related to the request, start your reply with the token
[NOT_FOUND] followed by one short polite sentence saying nothing matching was found.

PRODUCTS:
{context}"""),
    ("human", "{question}"),
])

PROMPT_BROWSE = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Flipkart shopping assistant.
Below is an overview of the catalog related to the customer's question:
the number of matching products per category, with example names.
Using ONLY this overview: say yes, list the main categories with their counts,
mention one or two example products per category, then invite the customer to ask
for a specific item, budget or brand.
These counts only include products whose name or category explicitly mention the
keyword, so present them as "at least" figures (e.g. "at least 25 jewellery items
labelled for women").
Reply in the same language as the customer's message.

OVERVIEW:
{overview}"""),
    ("human", "{question}"),
])

# ── Brique 1 : budget et marque (filets de sécurité par regex) ──
def extraire_budget(question):
    m = re.search(
        r"(?:under|below|less than|within|max(?:imum)?|moins de)\s*(?:rs\.?|₹|inr)?\s*([\d,]+)",
        question, re.IGNORECASE,
    )
    return float(m.group(1).replace(",", "")) if m else None

MARQUES = sorted(
    {d.metadata["brand"].lower() for d in TOUS_LES_DOCS
     if d.metadata["brand"] not in ("Unknown", "nan")},
    key=len, reverse=True,
)
MARQUES_SET = set(MARQUES)

def extraire_marque(question):
    q = question.lower()
    for m in MARQUES:
        if len(m) > 2 and re.search(rf"\b{re.escape(m)}\b", q):
            return m
    return None

# ── Brique 2 : comprendre le message (intention + filtres) ──────
def comprendre(question, historique):
    defaut = {"intent": "search", "query": question, "keyword": "",
              "max_price": None, "brand": "", "reply": ""}
    contexte = ""
    if historique:
        q, r = historique[-1]
        contexte = f"Previous question: {q}\nPrevious answer: {r[:300]}\n\n"

    prompt = (
        "You analyse messages sent to a Flipkart shopping assistant. "
        "Return ONLY a JSON object with these keys:\n"
        '- "intent": "chitchat" (greeting, thanks, who are you, small talk), '
        '"browse" (broad question about what the store sells or about a whole category '
        'or audience, e.g. "do you have products for women", "what do you sell"), '
        'or "search" (looking for a specific kind of product).\n'
        '- "query": a standalone English search query. Resolve words like it/its/that one '
        "using the previous exchange. Translate to English if needed.\n"
        '- "keyword": for browse only, ONE short English word for the category or audience '
        '(women, men, kids, watch, jewellery...), otherwise "".\n'
        '- "max_price": a number if the customer gives a budget, otherwise null.\n'
        '- "brand": a brand name if mentioned, otherwise "".\n'
        '- "reply": for chitchat only, a short friendly answer (max 2 sentences) in the '
        "customer's language inviting them to ask about products, otherwise an empty string.\n\n"
        f"{contexte}Customer message: {question}\nJSON:"
    )
    try:
        texte = llm.invoke(prompt).content
        brut = re.search(r"\{.*\}", texte, re.DOTALL)
        data = json.loads(brut.group(0))
        for cle in defaut:
            if data.get(cle) is not None:
                defaut[cle] = data[cle]
        if defaut["intent"] not in ("chitchat", "browse", "search"):
            defaut["intent"] = "search"
        if not str(defaut["query"]).strip():
            defaut["query"] = question
    except Exception:
        pass   # en cas de problème, on retombe sur une recherche classique
    return defaut

# ── Brique 3 : aperçu du catalogue (questions larges) ───────────
def texte_apercu(mot_cle):
    mot = (mot_cle or "").lower().strip()
    motif = re.compile(rf"\b{re.escape(mot)}(?:e?s|'s|’s)?\b") if mot else None
    compteur, exemples = Counter(), {}
    for d in TOUS_LES_DOCS:
        m = d.metadata
        if motif and not motif.search((m["product_name"] + " " + m["categorie"]).lower()):
            continue
        cat = m["categorie"].split(" > ")[0]
        compteur[cat] += 1
        exemples.setdefault(cat, [])
        nom_court = m["product_name"][:70]
        if len(exemples[cat]) < 2 and nom_court not in exemples[cat]:
            exemples[cat].append(nom_court)
    if not compteur:
        return None
    etiquette = f"mentioning '{mot}' in their name or category" if mot else "in the whole catalog"
    lignes = [f"Total products {etiquette}: {sum(compteur.values())}"]
    for cat, n in compteur.most_common(6):
        lignes.append(f"- {cat}: {n} products (e.g. {'; '.join(exemples[cat])})")
    return "\n".join(lignes)

# ── Brique 4 : recherche + filtres + dédoublonnage ──────────────
def rechercher(question, k_final=5, budget=None, marque=None):
    try:
        budget = float(budget) if budget not in (None, "") else extraire_budget(question)
    except (TypeError, ValueError):
        budget = extraire_budget(question)

    marque = (marque or "").lower().strip()
    if marque not in MARQUES_SET:
        marque = extraire_marque(question)

    candidats = vectorstore.similarity_search(question, k=300 if marque else 80)

    def filtrer(avec_marque):
        vus, res = set(), []
        for doc in candidats:
            nom, prix = doc.metadata["product_name"], doc.metadata["price"]
            if nom in vus:
                continue
            if budget is not None and (prix is None or prix > budget):
                continue
            if avec_marque and marque and doc.metadata["brand"].lower() != marque:
                continue
            vus.add(nom)
            res.append(doc)
            if len(res) == k_final:
                break
        return res

    resultats = filtrer(True)
    if not resultats and marque:
        resultats = filtrer(False)
    return resultats

# ── Fonction principale ─────────────────────────────────────────
def repondre(question, historique):
    info = comprendre(question, historique)
    requete = info["query"]

    # 1. Salutations et petites phrases
    if info["intent"] == "chitchat":
        return (info["reply"] or MSG_ACCUEIL), [], requete

    # 2. Question large sur le catalogue
    if info["intent"] == "browse":
        apercu = texte_apercu(info["keyword"])
        if apercu:
            messages = PROMPT_BROWSE.format_messages(overview=apercu, question=question)
            return llm.invoke(messages).content.strip(), [], requete
        # aucun produit pour ce mot-clé : on tente une recherche classique

    # 3. Recherche de produits
    docs = rechercher(requete, budget=info["max_price"], marque=info["brand"])
    if not docs:
        return NOT_FOUND, [], requete

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    context = context.replace("Price: nan", "Price: not listed")
    messages = PROMPT_SEARCH.format_messages(
        context=context,
        question=f"Customer message: {question}\nSearch query: {requete}",
    )
    reponse = llm.invoke(messages).content.strip()

    if reponse.startswith("[NOT_FOUND]"):
        return reponse.replace("[NOT_FOUND]", "").strip() or NOT_FOUND, [], requete
    return reponse, docs, requete

# ── Mode terminal ───────────────────────────────────────────────
if __name__ == "__main__":
    hist = []
    print("CHATBOT PRÊT — tape 'quit' pour quitter")
    while True:
        q = input("\nToi : ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        rep, docs, requete = repondre(q, hist)
        hist.append((q, rep))
        if requete != q:
            print(f"\n(recherche : {requete})")
        print(f"\nA : {rep}")
        if docs:
            print("\nSources :")
            for d in docs:
                prix = d.metadata["price"]
                prix_txt = f"Rs.{prix}" if prix is not None else "price not listed"
                print(f"  • {d.metadata['product_name'][:55]} — {prix_txt}")
import streamlit as st
from chatbot import repondre

st.set_page_config(page_title="Flipkart AI Assistant", page_icon="🛒", layout="wide")

# ── État de session : propre à chaque visiteur ──────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []     # pour l'affichage : role, content, sources
if "historique" not in st.session_state:
    st.session_state.historique = []   # mémoire envoyée au LLM : (question, réponse)

# ── Affichage des fiches produits ───────────────────────────────
def afficher_sources(sources):
    if not sources:
        return
    with st.expander(f"📦 {len(sources)} produits utilisés", expanded=True):
        cols = st.columns(len(sources))
        for col, meta in zip(cols, sources):
            with col, st.container(border=True):
                st.markdown(f"**{meta['product_name'][:60]}**")
                st.caption(f"{meta['brand']} · {meta['categorie']}")
                prix, ancien = meta["price"], meta["retail_price"]
                if prix is not None:
                    st.markdown(f"### ₹{prix:,.0f}")
                if ancien and prix and ancien > prix:
                    remise = round(100 * (ancien - prix) / ancien)
                    st.caption(f"~~₹{ancien:,.0f}~~  (-{remise}%)")
                st.caption(f"⭐ {meta['rating']}")

# ── Barre latérale ──────────────────────────────────────────────
with st.sidebar:
    st.header("À propos")
    st.write("Chatbot RAG sur ~17 500 produits Flipkart : mode, bijoux, montres, accessoires mobiles, maison...")
    st.caption("FAISS · all-MiniLM-L6-v2 · GPT-OSS 20B (Groq) · LangChain")

    st.subheader("Essaie ces questions")
    exemples = [
        "wireless bluetooth headphones under 1000",
        "wrist watch for men under 1500",
        "women cycling shorts",
        "gold necklace for women",
        "mobile back cover case",
    ]
    for ex in exemples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.historique = []
        st.rerun()

# ── Page principale ─────────────────────────────────────────────
st.title("🛒 Flipkart AI Shopping Assistant")
st.caption("Pose une question sur un produit : type, budget, marque.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            afficher_sources(msg.get("sources"))

question = st.chat_input("Ex : wrist watch for men under 1500")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le catalogue..."):
            try:
                reponse, docs, requete = repondre(question, st.session_state.historique)
                sources = [d.metadata for d in docs]
            except Exception as e:
                reponse, sources = f"Erreur : {e}", []
        st.write(reponse)
        afficher_sources(sources)

    st.session_state.historique.append((question, reponse))
    st.session_state.messages.append(
        {"role": "assistant", "content": reponse, "sources": sources}
    )
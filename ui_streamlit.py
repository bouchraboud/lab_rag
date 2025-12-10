# ui_streamlit.py
import streamlit as st
import requests

st.set_page_config(page_title="RAG IPCC Demo", page_icon="🌍")

st.title("🌍 RAG Demo - IPCC AR6")
st.subheader("Ollama + LangChain")

st.markdown("""
Posez vos questions sur les rapports IPCC AR6.
Le système utilisera la Retrieval-Augmented Generation pour répondre.
""")

# Input utilisateur
question = st.text_input(
    "Votre question:",
    placeholder="Ex: Quels sont les principaux impacts du changement climatique?"
)

if st.button("🔍 Rechercher", type="primary") and question:
    with st.spinner("Recherche en cours..."):
        try:
            # Appel API
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question},
                timeout=60
            )
            
            if response.ok:
                data = response.json()
                
                # Afficher la réponse
                st.success("✅ Réponse trouvée!")
                st.markdown("### 💡 Réponse")
                st.write(data["answer"])
                
                # Afficher les sources
                if data.get("sources"):
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(data["sources"], 1):
                        st.markdown(f"**Source {i}:** {source.get('source', 'N/A')}")
            else:
                st.error(f"❌ Erreur API: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est lancée.")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# Sidebar avec infos
with st.sidebar:
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Cette application utilise:
    - **Ollama** pour les LLMs locaux
    - **LangChain** pour le pipeline RAG
    - **ChromaDB** pour la recherche vectorielle
    - **FastAPI** pour l'API backend
    
    **Documents sources:**
    - IPCC AR6 WGI SPM
    - IPCC AR6 SYR Full Volume
    - IPCC AR6 SYR SPM
    """)
    
    st.markdown("### 🔗 Liens utiles")
    st.markdown("[API Docs](http://localhost:8000/docs)")
    st.markdown("[IPCC Reports](https://www.ipcc.ch)")

# ═══════════════════════════════════════════════════════════
# INTERFACE STREAMLIT POUR MINI-PALM (VERSION AMÉLIORÉE)
# - Vocabulaire enrichi
# - Courbes de perte
# - Mesure du temps d'inférence
# - Démonstration interactive
# ═══════════════════════════════════════════════════════════

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import time
import os
from mini_palm import MiniPaLM
from train import CharTokenizer, generer
from corpus import get_corpus


# ═══════════════════════════════════════════════════════════
# Configuration de la page
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mini-PaLM Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════
# CSS personnalisé
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC04, #34A853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #34A853;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Titre
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🤖 Mini-PaLM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Implémentation pédagogique du modèle PaLM de Google</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Chargement du modèle
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def charger_modele():
    """Charge le modèle entraîné"""
    texte = get_corpus(repetitions=5)
    tokenizer = CharTokenizer(texte)

    model = MiniPaLM(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=64
    )

    try:
        model.load_state_dict(torch.load("mini_palm.pt", map_location="cpu"))
        model.eval()
        return model, tokenizer, True
    except FileNotFoundError:
        return None, tokenizer, False


def charger_historique():
    """Charge l'historique d'entraînement"""
    try:
        with open("historique.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


model, tokenizer, modele_charge = charger_modele()
historique = charger_historique()


# ═══════════════════════════════════════════════════════════
# Fonction de génération avec mesure du temps
# ═══════════════════════════════════════════════════════════
def generer_avec_temps(model, tokenizer, debut, longueur, temperature):
    """Génère du texte et mesure le temps"""
    model.eval()

    debut_filtre = ''.join([c for c in debut if c in tokenizer.char_to_id])
    if not debut_filtre:
        return None

    tokens = tokenizer.encoder(debut_filtre)
    tokens = torch.tensor(tokens).unsqueeze(0)

    temps_debut = time.time()

    with torch.no_grad():
        for _ in range(longueur):
            # Tronquer si la séquence devient trop longue
            tokens_input = tokens[:, -64:] if tokens.shape[1] > 64 else tokens
            logits = model(tokens_input)
            dernier_logits = logits[:, -1, :] / temperature
            probas = F.softmax(dernier_logits, dim=-1)
            prochain_token = torch.multinomial(probas, num_samples=1)
            tokens = torch.cat([tokens, prochain_token], dim=1)

    temps_total = time.time() - temps_debut

    return {
        "texte": tokenizer.decoder(tokens[0].tolist()),
        "temps_ms": temps_total * 1000,
        "tokens_par_seconde": longueur / temps_total if temps_total > 0 else 0
    }


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📊 Informations du modèle")

    if modele_charge:
        st.success("✅ Modèle chargé")

        nb_params = sum(p.numel() for p in model.parameters())

        col_a, col_b = st.columns(2)
        col_a.metric("Paramètres", f"{nb_params/1000:.0f}K")
        col_b.metric("Vocab", f"{tokenizer.vocab_size}")

        col_c, col_d = st.columns(2)
        col_c.metric("Couches", "4")
        col_d.metric("Têtes", "4")

        st.divider()

        st.subheader("🏗️ Composants PaLM")
        st.markdown("""
        ✅ Decoder-only Transformer
        ✅ SwiGLU activation
        ✅ RMSNorm
        ✅ Multi-Head Attention
        ✅ Masque causal
        ✅ Connexions résiduelles
        """)

        st.divider()

        st.subheader("📚 Comparaison")
        st.markdown("""
        | Modèle | Paramètres |
        |--------|-----------|
        | PaLM | 540 G |
        | GPT-3 | 175 G |
        | LLaMA-2 | 70 G |
        | **Mini-PaLM** | **~290 K** |
        """)
    else:
        st.error("❌ Modèle non trouvé")
        st.warning("Lance `python train.py`")


# ═══════════════════════════════════════════════════════════
# CORPS PRINCIPAL : 6 onglets
# ═══════════════════════════════════════════════════════════
if modele_charge:

    onglet1, onglet2, onglet3, onglet4, onglet5, onglet6 = st.tabs([
        "🎲 Génération",
        "🔍 Probabilités",
        "📈 Entraînement",
        "⚡ Performance",
        "🏛️ Architecture",
        "📖 À propos"
    ])

    # ═════════════════════════════════════════════════════
    # ONGLET 1 : GÉNÉRATION
    # ═════════════════════════════════════════════════════
    with onglet1:
        st.header("Génération de texte interactive")

        col1, col2 = st.columns([2, 1])

        with col1:
            debut = st.text_input(
                "✏️ Début de phrase",
                value="Le chat",
                help="Tape un début de phrase. Tu peux utiliser majuscules et accents !"
            )

            st.markdown("**💡 Exemples rapides :**")
            cols = st.columns(5)
            exemples = ["Le chat", "Marie", "L'hiver", "La France", "Les enfants"]
            for col, ex in zip(cols, exemples):
                if col.button(ex, key=f"ex_{ex}"):
                    debut = ex

        with col2:
            longueur = st.slider("📏 Longueur", 20, 200, 100, 10)
            temperature = st.slider("🌡️ Température", 0.1, 2.0, 0.8, 0.1)

            # Indicateur visuel de température
            if temperature < 0.5:
                st.info("🥶 Mode prévisible")
            elif temperature < 1.2:
                st.success("⚖️ Mode équilibré")
            else:
                st.warning("🔥 Mode créatif")

        if st.button("🎲 Générer le texte", type="primary", use_container_width=True):
            if not debut:
                st.warning("⚠️ Tape un début de phrase")
            else:
                with st.spinner("🤖 Le modèle génère du texte..."):
                    resultat = generer_avec_temps(model, tokenizer, debut, longueur, temperature)

                if resultat is None:
                    st.error("Aucun caractère valide")
                else:
                    st.markdown("### 📝 Résultat")
                    st.markdown(
                        f'<div class="result-box">{resultat["texte"]}</div>',
                        unsafe_allow_html=True
                    )

                    # Métriques de performance
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("⏱️ Temps", f"{resultat['temps_ms']:.0f} ms")
                    col_m2.metric("⚡ Vitesse", f"{resultat['tokens_par_seconde']:.0f} tok/s")
                    col_m3.metric("📏 Longueur", f"{longueur} tokens")
                    col_m4.metric("🌡️ Température", f"{temperature:.1f}")

    # ═════════════════════════════════════════════════════
    # ONGLET 2 : PROBABILITÉS
    # ═════════════════════════════════════════════════════
    with onglet2:
        st.header("🔍 Visualisation des probabilités")
        st.markdown("""
        Voici les **caractères les plus probables** que le modèle prédit pour
        le prochain token, après ton texte d'entrée. C'est exactement ce que fait
        le softmax à la sortie du modèle.
        """)

        texte_test = st.text_input("Texte d'entrée", value="Le cha", key="viz_input")

        if st.button("🔬 Analyser les probabilités"):
            if not texte_test:
                st.warning("⚠️ Tape du texte d'abord")
            else:
                tokens_ids = tokenizer.encoder(texte_test)
                if not tokens_ids:
                    st.error("Aucun caractère valide")
                else:
                    tokens = torch.tensor(tokens_ids).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(tokens)
                        dernier_logits = logits[:, -1, :]
                        probas = F.softmax(dernier_logits, dim=-1).squeeze().numpy()

                    # Top 10
                    top_indices = np.argsort(probas)[::-1][:10]
                    top_chars = [tokenizer.id_to_char[i] for i in top_indices]
                    top_probas = [probas[i] for i in top_indices]

                    top_chars_display = [
                        "[espace]" if c == " " else
                        "[\\n]" if c == "\n" else c
                        for c in top_chars
                    ]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    couleurs = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_chars)))
                    bars = ax.barh(top_chars_display[::-1], top_probas[::-1], color=couleurs[::-1])

                    ax.set_xlabel("Probabilité", fontsize=12)
                    ax.set_title(f"Top 10 prochains caractères après : '{texte_test}'",
                                fontsize=14, fontweight='bold')
                    ax.set_xlim(0, max(top_probas) * 1.15)

                    for bar, proba in zip(bars, top_probas[::-1]):
                        width = bar.get_width()
                        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                               f'{proba:.3f}', va='center', fontsize=10)

                    plt.tight_layout()
                    st.pyplot(fig)

                    df = pd.DataFrame({
                        "Rang": range(1, 11),
                        "Caractère": top_chars_display,
                        "Probabilité": [f"{p:.4f}" for p in top_probas],
                        "Pourcentage": [f"{p*100:.2f}%" for p in top_probas]
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════
    # ONGLET 3 : ENTRAÎNEMENT (NOUVEAU)
    # ═════════════════════════════════════════════════════
    with onglet3:
        st.header("📈 Historique d'entraînement")

        if historique is None:
            st.warning("⚠️ Aucun historique trouvé. Relance `python train.py`")
        else:
            # Métriques globales
            col1, col2, col3, col4 = st.columns(4)

            perte_initiale = historique["loss_par_epoch"][0]
            perte_finale = historique["loss_par_epoch"][-1]
            amelioration = (1 - perte_finale / perte_initiale) * 100

            col1.metric("📊 Perte initiale", f"{perte_initiale:.4f}")
            col2.metric("✅ Perte finale", f"{perte_finale:.4f}",
                       delta=f"-{perte_initiale - perte_finale:.4f}", delta_color="inverse")
            col3.metric("📉 Amélioration", f"{amelioration:.1f}%")
            col4.metric("⏱️ Temps total", f"{historique['temps_total']:.1f}s")

            st.divider()

            # ── Graphique 1 : Courbe de perte par step ────
            st.subheader("📉 Courbe de perte (par step)")
            st.markdown("Évolution de la perte à chaque batch d'entraînement.")

            fig, ax = plt.subplots(figsize=(12, 5))
            steps = list(range(len(historique["loss_par_step"])))

            ax.plot(steps, historique["loss_par_step"],
                   color='#4285F4', alpha=0.4, linewidth=0.5, label='Perte par step')

            # Moyenne mobile pour lisser
            window = 20
            if len(historique["loss_par_step"]) > window:
                moving_avg = np.convolve(
                    historique["loss_par_step"],
                    np.ones(window)/window,
                    mode='valid'
                )
                ax.plot(range(window-1, len(historique["loss_par_step"])),
                       moving_avg, color='#EA4335', linewidth=2,
                       label=f'Moyenne mobile ({window} steps)')

            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Perte (Cross-Entropy)", fontsize=12)
            ax.set_title("Évolution de la perte pendant l'entraînement",
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            # Marquer les fins d'epochs
            steps_par_epoch = len(historique["loss_par_step"]) // len(historique["loss_par_epoch"])
            for i in range(1, len(historique["loss_par_epoch"])):
                ax.axvline(x=i*steps_par_epoch, color='gray', linestyle='--', alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig)

            st.divider()

            # ── Graphique 2 : Perte par epoch ─────────────
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.subheader("📊 Perte moyenne par epoch")

                fig2, ax2 = plt.subplots(figsize=(8, 5))
                epochs = historique["epochs"]
                losses = historique["loss_par_epoch"]

                ax2.plot(epochs, losses, marker='o', markersize=10,
                        linewidth=2, color='#34A853')
                ax2.fill_between(epochs, losses, alpha=0.2, color='#34A853')

                for i, (e, l) in enumerate(zip(epochs, losses)):
                    ax2.annotate(f'{l:.3f}', (e, l), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=9)

                ax2.set_xlabel("Epoch", fontsize=12)
                ax2.set_ylabel("Perte moyenne", fontsize=12)
                ax2.set_title("Convergence du modèle", fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_xticks(epochs)

                plt.tight_layout()
                st.pyplot(fig2)

            with col_g2:
                st.subheader("⏱️ Temps par epoch")

                fig3, ax3 = plt.subplots(figsize=(8, 5))
                temps = historique["temps_par_epoch"]

                bars = ax3.bar(epochs, temps, color='#FBBC04', edgecolor='#F09D00')

                for bar, t in zip(bars, temps):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

                ax3.set_xlabel("Epoch", fontsize=12)
                ax3.set_ylabel("Temps (secondes)", fontsize=12)
                ax3.set_title("Durée de chaque epoch", fontsize=13, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.set_xticks(epochs)

                plt.tight_layout()
                st.pyplot(fig3)

            st.divider()

            # ── Tableau récapitulatif ─────────────────────
            st.subheader("📋 Détails par epoch")
            df_recap = pd.DataFrame({
                "Epoch": historique["epochs"],
                "Perte moyenne": [f"{l:.4f}" for l in historique["loss_par_epoch"]],
                "Temps (s)": [f"{t:.2f}" for t in historique["temps_par_epoch"]],
            })
            st.dataframe(df_recap, use_container_width=True, hide_index=True)

            # ── Configuration utilisée ────────────────────
            st.divider()
            st.subheader("⚙️ Configuration de l'entraînement")
            config = historique["config"]
            col_c1, col_c2, col_c3 = st.columns(3)

            with col_c1:
                st.markdown("**Architecture**")
                st.markdown(f"- Vocabulaire : {config['vocab_size']}")
                st.markdown(f"- Dimension : {config['dim']}")
                st.markdown(f"- Couches : {config['num_layers']}")
                st.markdown(f"- Têtes : {config['num_heads']}")

            with col_c2:
                st.markdown("**Entraînement**")
                st.markdown(f"- Seq length : {config['seq_len']}")
                st.markdown(f"- Batch size : {config['batch_size']}")
                st.markdown(f"- Learning rate : {config['learning_rate']}")
                st.markdown(f"- Appareil : {config['device'].upper()}")

            with col_c3:
                st.markdown("**Statistiques**")
                st.markdown(f"- Paramètres : {config['nb_params']:,}")
                st.markdown(f"- Total steps : {len(historique['loss_par_step']):,}")
                st.markdown(f"- Epochs : {len(historique['epochs'])}")

    # ═════════════════════════════════════════════════════
    # ONGLET 4 : PERFORMANCE (NOUVEAU)
    # ═════════════════════════════════════════════════════
    with onglet4:
        st.header("⚡ Performance et benchmarks")

        st.markdown("""
        Cet onglet mesure la **vitesse d'inférence** du modèle.
        Tu peux générer plusieurs textes et voir combien de temps cela prend.
        """)

        col_bench1, col_bench2 = st.columns(2)

        with col_bench1:
            longueur_bench = st.slider("Longueur de génération", 50, 300, 100, 50,
                                      key="bench_len")
            nb_runs = st.slider("Nombre de tests", 3, 20, 5, key="nb_runs")

        with col_bench2:
            temp_bench = st.slider("Température", 0.1, 2.0, 0.8, 0.1,
                                  key="bench_temp")
            debut_bench = st.text_input("Début de phrase", "Le chat",
                                       key="bench_debut")

        if st.button("🚀 Lancer le benchmark", type="primary", use_container_width=True):
            with st.spinner(f"Exécution de {nb_runs} tests..."):
                resultats = []
                progress = st.progress(0)

                for i in range(nb_runs):
                    res = generer_avec_temps(model, tokenizer, debut_bench,
                                            longueur_bench, temp_bench)
                    if res:
                        resultats.append(res)
                    progress.progress((i + 1) / nb_runs)

                progress.empty()

            if resultats:
                # Statistiques
                temps_ms = [r["temps_ms"] for r in resultats]
                vitesses = [r["tokens_par_seconde"] for r in resultats]

                st.subheader("📊 Résultats du benchmark")

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("⏱️ Temps moyen", f"{np.mean(temps_ms):.0f} ms")
                col_s2.metric("⚡ Vitesse moyenne", f"{np.mean(vitesses):.0f} tok/s")
                col_s3.metric("🚀 Plus rapide", f"{min(temps_ms):.0f} ms")
                col_s4.metric("🐌 Plus lent", f"{max(temps_ms):.0f} ms")

                # Graphique des temps
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Histogramme des temps
                ax1.bar(range(1, len(temps_ms) + 1), temps_ms,
                       color='#4285F4', edgecolor='#1967D2')
                ax1.axhline(y=np.mean(temps_ms), color='red', linestyle='--',
                           label=f'Moyenne : {np.mean(temps_ms):.0f}ms')
                ax1.set_xlabel("Test #", fontsize=12)
                ax1.set_ylabel("Temps (ms)", fontsize=12)
                ax1.set_title("Temps d'inférence par test",
                             fontsize=13, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')

                # Histogramme des vitesses
                ax2.bar(range(1, len(vitesses) + 1), vitesses,
                       color='#34A853', edgecolor='#1E8E3E')
                ax2.axhline(y=np.mean(vitesses), color='red', linestyle='--',
                           label=f'Moyenne : {np.mean(vitesses):.0f} tok/s')
                ax2.set_xlabel("Test #", fontsize=12)
                ax2.set_ylabel("Tokens / seconde", fontsize=12)
                ax2.set_title("Vitesse de génération",
                             fontsize=13, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                st.pyplot(fig)

                # Détails de chaque génération
                st.subheader("📝 Échantillons générés")
                for i, res in enumerate(resultats[:3]):
                    with st.expander(f"Test #{i+1} ({res['temps_ms']:.0f}ms)"):
                        st.markdown(f"```\n{res['texte']}\n```")

                # Comparaison avec PaLM réel
                st.divider()
                st.subheader("🆚 Comparaison Mini-PaLM vs PaLM réel")

                df_compare = pd.DataFrame({
                    "Métrique": [
                        "Paramètres",
                        "Vitesse (tokens/s sur CPU)",
                        "Mémoire requise",
                        "Coût d'entraînement",
                        "Latence par token"
                    ],
                    "Mini-PaLM (ce projet)": [
                        f"{sum(p.numel() for p in model.parameters()):,}",
                        f"{np.mean(vitesses):.0f}",
                        "~5 MB",
                        "~30 secondes",
                        f"{np.mean(temps_ms)/longueur_bench:.1f} ms"
                    ],
                    "PaLM 540B (Google)": [
                        "540,000,000,000",
                        "~50 (sur TPU)",
                        "~1 TB",
                        "~$10 millions",
                        "~20 ms"
                    ]
                })
                st.dataframe(df_compare, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════
    # ONGLET 5 : ARCHITECTURE
    # ═════════════════════════════════════════════════════
    with onglet5:
        st.header("🏛️ Architecture du modèle")

        st.markdown("""
        ### Pipeline de traitement

        ```
        Texte d'entrée
              ↓
        Tokenizer → IDs (nombres)
              ↓
        Embedding → Vecteurs (128 dim)
              ↓
        ┌─────────────────────────────────┐
        │  Bloc Transformer (× 4)         │
        │  ├─ RMSNorm                     │
        │  ├─ Multi-Head Attention        │
        │  ├─ Connexion résiduelle        │
        │  ├─ RMSNorm                     │
        │  ├─ SwiGLU FFN                  │
        │  └─ Connexion résiduelle        │
        └─────────────────────────────────┘
              ↓
        RMSNorm finale
              ↓
        LM Head → Logits
              ↓
        Softmax → Probabilités
              ↓
        Échantillonnage → Prochain token
        ```
        """)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔧 Composants techniques")

            with st.expander("📐 RMSNorm"):
                st.latex(r"y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot g")
                st.markdown("Normalisation utilisée dans PaLM. Plus rapide que LayerNorm.")

            with st.expander("⚡ SwiGLU"):
                st.latex(r"\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_2)W_3")
                st.markdown("Fonction d'activation utilisée dans le FFN de PaLM.")

            with st.expander("🎯 Attention"):
                st.latex(r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
                st.markdown("Mécanisme central du Transformer.")

        with col2:
            st.subheader("📊 Répartition des paramètres")

            params_embedding = sum(p.numel() for p in model.embedding.parameters())
            params_layers = sum(p.numel() for p in model.layers.parameters())
            params_head = sum(p.numel() for p in model.lm_head.parameters())
            total = params_embedding + params_layers + params_head

            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['Embedding', 'Couches Transformer', 'LM Head']
            sizes = [params_embedding, params_layers, params_head]
            colors = ['#4285F4', '#EA4335', '#34A853']

            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 12})
            ax.set_title("Répartition des paramètres",
                        fontsize=14, fontweight='bold')

            st.pyplot(fig)

            st.markdown(f"""
            | Composant | Paramètres |
            |-----------|------------|
            | Embedding | {params_embedding:,} |
            | Couches Transformer | {params_layers:,} |
            | LM Head | {params_head:,} |
            | **TOTAL** | **{total:,}** |
            """)

    # ═════════════════════════════════════════════════════
    # ONGLET 6 : À PROPOS
    # ═════════════════════════════════════════════════════
    with onglet6:
        st.header("📖 À propos du projet")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Objectif")
            st.markdown("""
            Implémentation **pédagogique simplifiée** du modèle
            **PaLM (Pathways Language Model)** de Google.

            Tous les composants clés de l'architecture sont implémentés
            from scratch en PyTorch.
            """)

            st.subheader("🏗️ Différences avec PaLM réel")
            st.markdown("""
            | Aspect | PaLM réel | Mini-PaLM |
            |--------|-----------|-----------|
            | Paramètres | 540 G | ~290 K |
            | Couches | 118 | 4 |
            | Dimension | 18,432 | 128 |
            | Têtes | 48 | 4 |
            | Vocabulaire | 256K | ~70 |
            | Données | 780B tokens | ~20K caractères |
            """)

        with col2:
            st.subheader("🔬 Concepts démontrés")
            st.markdown("""
            ✅ Architecture decoder-only
            ✅ Self-attention multi-têtes
            ✅ SwiGLU activation
            ✅ RMSNorm
            ✅ Masque causal
            ✅ Next token prediction
            ✅ Sampling avec température
            ✅ Mesure de performance
            """)

            st.subheader("📚 Références")
            st.markdown("""
            - **PaLM** : Chowdhery et al., 2022
            - **SwiGLU** : Shazeer, 2020
            - **Transformer** : Vaswani et al., 2017
            """)

else:
    st.error("⚠️ Modèle non trouvé")
    st.markdown("""
    ### Comment résoudre ce problème :

    1. Ouvre un terminal dans ce dossier
    2. Lance : `python train.py`
    3. Attends ~5 minutes
    4. Relance cette interface
    """)


# ═══════════════════════════════════════════════════════════
# Pied de page
# ═══════════════════════════════════════════════════════════
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem;'>
        🎓 Mini-PaLM | Implémentation pédagogique pour présentation universitaire<br>
        Inspiré du papier <i>"PaLM: Scaling Language Modeling with Pathways"</i> (Google, 2022)
    </div>
    """,
    unsafe_allow_html=True
)

# Mini-PaLM : Implémentation Pédagogique avec Interface Web

Implémentation simplifiée du modèle **PaLM (Pathways Language Model)** de Google avec une **interface web interactive Streamlit** pour la démonstration.

---

## ✨ Nouveautés de cette version

- 📚 **Corpus enrichi** : ~70 caractères (majuscules, accents, chiffres, ponctuation)
- 📊 **Historique d'entraînement** sauvegardé (courbes de perte par step et epoch)
- ⚡ **Mesure de performance** : temps d'inférence et tokens/seconde
- 🎨 **Interface améliorée** avec 6 onglets dont des benchmarks

---

## 📁 Structure du projet

```
mini_palm_projet/
├── mini_palm.py           # Architecture du modèle (RMSNorm, SwiGLU, Attention)
├── corpus.py              # Corpus français enrichi
├── train.py               # Entraînement + sauvegarde de l'historique
├── utiliser_modele.py     # Mode console interactif
├── app.py                 # Interface web Streamlit ⭐
├── requirements.txt       # Dépendances
└── README.md              # Ce fichier
```

**Fichiers générés après entraînement :**
- `mini_palm.pt` : poids du modèle entraîné
- `historique.json` : historique de l'entraînement (perte, temps)
- `vocab.json` : vocabulaire utilisé

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

Ou manuellement :

```bash
pip install torch streamlit matplotlib pandas numpy
```

---

## ▶️ Utilisation (3 étapes)

### Étape 1 : Entraîner le modèle (une seule fois)

```bash
python train.py
```

⏳ Durée : ~3-5 minutes sur CPU
📁 Crée : `mini_palm.pt`, `historique.json`, `vocab.json`

### Étape 2 : Lancer l'interface web

```bash
python -m streamlit run app.py
```

🌐 Ouverture automatique sur **http://localhost:8501**

### Étape 3 : Explorer les 6 onglets

---

## 🎨 Les 6 onglets de l'interface

### 🎲 1. Génération
- Champ de saisie + boutons d'exemples rapides
- Sliders pour longueur et température
- **Affichage du temps d'inférence** et de la vitesse en tokens/s
- Indicateur visuel de température (prévisible / équilibré / créatif)

### 🔍 2. Probabilités
- Top 10 caractères les plus probables
- Graphique en barres avec valeurs
- Tableau détaillé avec pourcentages

### 📈 3. Entraînement (NOUVEAU)
- **Courbe de perte par step** avec moyenne mobile
- **Perte moyenne par epoch** avec annotations
- **Temps de chaque epoch** en histogramme
- Métriques : perte initiale, finale, amélioration en %
- Configuration complète de l'entraînement

### ⚡ 4. Performance (NOUVEAU)
- **Benchmark interactif** : lance N tests de génération
- Graphiques temps + vitesse
- Statistiques (moyenne, min, max)
- **Comparaison Mini-PaLM vs PaLM 540B**
- Affichage des échantillons générés

### 🏛️ 5. Architecture
- Pipeline complet du modèle
- Formules mathématiques (RMSNorm, SwiGLU, Attention)
- Camembert de répartition des paramètres

### 📖 6. À propos
- Comparaison avec PaLM réel
- Concepts démontrés
- Références scientifiques

---

## 🎬 Scénario de présentation recommandé (5 minutes)

### Intro (30s)
> "J'ai implémenté une version simplifiée mais fonctionnelle de PaLM avec tous les composants clés."

**→ Onglet "À propos"** : montrer la comparaison Mini-PaLM vs PaLM 540B

### Architecture (1 min)
> "Voici le pipeline et les composants techniques."

**→ Onglet "Architecture"** : pipeline + formules SwiGLU/RMSNorm/Attention

### Entraînement (1 min 30)
> "Voici comment le modèle a appris."

**→ Onglet "Entraînement"** : montrer la courbe de perte qui descend
- *"Perte initiale 4.2 → finale 0.3 = amélioration de 92%"*

### Probabilités (1 min)
> "Concrètement, le modèle calcule des probabilités."

**→ Onglet "Probabilités"** : taper "Le cha" et montrer le top 10

### Démo en direct (1 min)
> "Voyons une génération en direct !"

**→ Onglet "Génération"** :
- Demander un début au jury
- Générer avec température 0.5 puis 1.5
- Montrer la différence de créativité

### Bonus : Performance (30s)
**→ Onglet "Performance"** : lancer un benchmark
- *"Mini-PaLM génère X tokens/seconde sur CPU"*

---

## 🌡️ Comprendre la température

| Température | Comportement | Exemple |
|-------------|--------------|---------|
| 0.1 - 0.3 | Très prévisible | "Le chat dort. Le chat dort. Le chat dort..." |
| 0.7 - 0.9 | Équilibré ⭐ | "Le chat dort sur le canapé du salon." |
| 1.2 - 1.5 | Créatif | "Le chat saute dans les arbres fleuris." |
| 1.8 - 2.0 | Aléatoire | "Le chat xqz prk dans..." |

---

## 🛠️ Erreurs courantes

### ❌ "Modèle non trouvé"
→ Lance d'abord `python train.py`

### ❌ "Aucun historique trouvé"
→ Relance `python train.py` (la nouvelle version sauvegarde l'historique)

### ❌ "ModuleNotFoundError: streamlit"
→ Installe avec `pip install -r requirements.txt`

### ❌ Port 8501 déjà utilisé
→ `python -m streamlit run app.py --server.port 8502`

### ❌ Caractères inconnus
→ Le tokenizer ignore automatiquement les caractères inconnus.
   Le vocabulaire inclut : a-z, A-Z, 0-9, accents (é, è, à, ç...), ponctuation

---

## 📊 Comparaison des versions

| Aspect | Version 1 (basique) | Version 2 (actuelle) |
|--------|---------------------|---------------------|
| Vocabulaire | 23 caractères | **~70 caractères** |
| Corpus | Phrases simples × 50 | **Texte français riche × 5** |
| Onglets interface | 4 | **6** |
| Courbes de perte | ❌ | ✅ |
| Benchmarks | ❌ | ✅ |
| Comparaison PaLM | Statique | **Dynamique** |
| Mesure temps | ❌ | ✅ |

---

## 🎯 Liens avec la présentation

| Section présentation | Démontré dans |
|----------------------|---------------|
| III. Decoder-only Transformer | Onglet Architecture |
| III. SwiGLU, RMSNorm, Attention | Onglet Architecture (formules) |
| V. Next token prediction | Onglet Génération |
| V. CrossEntropy Loss | Onglet Entraînement (courbe) |
| V. Convergence | Onglet Entraînement |
| VI. Performance | Onglet Performance |
| VII. Coût et passage à l'échelle | Comparaison Mini-PaLM vs PaLM 540B |

---

## 📊 Spécifications techniques

- **Framework** : PyTorch 2.x
- **Interface** : Streamlit 1.x
- **Vocabulaire** : ~70 caractères français (a-z, A-Z, 0-9, accents)
- **Paramètres** : ~290K (vs 540B pour PaLM)
- **Couches** : 4 blocs Transformer
- **Dimension** : 128
- **Têtes d'attention** : 4
- **Longueur de séquence** : 64
- **Batch size** : 32
- **Learning rate** : 3e-4
- **Optimiseur** : AdamW
- **Epochs** : 8

---

## 📖 Références

- **PaLM** : "PaLM: Scaling Language Modeling with Pathways" (Chowdhery et al., 2022)
- **SwiGLU** : "GLU Variants Improve Transformer" (Shazeer, 2020)
- **RMSNorm** : "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- **Attention** : "Attention is All You Need" (Vaswani et al., 2017)
- **Streamlit** : https://docs.streamlit.io
- **PyTorch** : https://pytorch.org

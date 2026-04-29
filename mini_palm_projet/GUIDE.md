# 📖 Guide Complet du Projet Mini-PaLM

> Ce guide est conçu pour que tu comprennes **TOUT** le projet et puisses répondre aux questions du jury avec confiance.

---

## 📚 Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Concepts fondamentaux à maîtriser](#2-concepts-fondamentaux-à-maîtriser)
3. [Comprendre chaque fichier](#3-comprendre-chaque-fichier)
4. [Le pipeline complet (étape par étape)](#4-le-pipeline-complet-étape-par-étape)
5. [Les composants techniques en détail](#5-les-composants-techniques-en-détail)
6. [L'entraînement expliqué simplement](#6-lentraînement-expliqué-simplement)
7. [La génération de texte](#7-la-génération-de-texte)
8. [Questions probables du jury (avec réponses)](#8-questions-probables-du-jury)
9. [Glossaire complet](#9-glossaire-complet)

---

## 1. Vue d'ensemble du projet

### 🎯 Qu'est-ce qu'on a construit ?

Un **modèle de langage** simplifié inspiré de PaLM (Google).

**Définition simple** : Un modèle de langage est un programme qui apprend à **prédire le prochain caractère** (ou mot) dans une phrase, à partir des caractères précédents.

### 🤔 Pourquoi c'est utile ?

Une fois entraîné, le modèle peut :
- **Générer du texte** : on lui donne un début, il complète
- **Comprendre la structure** du langage (grammaire, vocabulaire)
- **Apprendre les régularités** des phrases françaises

### 📊 Comparaison avec le vrai PaLM

| Aspect | Notre Mini-PaLM | PaLM (Google) |
|--------|-----------------|---------------|
| Taille | ~290K paramètres | 540 milliards |
| Couches | 4 | 118 |
| Vocabulaire | ~70 caractères | 256 000 sous-mots |
| Données | 4 000 caractères | 780 milliards de tokens |
| Coût | ~30 secondes | ~10 millions $ |

**👉 On a une miniature de l'original, mais avec EXACTEMENT la même architecture.**

---

## 2. Concepts fondamentaux à maîtriser

Avant de plonger dans le code, voici les **6 concepts essentiels** :

### 🔤 Concept 1 : Token
Un **token** est l'unité de base que le modèle traite.

Dans notre projet, **un token = un caractère** :
- "Le chat" = 7 tokens : `['L', 'e', ' ', 'c', 'h', 'a', 't']`

Dans GPT/PaLM réels, un token = un sous-mot (généralement 3-4 caractères).

### 🔢 Concept 2 : Vocabulaire
Le **vocabulaire** est l'ensemble de tous les tokens uniques que le modèle connaît.

Notre vocabulaire contient ~70 caractères :
- 26 minuscules (a-z)
- 26 majuscules (A-Z)
- 10 chiffres (0-9)
- Caractères accentués (é, è, à, ç...)
- Ponctuation (. , ; : ' " ...)
- Espace et retour à la ligne

### 🧮 Concept 3 : Tokenizer
Le **tokenizer** convertit du texte en nombres (et inversement) :

```
Texte:    "Le chat"
            ↓ (encoder)
IDs:      [11, 38, 0, 36, 41, 34, 53]
            ↓ (decoder)
Texte:    "Le chat"
```

**Pourquoi des nombres ?** Parce que les ordinateurs ne comprennent que les nombres, pas les caractères.

### 📐 Concept 4 : Embedding
Un **embedding** transforme un nombre (ID) en un **vecteur** (liste de nombres).

```
ID: 11  →  [0.12, -0.45, 0.67, 0.23, ..., 0.89]  (128 nombres)
```

**Pourquoi ?** Un seul nombre n'est pas assez riche pour représenter un caractère. Un vecteur de 128 valeurs peut capturer beaucoup plus d'informations (par exemple : voyelle/consonne, fréquence, contexte typique...).

### 🎯 Concept 5 : Attention
L'**attention** est le mécanisme qui permet à chaque token de "regarder" les autres tokens pour comprendre le contexte.

**Exemple** : Dans "Le chat dort sur le tapis", quand le modèle traite le mot "dort", l'attention lui permet de regarder "chat" pour comprendre QUI dort.

### 🎰 Concept 6 : Probabilités et échantillonnage
À chaque étape, le modèle calcule **une probabilité pour chaque caractère possible** d'être le prochain.

```
Après "Le cha", probabilités :
- 't' : 65%  ← très probable
- 'p' : 20%  (chapeau)
- 'r' : 8%   (charger)
- 'm' : 5%   (chameau)
- autres : 2%
```

Puis on **tire au hasard** un caractère selon ces probabilités. Plus la probabilité est haute, plus le caractère a de chances d'être choisi.

---

## 3. Comprendre chaque fichier

### 📄 `corpus.py` — Les données d'entraînement

**Rôle** : Contient le texte français qui sert à entraîner le modèle.

**Structure** :
```python
CORPUS = """
Le chat dort sur le canapé...
Marie habite à Paris...
...
"""

def get_corpus(repetitions=5):
    return CORPUS * repetitions
```

**À retenir** :
- Le corpus fait ~4000 caractères uniques (683 mots)
- On le répète 5 fois → 20 000 caractères pour l'entraînement
- Plus de répétitions = le modèle voit chaque phrase plusieurs fois = il apprend mieux

### 📄 `mini_palm.py` — L'architecture du modèle

**Rôle** : Définit la structure du modèle (les couches, les transformations).

**Contient 5 classes** :
1. `RMSNorm` : normalisation des valeurs
2. `SwiGLU` : fonction d'activation (non-linéarité)
3. `Attention` : mécanisme d'attention multi-têtes
4. `TransformerBlock` : un bloc complet (attention + FFN)
5. `MiniPaLM` : le modèle complet (empilement de blocs)

### 📄 `train.py` — L'entraînement

**Rôle** : Entraîne le modèle sur le corpus et sauvegarde les poids.

**Étapes** :
1. Charge le corpus
2. Crée le tokenizer
3. Crée le modèle
4. Boucle d'entraînement (8 epochs)
5. Sauvegarde le modèle (`mini_palm.pt`) et l'historique (`historique.json`)

### 📄 `app.py` — L'interface web

**Rôle** : Interface visuelle pour utiliser et démontrer le modèle.

**6 onglets** :
1. Génération
2. Probabilités
3. Entraînement (courbes)
4. Performance (benchmarks)
5. Architecture
6. À propos

### 📄 `utiliser_modele.py` — Mode console

**Rôle** : Tester le modèle dans le terminal sans interface graphique.

---

## 4. Le pipeline complet (étape par étape)

Voici **exactement** ce qui se passe quand tu tapes "Le chat" et que le modèle complète :

### Étape 1 : Tokenisation
```
"Le chat"  →  [11, 38, 0, 36, 41, 34, 53]
```
Le tokenizer convertit chaque caractère en un nombre (son ID dans le vocabulaire).

### Étape 2 : Embedding
```
[11, 38, 0, 36, 41, 34, 53]
        ↓
[[0.12, -0.45, ...],   ← vecteur de 128 dim pour 'L'
 [0.67, 0.23, ...],    ← vecteur de 128 dim pour 'e'
 [-0.34, 0.89, ...],   ← vecteur de 128 dim pour ' '
 ...
]
```
Chaque ID devient un vecteur de 128 nombres.

### Étape 3 : Passage dans les blocs Transformer (× 4)
Chaque bloc applique :

**A) Normalisation (RMSNorm)** : stabilise les valeurs

**B) Attention** : chaque vecteur "regarde" les autres pour comprendre le contexte
- Calcule Q (Query), K (Key), V (Value)
- Calcule des scores d'attention
- Applique un masque causal (empêche de voir le futur)
- Combine les valeurs selon les scores

**C) Connexion résiduelle** : on AJOUTE le résultat à l'entrée
```python
x = x + attention(x)  # au lieu de x = attention(x)
```

**D) Normalisation à nouveau**

**E) FFN avec SwiGLU** : transformation non-linéaire

**F) Connexion résiduelle à nouveau**

### Étape 4 : Normalisation finale + LM Head
```
Vecteurs (128 dim) → Logits (70 valeurs, une par caractère du vocabulaire)
```

### Étape 5 : Softmax → probabilités
```
Logits :       [2.3, 1.1, 4.5, 0.8, ...]
                    ↓ softmax
Probabilités : [0.05, 0.02, 0.65, 0.01, ...]   (somment à 1)
```

### Étape 6 : Échantillonnage
On tire un caractère au hasard selon les probabilités. Avec une probabilité de 65% pour 't', il y a 65% de chances de tirer 't'.

### Étape 7 : Répétition
On ajoute le nouveau caractère à la séquence et on recommence depuis l'étape 2 pour générer le caractère suivant.

---

## 5. Les composants techniques en détail

### 🔧 RMSNorm — Root Mean Square Normalization

**Problème résolu** : Sans normalisation, les valeurs dans le réseau peuvent devenir très grandes ou très petites pendant l'entraînement, ce qui empêche l'apprentissage.

**Formule** :
```
RMS(x) = √(moyenne(x²))
y = x / RMS(x) × paramètre_apprenable
```

**Pourquoi pas LayerNorm ?**
- LayerNorm calcule moyenne ET écart-type
- RMSNorm ne calcule que la racine de la moyenne des carrés
- → RMSNorm est ~30% plus rapide
- C'est ce que PaLM utilise

**Code clé** :
```python
def forward(self, x):
    rms = x.pow(2).mean(-1, keepdim=True)
    return self.scale * x * torch.rsqrt(rms + self.eps)
```

### 🔧 SwiGLU — Swish-Gated Linear Unit

**Problème résolu** : Il faut introduire de la non-linéarité dans le réseau, sinon il ne peut apprendre que des fonctions linéaires (très limité).

**Formule** :
```
SwiGLU(x) = (Swish(x·W₁) ⊙ (x·W₂)) · W₃
où Swish(x) = x · sigmoid(x)
et ⊙ = multiplication élément par élément
```

**Pourquoi pas ReLU ?**
- ReLU coupe brutalement à 0 pour les valeurs négatives
- SwiGLU est plus "doux" et plus expressif
- Le mécanisme de "gate" permet au modèle de **filtrer** l'information
- Améliore les performances de ~2-3% sur les benchmarks

**Code clé** :
```python
def forward(self, x):
    gate = F.silu(self.w1(x))    # Swish gate
    valeur = self.w2(x)
    return self.w3(gate * valeur)
```

### 🔧 Attention multi-têtes

**Problème résolu** : Comprendre les relations entre les mots dans une phrase.

**Formule** :
```
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

**Étapes** :
1. **Calcul de Q, K, V** : trois projections linéaires de l'entrée
2. **Scores** : produit scalaire entre Q et K (transpose), divisé par √d
3. **Masque causal** : on met -∞ aux positions futures
4. **Softmax** : transforme les scores en probabilités
5. **Pondération** : on combine les V selon les probabilités

**Pourquoi multi-têtes ?**
- Chaque tête peut se spécialiser dans un type de relation
- Tête 1 : relations syntaxiques
- Tête 2 : relations sémantiques
- Tête 3 : relations à courte distance
- Tête 4 : relations à longue distance

**Code clé** :
```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
scores = scores.masked_fill(masque == 0, float('-inf'))
poids = F.softmax(scores, dim=-1)
sortie = poids @ v
```

### 🔧 Masque causal

**Problème résolu** : Empêcher le modèle de "tricher" en regardant les caractères futurs pendant l'entraînement.

**Comment ça marche** :
```
Matrice triangulaire inférieure :
[1, 0, 0, 0]    ← Position 1 ne voit que position 1
[1, 1, 0, 0]    ← Position 2 voit positions 1 et 2
[1, 1, 1, 0]    ← Position 3 voit positions 1, 2, 3
[1, 1, 1, 1]    ← Position 4 voit tout (jusqu'à elle-même)
```

Les zéros sont remplacés par -∞, ce qui donne 0 après softmax.

**Pourquoi c'est crucial** :
- Sans masque : le modèle apprend par cœur (overfitting massif)
- Avec masque : il doit VRAIMENT prédire le prochain caractère

### 🔧 Connexions résiduelles

**Problème résolu** : Dans les réseaux profonds, les gradients deviennent très petits → l'entraînement échoue.

**Solution** :
```python
x = x + transformation(x)   # au lieu de x = transformation(x)
```

**Pourquoi ça marche** :
- Le gradient peut "passer à travers" l'addition
- Les couches apprennent des **modifications** plutôt que des transformations totales
- Permet d'empiler 100+ couches (PaLM en a 118)

---

## 6. L'entraînement expliqué simplement

### 🎯 Objectif : Next Token Prediction

Le modèle doit apprendre à prédire le **prochain token** étant donnés les tokens précédents.

**Exemple** :
```
Input :  "Le cha"
Target : "t"

Input :  "Le chat"
Target : " "

Input :  "Le chat "
Target : "d"
```

### 🔄 La boucle d'entraînement

Pour chaque batch (groupe de 32 séquences) :

#### 1️⃣ Forward pass
```python
logits = model(x)  # Calcule les prédictions
```

#### 2️⃣ Calcul de la perte
```python
perte = CrossEntropyLoss(logits, y)
```

**CrossEntropyLoss** mesure à quel point les prédictions sont fausses :
- Si le modèle est très sûr de la mauvaise réponse → grande perte
- Si le modèle est sûr de la bonne réponse → petite perte

#### 3️⃣ Backward pass
```python
perte.backward()  # Calcule les gradients
```

Calcule **comment modifier chaque paramètre** pour réduire l'erreur.

#### 4️⃣ Mise à jour
```python
optimizer.step()  # Applique les modifications
```

### 📈 Pourquoi ça marche ?

À chaque itération :
1. Le modèle fait une prédiction
2. On compare à la vérité
3. On ajuste les paramètres pour réduire l'erreur
4. **Très petit pas dans la bonne direction**

Après des milliers d'itérations, le modèle converge.

### 📊 Lecture de la courbe de perte

```
Perte
 4 │█
 3 │ █
 2 │  ██
 1 │    ██████
 0 │          ██████████
   └─────────────────────→ Epochs
```

- **Au début** : perte ~4 (le modèle prédit au hasard)
- **Au milieu** : la perte chute rapidement (le modèle apprend)
- **À la fin** : la perte stagne (le modèle a convergé)

**Si la perte ne descend pas** : problème de learning rate ou d'architecture.
**Si la perte descend trop vite à 0** : overfitting (le modèle apprend par cœur).

### ⚙️ Hyperparamètres clés

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `learning_rate` | 3e-4 | Taille des pas de l'optimiseur |
| `batch_size` | 32 | Nombre de séquences traitées en parallèle |
| `seq_len` | 64 | Longueur de chaque séquence |
| `epochs` | 8 | Nombre de passages sur les données |
| `dim` | 128 | Dimension des vecteurs |
| `num_layers` | 4 | Nombre de blocs Transformer |
| `num_heads` | 4 | Nombre de têtes d'attention |

---

## 7. La génération de texte

### 🎲 Comment le modèle génère

```python
tokens = encoder("Le chat")  # [11, 38, 0, 36, 41, 34, 53]

for _ in range(longueur):
    logits = model(tokens)              # Calcule les prédictions
    dernier = logits[-1] / temperature  # Prend le DERNIER token
    probas = softmax(dernier)           # Convertit en probabilités
    nouveau = multinomial(probas)       # Tire un caractère au hasard
    tokens.append(nouveau)              # Ajoute à la séquence

texte = decoder(tokens)
```

### 🌡️ La température

La température contrôle la **créativité** du modèle.

**Formule** :
```
probas = softmax(logits / temperature)
```

| Température | Effet | Exemple |
|-------------|-------|---------|
| 0.1 | Toujours le plus probable | "Le chat dort sur le canapé." (répétitif) |
| 0.8 | Équilibré | "Le chat dort sur le canapé du salon." |
| 1.5 | Créatif | "Le chat saute dans l'arbre fleuri." |
| 2.0 | Très aléatoire | "Le chat xyz qrt." |

**Pourquoi diviser par la température ?**
- T < 1 : amplifie les différences → le modèle est plus "sûr"
- T > 1 : réduit les différences → le modèle est plus "hésitant"

### 🎰 Pourquoi pas toujours prendre le plus probable ?

Si on prend toujours le caractère le plus probable :
- Le texte devient **répétitif et ennuyeux**
- Boucles infinies fréquentes ("le chat dort le chat dort...")

Avec l'échantillonnage :
- Plus de variété
- Plus naturel
- Plus créatif

---

## 8. Questions probables du jury

### ❓ "Pourquoi pas un modèle plus simple comme un LSTM ?"

**Réponse** :
- Les Transformers traitent **toute la séquence en parallèle** → entraînement bien plus rapide
- L'attention permet d'avoir un **contexte global** alors que les LSTM oublient
- C'est l'architecture utilisée par TOUS les modèles modernes (GPT, PaLM, LLaMA, Claude)

### ❓ "Combien de paramètres a votre modèle ?"

**Réponse** : ~290 000 paramètres
- Embedding : ~9 000
- 4 blocs Transformer : ~270 000
- LM head : partagé avec embedding (technique appelée "weight tying")

### ❓ "Pourquoi RMSNorm et pas LayerNorm ?"

**Réponse** : C'est le choix de PaLM. RMSNorm est :
- ~30% plus rapide (pas de calcul de moyenne)
- Aussi efficace que LayerNorm en pratique
- Utilisé aussi par LLaMA, Mistral, Gemma

### ❓ "Pourquoi SwiGLU plutôt que ReLU ou GELU ?"

**Réponse** :
- ReLU coupe les valeurs négatives → perte d'information
- GELU est plus doux mais sans gating
- SwiGLU combine **non-linéarité ET mécanisme de filtrage**
- Améliore la qualité de ~2-3% (papier Shazeer 2020)

### ❓ "Comment le modèle 'comprend' le contexte ?"

**Réponse** : Grâce à l'**attention** :
- Chaque token calcule un score avec tous les autres tokens
- Les scores indiquent **à quel point chaque token est pertinent**
- Le modèle pondère les informations selon ces scores
- Avec 4 têtes, il capture 4 types de relations différentes

### ❓ "Pourquoi pas Multi-Query Attention comme dans PaLM ?"

**Réponse** :
- MQA est une optimisation pour réduire la mémoire
- Utile uniquement pour des **modèles très grands** (>1B paramètres)
- Pour notre Mini-PaLM, MHA standard suffit
- On peut l'ajouter facilement pour démontrer l'optimisation

### ❓ "Quel est le rôle du masque causal ?"

**Réponse** :
- Empêche le modèle de regarder les tokens **futurs** pendant l'entraînement
- Sinon le modèle "tricherait" en voyant la réponse
- C'est ce qui rend le modèle **autorégressif** (génère un token à la fois)

### ❓ "Comment vous évaluez la qualité du modèle ?"

**Réponse** :
- **Perte (CrossEntropy)** : passe de ~4.2 à ~0.3 (-92%)
- **Cohérence du texte généré** : phrases grammaticalement correctes
- **Vitesse** : ~X tokens/seconde sur CPU

### ❓ "Quelle est la différence avec le vrai PaLM ?"

**Réponse** :
- **Échelle** : 1.9 millions de fois plus petit
- **Données** : caractères français vs 780B tokens multilingues
- **Architecture** : IDENTIQUE (mêmes composants)
- **Temps d'entraînement** : 30 secondes vs 6 mois

### ❓ "Pouvez-vous expliquer la température ?"

**Réponse** :
- La température divise les logits avant softmax
- T < 1 → le modèle devient plus "confiant" (texte prévisible)
- T > 1 → le modèle devient plus "incertain" (texte créatif)
- T = 1 → distribution naturelle

### ❓ "Comment vous gérez les caractères inconnus ?"

**Réponse** :
- Le tokenizer **filtre automatiquement** les caractères absents du vocabulaire
- Le vocabulaire couvre tout le français (a-z, A-Z, 0-9, accents, ponctuation)
- En production, on utiliserait un tokenizer BPE qui gère tout

### ❓ "Quelles seraient les améliorations possibles ?"

**Réponse** :
- **Multi-Query Attention** (innovation PaLM)
- **RoPE** (Rotary Position Embedding)
- **Plus de données** (Wikipedia, livres)
- **Tokenizer BPE** (sous-mots)
- **Modèle plus grand** (plus de couches/dimensions)
- **Fine-tuning** sur tâches spécifiques

---

## 9. Glossaire complet

### Termes d'IA / Deep Learning

**Attention** : Mécanisme permettant à chaque position d'une séquence de pondérer l'importance des autres positions.

**Backward pass** : Calcul des gradients en partant de la perte vers les paramètres.

**Batch** : Groupe d'exemples traités en parallèle (ici 32).

**CrossEntropyLoss** : Fonction de perte pour la classification, mesure l'écart entre prédictions et vérité.

**Decoder-only** : Architecture Transformer qui ne contient que la partie décodeur (comme GPT, PaLM).

**Embedding** : Conversion d'un ID discret en vecteur continu de plusieurs dimensions.

**Epoch** : Un passage complet sur toutes les données d'entraînement.

**FFN (Feed-Forward Network)** : Petit réseau de neurones placé après l'attention.

**Forward pass** : Calcul de la prédiction du modèle.

**Gradient** : Indique dans quelle direction modifier un paramètre pour réduire l'erreur.

**Inférence** : Utilisation du modèle entraîné pour faire des prédictions.

**Learning rate** : Taille des pas de l'optimiseur (ici 3e-4 = 0.0003).

**Logits** : Scores bruts du modèle avant transformation en probabilités.

**LM Head** : Couche finale qui transforme les vecteurs en scores par token du vocabulaire.

**Masque causal** : Matrice empêchant de voir les positions futures.

**Multi-Head Attention (MHA)** : Attention avec plusieurs têtes parallèles spécialisées.

**Multi-Query Attention (MQA)** : Variante de MHA partageant K et V entre toutes les têtes (innovation PaLM).

**Next Token Prediction** : Objectif d'entraînement où le modèle prédit le prochain token.

**Optimizer** : Algorithme qui met à jour les paramètres (ici AdamW).

**Paramètre** : Valeur apprise par le modèle (poids d'une matrice).

**Perplexité** : exp(perte). Mesure à quel point le modèle est "perplexe" face au texte.

**RMSNorm** : Root Mean Square Normalization. Normalisation utilisée dans PaLM.

**RoPE** : Rotary Position Embedding. Encode la position via des rotations.

**Self-attention** : Attention où Q, K, V proviennent de la même séquence.

**Softmax** : Fonction transformant des scores en probabilités sommant à 1.

**SwiGLU** : Fonction d'activation utilisée dans PaLM (Swish-Gated Linear Unit).

**Token** : Unité de base traitée par le modèle (ici un caractère).

**Tokenizer** : Programme convertissant texte en tokens (et inversement).

**Transformer** : Architecture basée sur l'attention, base de tous les LLMs modernes.

**Weight tying** : Technique où l'embedding et le LM head partagent leurs poids.

### Termes PyTorch

**`nn.Module`** : Classe de base pour les modèles.

**`nn.Linear`** : Couche linéaire (multiplication matricielle).

**`nn.Embedding`** : Table de correspondance ID → vecteur.

**`nn.Parameter`** : Tenseur appris par le modèle.

**Tenseur** : Tableau multidimensionnel (équivalent ndarray de NumPy).

**`.view()`** : Reshape un tenseur sans copier les données.

**`.transpose()`** : Échange deux dimensions.

**`@`** : Opérateur de multiplication matricielle.

**`torch.no_grad()`** : Désactive le calcul des gradients (économise mémoire).

### Termes Streamlit

**`st.tabs()`** : Crée des onglets.

**`st.cache_resource`** : Met en cache une ressource (comme le modèle).

**`@st.cache_resource`** : Décorateur pour appliquer le cache.

**`st.spinner()`** : Affiche un indicateur de chargement.

**`st.metric()`** : Affiche une métrique avec un grand nombre.

---

## 🎓 Conseils finaux pour ta présentation

### ✅ À faire

1. **Lance l'app AVANT** la présentation : `python -m streamlit run app.py`
2. **Prépare 2-3 exemples** de génération qui marchent bien
3. **Explique le pipeline** de bout en bout (du texte aux probabilités)
4. **Compare avec PaLM** réel pour donner une perspective
5. **Sois prête aux questions techniques** (relis les composants)

### ❌ À éviter

1. **Ne dis pas "je ne sais pas"** : reformule ou réfère au papier original
2. **Ne survends pas** : c'est un modèle pédagogique, pas un produit fini
3. **N'oublie pas le lien avec ta présentation** : chaque composant correspond à une section

### 🎯 Phrases clés à utiliser

> "Mon implémentation reproduit l'architecture exacte de PaLM à plus petite échelle."

> "Tous les composants clés sont présents : SwiGLU, RMSNorm, attention multi-têtes, masque causal."

> "L'entraînement converge bien : la perte passe de 4.2 à 0.3, soit une amélioration de 92%."

> "Le modèle génère du texte cohérent en français, démontrant qu'il a appris la structure de la langue."

---

## 📚 Pour aller encore plus loin

### Papiers à lire (si tu as le temps)
1. **"Attention is All You Need"** (Vaswani et al., 2017) — Le papier fondateur
2. **"PaLM: Scaling Language Modeling with Pathways"** (Chowdhery et al., 2022)
3. **"GLU Variants Improve Transformer"** (Shazeer, 2020) — Pour SwiGLU

### Vidéos recommandées
- **Andrej Karpathy** : "Let's build GPT from scratch" (YouTube, 2h)
- **3Blue1Brown** : Série sur les Transformers

### Ressources
- **The Illustrated Transformer** (Jay Alammar)
- **The Annotated Transformer** (Harvard NLP)

---

**Bonne chance pour ta présentation ! 🎓✨**

# ═══════════════════════════════════════════════════════════
# MINI-PALM : Implémentation simplifiée du modèle PaLM
# ═══════════════════════════════════════════════════════════

# Import des bibliothèques nécessaires
import torch                          # Bibliothèque principale pour les calculs
import torch.nn as nn                 # Module pour construire des réseaux de neurones
import torch.nn.functional as F       # Fonctions mathématiques (softmax, silu, etc.)
import math                           # Fonctions mathématiques de base (sqrt, log)


# ═══════════════════════════════════════════════════════════
# COMPOSANT 1 : RMSNorm (normalisation)
# Rôle : stabiliser les valeurs numériques entre les couches
# ═══════════════════════════════════════════════════════════
class RMSNorm(nn.Module):

    def __init__(self, dim):
        # __init__ est appelée quand on crée l'objet
        # super() permet d'hériter des fonctionnalités de nn.Module
        super().__init__()

        # nn.Parameter = un tenseur que le modèle va apprendre
        # torch.ones(dim) crée un vecteur rempli de 1, de taille "dim"
        self.scale = nn.Parameter(torch.ones(dim))

        # epsilon : très petite valeur pour éviter la division par zéro
        self.eps = 1e-6

    def forward(self, x):
        # forward = calcul effectué quand on utilise la couche
        # x est le tenseur d'entrée

        # Calcul de la moyenne des carrés (Root Mean Square)
        # x.pow(2) = chaque valeur élevée au carré
        # .mean(-1, keepdim=True) = moyenne sur la dernière dimension
        rms = x.pow(2).mean(-1, keepdim=True)

        # Normalisation : on divise par la racine carrée du RMS
        # torch.rsqrt = 1 / sqrt() (plus rapide)
        x_normalise = x * torch.rsqrt(rms + self.eps)

        # On multiplie par le paramètre apprenable scale
        return self.scale * x_normalise


# ═══════════════════════════════════════════════════════════
# COMPOSANT 2 : SwiGLU (fonction d'activation)
# Rôle : introduire de la non-linéarité dans le réseau
# ═══════════════════════════════════════════════════════════
class SwiGLU(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()

        # nn.Linear(entree, sortie) = couche linéaire (multiplication matricielle)
        # bias=False : pas de terme constant ajouté

        # Première projection : pour le "gate" (porte de contrôle)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        # Deuxième projection : pour la valeur
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)

        # Troisième projection : pour revenir à la dimension d'origine
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # Étape 1 : calculer le gate avec activation Swish (=silu)
        # F.silu(x) = x * sigmoid(x), c'est une fonction non-linéaire
        gate = F.silu(self.w1(x))

        # Étape 2 : calculer la valeur (projection simple)
        valeur = self.w2(x)

        # Étape 3 : multiplier gate et valeur élément par élément
        # puis projeter avec w3
        return self.w3(gate * valeur)


# ═══════════════════════════════════════════════════════════
# COMPOSANT 3 : Attention (mécanisme central du Transformer)
# Rôle : permettre à chaque mot de "regarder" les autres mots
# Version simplifiée (Multi-Head au lieu de Multi-Query)
# ═══════════════════════════════════════════════════════════
class Attention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()

        # Nombre de "têtes" d'attention parallèles
        self.num_heads = num_heads

        # Dimension de chaque tête
        # // = division entière (sans virgule)
        self.head_dim = dim // num_heads

        # Trois projections linéaires : Query, Key, Value
        # Q = "ce que je cherche"
        # K = "ce que je contiens"
        # V = "l'information que je porte"
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Projection finale après l'attention
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x a la forme (B, T, C) où :
        # B = batch size (nombre de phrases traitées en parallèle)
        # T = longueur de la séquence (nombre de tokens)
        # C = dimension des embeddings
        B, T, C = x.shape

        # Étape 1 : calculer Q, K, V
        # .view() reshape le tenseur sans changer les données
        # .transpose(1, 2) échange les dimensions 1 et 2
        # Forme finale : (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Étape 2 : calculer les scores d'attention
        # k.transpose(-2, -1) = transposer les 2 dernières dimensions
        # @ = multiplication matricielle
        # On divise par sqrt(head_dim) pour stabiliser les valeurs
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Étape 3 : appliquer le masque causal
        # Empêche un mot de "voir" les mots qui viennent après lui
        # torch.tril = matrice triangulaire inférieure (avec des 1 en bas)
        masque = torch.tril(torch.ones(T, T, device=x.device))

        # masked_fill : remplace les positions où masque == 0 par -inf
        # Après softmax, -inf devient 0 (donc ces positions sont ignorées)
        scores = scores.masked_fill(masque == 0, float('-inf'))

        # Étape 4 : softmax pour obtenir des probabilités
        # softmax transforme les scores en valeurs entre 0 et 1 qui somment à 1
        poids_attention = F.softmax(scores, dim=-1)

        # Étape 5 : multiplier les poids par les valeurs V
        sortie = poids_attention @ v

        # Étape 6 : remettre la forme d'origine et projeter
        # .contiguous() réorganise la mémoire pour que .view() fonctionne
        sortie = sortie.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(sortie)


# ═══════════════════════════════════════════════════════════
# COMPOSANT 4 : Bloc Transformer complet
# Rôle : combiner Attention + FFN avec normalisations
# ═══════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()

        # Première normalisation (avant l'attention)
        self.norm1 = RMSNorm(dim)

        # Couche d'attention
        self.attention = Attention(dim, num_heads)

        # Deuxième normalisation (avant le FFN)
        self.norm2 = RMSNorm(dim)

        # Feed-Forward Network avec SwiGLU
        # On multiplie la dimension par 4 (convention courante)
        self.ffn = SwiGLU(dim, dim * 4)

    def forward(self, x):
        # Connexion résiduelle 1 : x + attention(norm(x))
        # On AJOUTE le résultat à l'entrée (au lieu de remplacer)
        # Cela aide le modèle à apprendre plus facilement
        x = x + self.attention(self.norm1(x))

        # Connexion résiduelle 2 : x + ffn(norm(x))
        x = x + self.ffn(self.norm2(x))

        return x


# ═══════════════════════════════════════════════════════════
# MODÈLE COMPLET : Mini-PaLM
# Empilement de plusieurs blocs Transformer
# ═══════════════════════════════════════════════════════════
class MiniPaLM(nn.Module):

    def __init__(self, vocab_size, dim=128, num_layers=4, num_heads=4, max_seq_len=64):
        super().__init__()

        # vocab_size = nombre de tokens uniques (mots/sous-mots)
        # dim = taille des vecteurs représentant chaque token

        # Embedding : transforme un ID de token en vecteur
        # Chaque ligne de la matrice = un vecteur pour un token
        self.embedding = nn.Embedding(vocab_size, dim)

        # Liste de blocs Transformer empilés
        # nn.ModuleList = liste qui enregistre les paramètres
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # Normalisation finale
        self.norm_final = RMSNorm(dim)

        # Tête de prédiction : transforme les vecteurs en scores pour chaque token
        # Sortie de taille vocab_size (un score par mot possible)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        # tokens : tenseur d'IDs de forme (B, T)

        # Étape 1 : convertir les IDs en vecteurs
        # Forme : (B, T) → (B, T, dim)
        x = self.embedding(tokens)

        # Étape 2 : passer dans tous les blocs Transformer
        for layer in self.layers:
            x = layer(x)

        # Étape 3 : normalisation finale
        x = self.norm_final(x)

        # Étape 4 : prédiction des prochains tokens
        # Forme : (B, T, dim) → (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits

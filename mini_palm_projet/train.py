# ═══════════════════════════════════════════════════════════
# ENTRAÎNEMENT DU MINI-PALM (VERSION AMÉLIORÉE)
# - Corpus enrichi en français
# - Sauvegarde de l'historique de perte
# - Mesure du temps d'entraînement
# ═══════════════════════════════════════════════════════════

import sys
import torch
import torch.nn as nn
import json
import time
from torch.utils.data import Dataset, DataLoader
from mini_palm import MiniPaLM
from corpus import get_corpus


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════
# TOKENIZER : caractère par caractère
# ═══════════════════════════════════════════════════════════
class CharTokenizer:

    def __init__(self, texte):
        # set() : ensemble sans doublons
        # sorted() : tri pour avoir un ordre stable
        self.chars = sorted(list(set(texte)))
        self.vocab_size = len(self.chars)

        # Dictionnaire caractère → ID
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}

        # Dictionnaire inverse ID → caractère
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encoder(self, texte):
        """Convertit une chaîne en liste d'IDs"""
        # On ignore les caractères inconnus pour éviter les erreurs
        return [self.char_to_id[ch] for ch in texte if ch in self.char_to_id]

    def decoder(self, ids):
        """Convertit une liste d'IDs en chaîne"""
        return ''.join([self.id_to_char[i] for i in ids])


# ═══════════════════════════════════════════════════════════
# DATASET : prépare les données pour l'entraînement
# ═══════════════════════════════════════════════════════════
class TextDataset(Dataset):

    def __init__(self, texte, tokenizer, seq_len=64):
        # Encoder tout le texte en une longue liste d'IDs
        self.donnees = torch.tensor(tokenizer.encoder(texte), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.donnees) - self.seq_len

    def __getitem__(self, idx):
        # x = séquence d'entrée
        x = self.donnees[idx : idx + self.seq_len]
        # y = séquence cible (décalée d'un caractère)
        y = self.donnees[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ═══════════════════════════════════════════════════════════
# FONCTION D'ENTRAÎNEMENT AMÉLIORÉE
# ═══════════════════════════════════════════════════════════
def entrainer():

    # ── Préparation du corpus ─────────────────────────────
    # Corpus français riche, répété 5 fois pour avoir plus de données
    texte = get_corpus(repetitions=5)

    print("=" * 60)
    print("📚 PRÉPARATION DES DONNÉES")
    print("=" * 60)
    print(f"Taille du texte : {len(texte):,} caractères")

    # ── Création du tokenizer ─────────────────────────────
    tokenizer = CharTokenizer(texte)
    print(f"Vocabulaire : {tokenizer.vocab_size} caractères uniques")

    # Sauvegarder le vocabulaire pour l'app
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({
            "chars": tokenizer.chars,
            "vocab_size": tokenizer.vocab_size
        }, f, ensure_ascii=False, indent=2)

    # ── Création du dataset ───────────────────────────────
    # seq_len plus grand (64) pour capturer plus de contexte
    seq_len = 64
    dataset = TextDataset(texte, tokenizer, seq_len=seq_len)
    print(f"Exemples d'entraînement : {len(dataset):,}")

    # DataLoader avec batch plus grand
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Batches par epoch : {len(loader):,}")

    # ── Création du modèle ────────────────────────────────
    print("\n" + "=" * 60)
    print("🏗️  CRÉATION DU MODÈLE")
    print("=" * 60)

    model = MiniPaLM(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=seq_len
    )

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux : {nb_params:,}")

    # ── Optimiseur ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # ── Détection automatique CPU/GPU ─────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Appareil utilisé : {device.upper()}")

    # ── Boucle d'entraînement avec historique ─────────────
    print("\n" + "=" * 60)
    print("🚀 ENTRAÎNEMENT EN COURS")
    print("=" * 60)

    nombre_epochs = 8

    # Historique pour les graphiques
    historique = {
        "loss_par_step": [],      # Perte à chaque step
        "loss_par_epoch": [],     # Perte moyenne par epoch
        "temps_par_epoch": [],    # Temps de chaque epoch
        "epochs": list(range(1, nombre_epochs + 1)),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "learning_rate": 3e-4,
            "nb_params": nb_params,
            "device": device
        }
    }

    temps_debut_total = time.time()

    for epoch in range(nombre_epochs):
        temps_debut_epoch = time.time()
        perte_totale = 0
        nombre_batches = 0

        for step, (x, y) in enumerate(loader):
            # Déplacer sur le bon appareil (CPU/GPU)
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)

            # Calcul de la perte
            perte = criterion(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1)
            )

            # Backward pass + mise à jour
            optimizer.zero_grad()
            perte.backward()

            # Gradient clipping (évite les gradients explosifs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Sauvegarder dans l'historique
            historique["loss_par_step"].append(perte.item())
            perte_totale += perte.item()
            nombre_batches += 1

            # Affichage périodique
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{nombre_epochs} | "
                      f"Step {step:4d}/{len(loader)} | "
                      f"Perte : {perte.item():.4f}")

        # Statistiques de l'epoch
        perte_moyenne = perte_totale / nombre_batches
        temps_epoch = time.time() - temps_debut_epoch

        historique["loss_par_epoch"].append(perte_moyenne)
        historique["temps_par_epoch"].append(temps_epoch)

        print(f"\n→ Fin epoch {epoch+1} | "
              f"Perte moyenne : {perte_moyenne:.4f} | "
              f"Temps : {temps_epoch:.1f}s\n")

    temps_total = time.time() - temps_debut_total
    historique["temps_total"] = temps_total

    print("=" * 60)
    print(f"✅ ENTRAÎNEMENT TERMINÉ EN {temps_total:.1f}s")
    print("=" * 60)

    # ── Sauvegarde du modèle ──────────────────────────────
    # Remettre le modèle sur CPU pour la sauvegarde
    model = model.to("cpu")
    torch.save(model.state_dict(), "mini_palm.pt")
    print("📁 Modèle sauvegardé : mini_palm.pt")

    # ── Sauvegarde de l'historique ────────────────────────
    with open("historique.json", "w", encoding="utf-8") as f:
        json.dump(historique, f, indent=2)
    print("📁 Historique sauvegardé : historique.json")

    return model, tokenizer, historique


# ═══════════════════════════════════════════════════════════
# GÉNÉRATION DE TEXTE (avec mesure du temps)
# ═══════════════════════════════════════════════════════════
def generer(model, tokenizer, debut="Le chat", longueur=100, temperature=0.8):
    """Génère du texte et mesure le temps d'inférence"""
    model.eval()

    # Filtrer les caractères inconnus dans le début
    debut_filtre = ''.join([c for c in debut if c in tokenizer.char_to_id])

    if not debut_filtre:
        return "Erreur : aucun caractère valide dans le début"

    tokens = tokenizer.encoder(debut_filtre)
    tokens = torch.tensor(tokens).unsqueeze(0)

    # Mesure du temps
    temps_debut = time.time()

    with torch.no_grad():
        for _ in range(longueur):
            logits = model(tokens)
            dernier_logits = logits[:, -1, :] / temperature
            probas = torch.softmax(dernier_logits, dim=-1)
            prochain_token = torch.multinomial(probas, num_samples=1)
            tokens = torch.cat([tokens, prochain_token], dim=1)

    temps_inference = time.time() - temps_debut

    texte_genere = tokenizer.decoder(tokens[0].tolist())

    return {
        "texte": texte_genere,
        "temps": temps_inference,
        "tokens_par_seconde": longueur / temps_inference if temps_inference > 0 else 0
    }


# ═══════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Entraîner le modèle
    model, tokenizer, historique = entrainer()

    # Tester la génération
    print("\n" + "=" * 60)
    print("🎲 TESTS DE GÉNÉRATION")
    print("=" * 60)

    debuts_test = ["Le chat", "Marie habite", "L'hiver", "La France"]

    for debut in debuts_test:
        resultat = generer(model, tokenizer, debut=debut, longueur=80)
        print(f"\n📝 Début : '{debut}'")
        print(f"⏱️  Temps : {resultat['temps']*1000:.1f}ms ({resultat['tokens_par_seconde']:.1f} tokens/s)")
        print(f"Texte : {resultat['texte']}")

    print("\n" + "=" * 60)
    print(f"📊 STATISTIQUES FINALES")
    print("=" * 60)
    print(f"Perte initiale : {historique['loss_par_epoch'][0]:.4f}")
    print(f"Perte finale   : {historique['loss_par_epoch'][-1]:.4f}")
    amelioration = (1 - historique['loss_par_epoch'][-1] / historique['loss_par_epoch'][0]) * 100
    print(f"Amélioration   : {amelioration:.1f}%")
    print(f"Temps total    : {historique['temps_total']:.1f}s")

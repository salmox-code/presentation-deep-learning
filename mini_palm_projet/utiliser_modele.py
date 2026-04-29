# ═══════════════════════════════════════════════════════════
# UTILISER LE MODÈLE DÉJÀ ENTRAÎNÉ
# (À lancer APRÈS avoir exécuté train.py)
# ═══════════════════════════════════════════════════════════

import sys
import torch
import time
from mini_palm import MiniPaLM
from train import CharTokenizer, generer
from corpus import get_corpus


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════
# Étape 1 : Recréer le tokenizer avec le MÊME corpus
# ═══════════════════════════════════════════════════════════
texte = get_corpus(repetitions=5)
tokenizer = CharTokenizer(texte)
print(f"Vocabulaire chargé : {tokenizer.vocab_size} caractères")


# ═══════════════════════════════════════════════════════════
# Étape 2 : Recréer le modèle avec les MÊMES paramètres
# ═══════════════════════════════════════════════════════════
model = MiniPaLM(
    vocab_size=tokenizer.vocab_size,
    dim=128,
    num_layers=4,
    num_heads=4,
    max_seq_len=64
)


# ═══════════════════════════════════════════════════════════
# Étape 3 : Charger les poids sauvegardés
# ═══════════════════════════════════════════════════════════
try:
    model.load_state_dict(torch.load("mini_palm.pt", map_location="cpu"))
    model.eval()
    print("✅ Modèle chargé avec succès !\n")
except FileNotFoundError:
    print("❌ Fichier mini_palm.pt non trouvé.")
    print("   Lance d'abord : python train.py")
    exit(1)


# ═══════════════════════════════════════════════════════════
# Étape 4 : Tests de génération
# ═══════════════════════════════════════════════════════════
debuts_a_tester = [
    "Le chat",
    "Marie habite",
    "L'hiver",
    "La France",
    "Les enfants"
]

print("=" * 60)
print("TESTS AUTOMATIQUES")
print("=" * 60)

for debut in debuts_a_tester:
    print(f"\n📝 Début : '{debut}'")
    print("-" * 60)
    resultat = generer(model, tokenizer, debut=debut, longueur=100, temperature=0.8)
    print(f"⏱️  {resultat['temps']*1000:.0f}ms ({resultat['tokens_par_seconde']:.0f} tokens/s)")
    print(resultat['texte'])


# ═══════════════════════════════════════════════════════════
# Étape 5 : Mode interactif
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODE INTERACTIF")
print("=" * 60)
print("Commandes :")
print("  - Tape un début de phrase pour générer du texte")
print("  - Tape 'temp X' pour changer la température (ex: temp 1.5)")
print("  - Tape 'quit' pour quitter")
print()

temperature_courante = 0.8

while True:
    entree = input(f"\n[T={temperature_courante}] Ton début : ").strip()

    if entree.lower() == "quit":
        print("Au revoir !")
        break

    # Commande pour changer la température
    if entree.lower().startswith("temp "):
        try:
            nouvelle_temp = float(entree.split()[1])
            if 0.1 <= nouvelle_temp <= 2.0:
                temperature_courante = nouvelle_temp
                print(f"✅ Température réglée à {temperature_courante}")
            else:
                print("⚠️ La température doit être entre 0.1 et 2.0")
        except (ValueError, IndexError):
            print("⚠️ Format : temp 0.8")
        continue

    if not entree:
        continue

    resultat = generer(model, tokenizer, debut=entree, longueur=120,
                      temperature=temperature_courante)
    print(f"\n⏱️  {resultat['temps']*1000:.0f}ms")
    print(f"📝 {resultat['texte']}")

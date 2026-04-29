# ═══════════════════════════════════════════════════════════
# CORPUS D'ENTRAÎNEMENT ENRICHI
# Texte français varié avec majuscules, nombres, accents, ponctuation
# ═══════════════════════════════════════════════════════════

CORPUS = """
Le chat dort sur le canapé du salon. Le chien court dans le grand jardin fleuri.
Les oiseaux chantent dans les arbres au lever du soleil. Le soleil brille dans le ciel bleu.
Les enfants jouent dans le parc avec leurs amis. La lune éclaire la nuit étoilée.

Marie habite à Paris depuis 5 ans. Elle travaille dans une grande entreprise.
Pierre étudie à l'université de Lyon. Il a 22 ans et passe son diplôme en juin.
La famille Martin vit dans une maison à la campagne. Ils ont 3 enfants et 2 chiens.

Le matin, je bois un café noir avec du sucre. Je mange une tartine avec du beurre.
À midi, nous mangeons souvent au restaurant. Le menu coûte 15 euros environ.
Le soir, la famille se réunit pour le dîner. On parle de notre journée à table.

L'hiver est froid en France. Il neige souvent en montagne entre décembre et mars.
Le printemps apporte les fleurs et le retour des oiseaux. Les jours rallongent vite.
L'été est chaud et ensoleillé. Les vacances commencent en juillet pour 2 mois.
L'automne colore les feuilles en rouge, jaune et orange. La pluie tombe souvent.

La France compte 67 millions d'habitants. Sa capitale est Paris depuis longtemps.
Le Louvre est le musée le plus visité au monde. Il contient 35000 œuvres environ.
La tour Eiffel mesure 330 mètres de haut. Elle a été construite en 1889 par Gustave Eiffel.

Les chats aiment dormir 16 heures par jour. Ils chassent les souris la nuit.
Les chiens sont des animaux fidèles et sociables. Ils protègent leur famille.
Les oiseaux migrent vers le sud en hiver. Ils reviennent au printemps faire leur nid.

Le sport est important pour la santé. Il faut marcher 30 minutes par jour minimum.
Le football est le sport le plus populaire en France. L'équipe de France a gagné 2 fois la coupe.
Le tennis se joue sur 3 surfaces différentes : terre battue, gazon et dur.

L'école commence à 8 heures et finit à 17 heures. Les élèves apprennent les mathématiques.
La lecture développe l'imagination et le vocabulaire. Il faut lire au moins 1 livre par mois.
L'histoire de France est riche et passionnante. Napoléon est devenu empereur en 1804.

La musique adoucit les mœurs et apaise l'esprit. Mozart a composé plus de 600 œuvres.
Le cinéma français est connu dans le monde entier. Le festival de Cannes a lieu en mai.
La peinture impressionniste est née en France au 19e siècle. Monet et Renoir sont célèbres.

La cuisine française est réputée internationalement. Le pain et le fromage sont essentiels.
Les croissants sont une spécialité du petit déjeuner. On les mange chauds avec du beurre.
Le vin accompagne les repas en France. Il existe plus de 3000 appellations différentes.

Les nouvelles technologies changent notre vie. Internet connecte 5 milliards de personnes.
L'intelligence artificielle progresse rapidement. Les modèles de langage écrivent comme des humains.
Les ordinateurs calculent des milliards d'opérations par seconde. Leur puissance double tous les 2 ans.

Le voyage ouvre l'esprit et enrichit la culture. Il faut découvrir au moins 10 pays dans sa vie.
Le Japon est connu pour ses traditions et sa modernité. Tokyo compte 14 millions d'habitants.
L'Italie attire les touristes avec Rome, Venise et Florence. La cuisine y est délicieuse.

La nature est belle et fragile à la fois. Il faut la protéger pour les générations futures.
Les forêts couvrent 30% de la surface de la Terre. Elles produisent l'oxygène que nous respirons.
Les océans contiennent 97% de l'eau de la planète. La biodiversité marine est immense.

Apprendre une langue demande du temps et de la pratique. Il faut étudier 1 heure par jour.
Le français est parlé par 300 millions de personnes dans le monde. C'est la 5e langue mondiale.
L'anglais est devenu la langue internationale des affaires et des sciences.

Les amis sont précieux dans la vie. Une vraie amitié dure des années sans jamais faiblir.
La famille est le pilier de notre existence. Les parents nous soutiennent toujours.
L'amour donne un sens à la vie et nous rend plus heureux chaque jour passé ensemble.
"""


def get_corpus(repetitions=3):
    """
    Retourne le corpus, éventuellement répété pour avoir plus de données.

    Args:
        repetitions: nombre de fois où le corpus est répété
    """
    return CORPUS * repetitions


if __name__ == "__main__":
    # Statistiques sur le corpus
    texte = get_corpus(repetitions=1)

    print("=" * 60)
    print("STATISTIQUES DU CORPUS")
    print("=" * 60)
    print(f"Nombre de caractères : {len(texte):,}")
    print(f"Nombre de caractères uniques : {len(set(texte))}")
    print(f"Nombre de lignes : {len(texte.split(chr(10)))}")
    print(f"Nombre de mots : {len(texte.split())}")
    print()
    print("Caractères du vocabulaire :")
    print(sorted(set(texte)))

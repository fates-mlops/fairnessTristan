# UseCases

## 1_data

Contient les données utilisées.

### 1_raw

Contient la base de donnée originale et une explication sur son origine.

Indications dans les métadonnées -> Objectif :
- Source des données originales -> Reproductibilité

### 2_preprocessed

Contient les données nettoyées, et une éventuelle indication sur la stratégie d'échantillonnage.

Indications dans les métadonnées -> Objectif :
- Nom de l'échantillon -> Chargement du fichier
- Méthode d'échantillonnage -> Reproductibilité

### 3_split

Contient les données séparées pour l'entrainement et le test et une indication sur les paramètres utilisés pour cette séparation.

Indications dans les métadonnées -> Objectif :
- Nom de l'échantillon chargé -> Reproductibilité
- Colonne cible -> Reproductibilité (permet de définir X et y)
- Taille du test/de l'entrainement -> Reproductibilité
- Graine aléatoire -> Reproductibilité

## 2_models

Contient le modèle entrainé et des indications sur ses hyperparamètres.

Indications dans les métadonnées -> Objectif :
- Type de modèle de ML -> Reproductibilité
- Hyperparamètres -> Reproductibilité
- Métriques -> Critique du modèle (pas nécessaire si on les calcule sur le moment à l'aide des échantillons stockés)

## 3_exigences

Contient le modèle de justification, la description des exigences, et les fonctions de mesure nécessaires.

Indications dans les exigeances -> Objectif :
- Nom de l'exigence -> Différencier des exigeances dans un ensemble d'exigeances
- Fonction -> Accès à un script pour satisfaire l'exigeance
- Variable sensible -> Nécessaire à la mesure de fairness
- Seuil et type de seuil -> Donner une réponse binaire à l'exigeance (satisfait/pas satisfait) à partir de la mesure

## 4_notebooks

Contient le code python

## jpipe-libs

Contient "steps.py", les validations pour le déploiement, et test.ipynb, le test des différentes vérifications dans steps.py.
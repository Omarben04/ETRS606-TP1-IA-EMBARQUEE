# ETRS606 — TP1 : Classification de chiffres manuscrits MNIST avec un MLP

> **Module :** ETRS606 — Intelligence Artificielle Embarquée  
> **Sujet :** Étude et implémentation d'un réseau de neurones dense (MLP) pour la classification du dataset MNIST  
> **Plateforme :** Google Colab / TensorFlow 2.x  
> **Niveau :** Licence 3 TRI — Université Savoie Mont Blanc (USMB)

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Dataset MNIST — Comprendre les données](#2-dataset-mnist--comprendre-les-données)
   - [Description](#21-description)
   - [Pourquoi prétraiter les données ?](#22-pourquoi-prétraiter-les-données-)
   - [Chargement et préparation](#23-chargement-et-préparation)
3. [Notions théoriques fondamentales](#3-notions-théoriques-fondamentales)
   - [Le MLP (Multilayer Perceptron)](#31-le-mlp-multilayer-perceptron)
   - [Les fonctions d'activation](#32-les-fonctions-dactivation)
   - [Les optimiseurs](#33-les-optimiseurs)
   - [Les fonctions de coût](#34-les-fonctions-de-coût)
4. [Expériences et résultats](#4-expériences-et-résultats)
   - [Modèle 1 — Sans couche cachée (baseline)](#41-modèle-1--sans-couche-cachée-baseline)
   - [Modèle 2 — Activation ReLU](#42-modèle-2--activation-relu)
   - [Modèle 3 — Activation tanh](#43-modèle-3--activation-tanh)
   - [Modèle 4 — Activation sigmoid](#44-modèle-4--activation-sigmoid)
   - [Étude des optimiseurs](#45-étude-des-optimiseurs)
   - [Étude des fonctions de coût](#46-étude-des-fonctions-de-coût)
5. [Tableau comparatif global](#5-tableau-comparatif-global)
6. [Analyse mémoire et contraintes embarquées](#6-analyse-mémoire-et-contraintes-embarquées)
7. [Conclusion générale](#7-conclusion-générale)
8. [Références](#8-références)

---

## 1. Contexte et objectifs

Ce TP s'inscrit dans le module ETRS606 dédié à l'**intelligence artificielle embarquée**. L'objectif est de comprendre et d'expérimenter les mécanismes fondamentaux qui gouvernent les performances d'un réseau de neurones dense, aussi appelé **MLP (Multilayer Perceptron)**.

Le dataset utilisé est **MNIST**, un benchmark classique en apprentissage automatique, qui consiste à reconnaître des chiffres manuscrits (0 à 9) à partir d'images en niveaux de gris 28×28 pixels.

Les questions abordées dans ce TP sont les suivantes :

- Quel est l'impact du **nombre de couches cachées** sur la précision ?
- Quel rôle jouent les **fonctions d'activation** (ReLU, tanh, sigmoid) ?
- Comment le choix de l'**algorithme d'optimisation** influence-t-il la convergence et les performances finales ?
- Quelle **fonction de coût** est la plus adaptée à ce problème de classification multi-classes ?
- Comment trouver un **compromis précision / complexité** dans un contexte d'IA embarquée ?

---

## 2. Dataset MNIST — Comprendre les données

### 2.1 Description

MNIST (Modified National Institute of Standards and Technology) est l'un des datasets les plus utilisés en deep learning. Il contient des images de chiffres manuscrits, répartis comme suit :

| Ensemble | Nombre d'images |
|---|---|
| Entraînement (`x_train`) | 60 000 |
| Test (`x_test`) | 10 000 |
| **Total** | **70 000** |

Chaque image est :
- En **niveaux de gris** (1 canal)
- De taille **28 × 28 pixels**
- Associée à un **label entier** de 0 à 9 (la classe du chiffre représenté)

### 2.2 Pourquoi prétraiter les données ?

Un MLP (réseau dense) ne peut pas recevoir une image brute 2D en entrée. Deux opérations sont nécessaires avant l'entraînement :

**Normalisation des pixels :**

Les valeurs de pixels vont de 0 (noir) à 255 (blanc). Un réseau de neurones apprend mieux lorsque les valeurs d'entrée sont dans une **plage raisonnable et homogène**. On divise donc chaque pixel par 255 pour obtenir des valeurs entre 0 et 1.

```
valeur_normalisée = valeur_pixel / 255.0
```

Sans normalisation, les gradients peuvent exploser ou disparaître pendant la rétropropagation, rendant l'apprentissage instable.

**Aplatissement (flatten) :**

Le MLP prend un **vecteur 1D** en entrée, pas une matrice 2D. Chaque image 28×28 est donc transformée en un vecteur de **784 valeurs** (784 = 28 × 28). Chaque pixel devient une **feature** (caractéristique d'entrée).

```
Image (28, 28)  →  Vecteur (784,)
```

> Note : Cette opération entraîne une perte de l'information spatiale (voisinage des pixels). C'est pourquoi les CNN (Convolutional Neural Networks) sont plus adaptés aux images — mais ce TP se concentre sur les MLP.

### 2.3 Chargement et préparation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Chargement du dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Forme brute x_train :", x_train.shape)   # (60000, 28, 28)
print("Forme brute y_train :", y_train.shape)   # (60000,)

# Normalisation
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Aplatissement
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

print("Nouvelle forme x_train :", x_train.shape)  # (60000, 784)

# Encodage one-hot des labels (pour categorical_crossentropy)
# Exemple : label 3  →  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test, 10)
```

**Sortie attendue :**
```
Forme brute x_train   : (60000, 28, 28)
Forme brute y_train   : (60000,)
Nouvelle forme x_train: (60000, 784)
```

---

## 3. Notions théoriques fondamentales

### 3.1 Le MLP (Multilayer Perceptron)

Un MLP est un réseau de neurones **entièrement connecté** (fully connected / dense). Il est composé de :

```
Couche d'entrée  →  [Couches cachées]  →  Couche de sortie
    (784)              (optionnel)             (10)
```

Chaque neurone d'une couche est connecté à **tous les neurones** de la couche suivante. Le calcul effectué par un neurone est :

```
sortie = activation( W · x + b )
```

Où :
- `W` = matrice de poids (appris pendant l'entraînement)
- `x` = vecteur d'entrée
- `b` = vecteur de biais
- `activation` = fonction d'activation non-linéaire

**Calcul du nombre de paramètres entre deux couches :**

```
paramètres = (neurones_entrée × neurones_sortie) + neurones_sortie
           =  n_in × n_out  +  n_out  (biais)
```

Exemple pour `784 → 128` :
```
784 × 128 + 128 = 100 480 paramètres
```

### 3.2 Les fonctions d'activation

Les fonctions d'activation introduisent de la **non-linéarité** dans le réseau, ce qui lui permet d'apprendre des relations complexes entre les données.

#### ReLU — Rectified Linear Unit

```
ReLU(x) = max(0, x)
```

- Sortie entre **0 et +∞**
- Annule toutes les valeurs négatives
- Très simple à calculer → **rapide**
- Peut provoquer des « neurones morts » (dying ReLU) si trop de valeurs sont négatives
- **Choix par défaut** pour les couches cachées dans la plupart des architectures modernes

| x | ReLU(x) |
|---|---|
| -5 | 0 |
| 0 | 0 |
| 2 | 2 |
| 10 | 10 |

#### tanh — Tangente hyperbolique

```
tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

- Sortie entre **-1 et +1**
- **Centrée autour de 0** → favorise une meilleure convergence que sigmoid
- Peut saturer (gradients proches de 0) pour des valeurs très grandes ou très petites
- Utilisée historiquement avant ReLU

| x | tanh(x) |
|---|---|
| -3 | ≈ -0.995 |
| 0 | 0 |
| 2 | ≈ 0.964 |

#### Sigmoid

```
sigmoid(x) = 1 / (1 + e^-x)
```

- Sortie entre **0 et 1**
- Souvent utilisée pour les couches de sortie en classification binaire
- Sur les couches cachées : **saturation fréquente** → gradients qui disparaissent
- Moins efficace que tanh ou ReLU sur les réseaux profonds

| x | sigmoid(x) |
|---|---|
| -4 | ≈ 0.018 |
| 0 | 0.5 |
| 3 | ≈ 0.952 |

#### Softmax (couche de sortie)

```
softmax(x_i) = e^x_i / Σ e^x_j
```

- Utilisée **uniquement en sortie** pour la classification multi-classes
- Transforme les scores bruts en **probabilités** (somme = 1)
- Le neurone avec la plus haute probabilité correspond à la classe prédite

#### Comparatif synthétique

| Fonction | Plage | Centrée en 0 | Saturante | Usage typique |
|---|---|---|---|---|
| ReLU | [0, +∞] | Non | Non | Couches cachées (moderne) |
| tanh | [-1, +1] | Oui | Oui | Couches cachées (classique) |
| sigmoid | [0, +1] | Non | Oui | Sortie binaire / historique |
| softmax | [0, 1] | Non | Non | Sortie multi-classes |

### 3.3 Les optimiseurs

L'optimiseur est l'algorithme qui **met à jour les poids** du réseau après chaque batch. Il contrôle la descente de gradient.

Pendant l'entraînement :
```
1. Le réseau prédit une sortie
2. La fonction de coût mesure l'erreur
3. La rétropropagation calcule le gradient de l'erreur par rapport à chaque poids
4. L'optimiseur modifie les poids pour réduire l'erreur
```

#### SGD — Stochastic Gradient Descent

```
w = w - lr × gradient
```

- Algorithme de base, simple
- Taux d'apprentissage fixe
- Convergence **lente** mais stable
- Peut rester bloqué dans des minima locaux

#### Adam — Adaptive Moment Estimation

- Combine les avantages de **Momentum** (direction) et de **RMSprop** (taux adaptatif)
- Adapte le taux d'apprentissage **individuellement pour chaque paramètre**
- Convergence **rapide** et **stable**
- **Choix par défaut** dans la plupart des projets modernes

#### RMSprop — Root Mean Square Propagation

- Adapte le taux d'apprentissage en divisant par la moyenne mobile des gradients carrés
- Très efficace pour les données **non-stationnaires**
- Souvent légèrement meilleur qu'Adam dans certains cas

#### Adagrad — Adaptive Gradient

- Accumule les gradients carrés depuis le début de l'entraînement
- Le taux d'apprentissage **diminue continuellement**
- Efficace pour les données **creuses** (sparse features)
- Peut s'arrêter prématurément sur des tâches longues → **sous-performant** ici

### 3.4 Les fonctions de coût

La fonction de coût (loss) **mesure l'écart** entre la prédiction du modèle et la vraie valeur. L'optimiseur cherche à minimiser cette valeur.

#### categorical_crossentropy

```
L = -Σ y_true × log(y_pred)
```

- Labels doivent être en format **one-hot** : `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`
- Nécessite `keras.utils.to_categorical()` en amont
- Pénalise fortement les prédictions très confiantes mais fausses

#### sparse_categorical_crossentropy

- Même formule mathématique que `categorical_crossentropy`
- Labels fournis directement comme **entiers** : `2`, `7`, `4`...
- **Avantage** : aucun encodage one-hot nécessaire → code plus simple et léger
- Légèrement plus efficace en mémoire pour les datasets avec beaucoup de classes

---

## 4. Expériences et résultats

> Toutes les expériences utilisent :
> - **10 epochs** d'entraînement
> - **batch_size = 32**
> - **validation_split = 0.1** (10% des données d'entraînement servent à valider)
> - Évaluation finale sur les 10 000 images de test

---

### 4.1 Modèle 1 — Sans couche cachée (baseline)

**Architecture :**
```
Entrée (784) → Dense(10, softmax)
```

```python
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

**Résumé du modèle :**

```
Layer (type)        Output Shape    Param #
dense (Dense)       (None, 10)      7,850
─────────────────────────────────────────
Total params: 7,850 (30.66 KB)
```

**Calcul détaillé des paramètres :**

```
784 entrées × 10 neurones = 7 840 poids
+ 10 biais
─────────────────────────
= 7 850 paramètres
```

**Évolution de l'entraînement :**

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 81.1 % | 92.8 % |
| 5 | 92.2 % | 93.8 % |
| 10 | 92.8 % | 94.0 % |

**Résultats finaux :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.2670 |
| **Accuracy test** | **92.45 %** |

**Analyse :**

Ce réseau ne possède **aucune couche cachée**. Il ne peut apprendre que des **relations linéaires** entre les pixels et les classes. La transformation effectuée est simplement :

```
sortie = softmax( W × image + b )
```

Les 10 neurones de sortie apprennent chacun un **modèle linéaire** d'un chiffre. C'est pourquoi la précision plafonne à ~92 %. Pour dépasser cette limite, il faut des couches cachées qui permettent d'apprendre des **représentations non-linéaires**.

Remarque : train accuracy (92.8 %) ≈ val accuracy (94.0 %) → **pas d'overfitting**.

---

### 4.2 Modèle 2 — Activation ReLU

#### 2 couches cachées : 784 → 128 → 64 → 10

```python
model_relu_2 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

**Résumé du modèle :**

```
Layer (type)        Output Shape    Param #
dense_1 (Dense)     (None, 128)     100,480
dense_2 (Dense)     (None, 64)      8,256
dense_3 (Dense)     (None, 10)      650
────────────────────────────────────────────
Total params: 109,386 (427.29 KB)
```

**Calcul des paramètres :**

```
784 × 128 + 128 = 100 480   (couche 1)
128 ×  64 +  64 =   8 256   (couche 2)
 64 ×  10 +  10 =     650   (couche sortie)
─────────────────────────
Total = 109 386 paramètres
```

**Résultats finaux :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.0931 |
| **Accuracy test** | **97.59 %** |

#### 3 couches cachées : 784 → 256 → 128 → 64 → 10

```python
model_relu_3 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

**Résumé du modèle :**

```
Layer (type)        Output Shape    Param #
dense_4 (Dense)     (None, 256)     200,960
dense_5 (Dense)     (None, 128)     32,896
dense_6 (Dense)     (None, 64)      8,256
dense_7 (Dense)     (None, 10)      650
────────────────────────────────────────────
Total params: 242,762 (948.29 KB)
```

**Résultats finaux :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.1076 |
| **Accuracy test** | **97.70 %** |

**Analyse ReLU :**

| Architecture | Params | Mémoire | Accuracy test | Gain vs baseline |
|---|---|---|---|---|
| Sans couche cachée | 7 850 | 30 KB | 92.45 % | — |
| 2 couches (128-64) | 109 386 | 427 KB | 97.59 % | **+5.14 pts** |
| 3 couches (256-128-64) | 242 762 | 948 KB | 97.70 % | +0.11 pts |

> L'ajout d'une 3ème couche n'apporte que **+0.11 %** de précision pour un coût en paramètres **×2.2**. Ce compromis est défavorable, surtout en contexte embarqué.

---

### 4.3 Modèle 3 — Activation tanh

#### 1 couche cachée : 784 → 128 → 10

```python
model_tanh_1 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="tanh"),
    layers.Dense(10, activation="softmax")
])
```

**Résultats :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.0747 |
| **Accuracy test** | **97.80 %** |

**Évolution de l'entraînement :**

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 86.7 % | 95.7 % |
| 5 | 98.6 % | 97.6 % |
| 10 | 99.7 % | 98.2 % |

> On observe un léger début d'overfitting à partir de l'epoch 7-8 (train ≈ 99.4 %, val ≈ 98 %).

#### 2 couches cachées : 784 → 128 → 64 → 10

```python
model_tanh_2 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="tanh"),
    layers.Dense(64, activation="tanh"),
    layers.Dense(10, activation="softmax")
])
```

**Résultats :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.0848 |
| **Accuracy test** | **97.86 %** |

**Comparatif tanh :**

| Architecture | Accuracy test |
|---|---|
| 1 couche cachée (128) | 97.80 % |
| 2 couches cachées (128-64) | 97.86 % |

> La 2ème couche apporte **+0.06 %** de précision. L'amélioration est marginale.

---

### 4.4 Modèle 4 — Activation sigmoid

#### 1 couche cachée : 784 → 128 → 10

```python
model_sigmoid_1 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="sigmoid"),
    layers.Dense(10, activation="softmax")
])
```

**Résultats :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.0751 |
| **Accuracy test** | **97.60 %** |

**Évolution de l'entraînement :**

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 81.8 % | 94.3 % |
| 5 | 97.2 % | 97.5 % |
| 10 | 99.1 % | 97.9 % |

> Sigmoid converge **plus lentement** que tanh à l'epoch 1 (81.8 % vs 86.7 %) en raison de sa sortie non centrée en 0.

#### 2 couches cachées : 784 → 128 → 64 → 10

```python
model_sigmoid_2 = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="sigmoid"),
    layers.Dense(64, activation="sigmoid"),
    layers.Dense(10, activation="softmax")
])
```

**Résultats :**

| Métrique | Valeur |
|---|---|
| Loss test | 0.0765 |
| **Accuracy test** | **97.75 %** |

**Comparatif global des fonctions d'activation (2 couches cachées 128-64) :**

| Activation | Accuracy test | Remarque |
|---|---|---|
| ReLU | 97.59 % | Rapide, efficace |
| tanh | 97.86 % | Légèrement supérieure ici |
| sigmoid | 97.75 % | Convergence plus lente |

> Sur MNIST, les trois fonctions donnent des résultats très proches. **ReLU reste le meilleur choix pratique** en général car plus efficace sur des réseaux plus profonds.

---

### 4.5 Étude des optimiseurs

**Architecture fixe pour cette étude :**
```
784 → Dense(128, tanh) → Dense(64, tanh) → Dense(10, softmax)
Loss : categorical_crossentropy
```

```python
optimizers = {
    "SGD":      keras.optimizers.SGD(),
    "Adam":     keras.optimizers.Adam(),
    "RMSprop":  keras.optimizers.RMSprop(),
    "Adagrad":  keras.optimizers.Adagrad()
}
```

**Résultats détaillés :**

#### SGD

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 73.97 % | 91.42 % |
| 5 | 93.35 % | 94.83 % |
| 10 | 95.23 % | 96.25 % |

```
Loss test : 0.1615
Accuracy test : 95.16 %
```

#### Adam

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 86.97 % | 96.10 % |
| 5 | 98.60 % | 97.75 % |
| 10 | 99.66 % | 97.58 % |

```
Loss test : 0.0928
Accuracy test : 97.62 %
```

#### RMSprop

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 87.96 % | 96.27 % |
| 5 | 98.33 % | 98.08 % |
| 10 | 99.54 % | 98.03 % |

```
Loss test : 0.0830
Accuracy test : 97.99 %  ← meilleur résultat
```

#### Adagrad

| Epoch | Accuracy train | Accuracy val |
|---|---|---|
| 1 | 57.18 % | 87.82 % |
| 5 | 89.06 % | 91.92 % |
| 10 | 90.69 % | 92.90 % |

```
Loss test : 0.3153
Accuracy test : 91.66 %  ← moins performant
```

**Tableau récapitulatif :**

| Optimiseur | Loss test | Accuracy test | Accuracy val (finale) | Convergence |
|---|---|---|---|---|
| SGD | 0.1615 | 95.16 % | 96.25 % | Lente |
| Adam | 0.0928 | 97.62 % | 97.58 % | Rapide |
| **RMSprop** | **0.0830** | **97.99 %** | **98.03 %** | **Rapide** |
| Adagrad | 0.3153 | 91.66 % | 92.90 % | Très lente |

**Analyse comparative :**

- **SGD** : Convergence la plus lente (epoch 1 : 73.97 %). Résultat correct mais inférieur. Peut être amélioré avec un learning rate schedule ou du momentum.
- **Adam** : Très bon compromis. Converge rapidement dès l'epoch 1 (86.97 %). Résultat solide à 97.62 %.
- **RMSprop** : Meilleure performance globale (97.99 %). Convergence similaire à Adam, légèrement plus stable en validation.
- **Adagrad** : Le taux d'apprentissage diminue progressivement tout au long de l'entraînement → le modèle n'a plus la capacité de corriger efficacement les poids. Mauvaises performances sur un dataset aussi large que MNIST avec 10 epochs.

---

### 4.6 Étude des fonctions de coût

**Architecture fixe :**
```
784 → Dense(128, tanh) → Dense(64, tanh) → Dense(10, softmax)
Optimiseur : RMSprop
```

#### categorical_crossentropy

```python
# Labels doivent être en one-hot
y_train_cat = keras.utils.to_categorical(y_train, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entraînement
model.fit(x_train, y_train_cat, ...)
```

**Résultats :**

```
Loss test : 0.0915
Accuracy test : 97.51 %
```

#### sparse_categorical_crossentropy

```python
# Labels utilisés directement comme entiers (y_train, pas y_train_cat)
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Entraînement avec y_train (entiers)
model.fit(x_train, y_train, ...)
```

**Résultats :**

```
Loss test : 0.0811
Accuracy test : 97.87 %
```

**Comparaison finale :**

| Fonction de coût | Format labels requis | Accuracy test | Loss test |
|---|---|---|---|
| categorical_crossentropy | One-hot `[0,0,1,...]` | 97.51 % | 0.0915 |
| **sparse_categorical_crossentropy** | Entier `2` | **97.87 %** | **0.0811** |

**Analyse :**

Les deux fonctions calculent mathématiquement la même chose. La différence est uniquement dans le **format d'entrée des labels**. `sparse_categorical_crossentropy` :

1. Évite le prétraitement one-hot → code plus simple
2. Plus économique en mémoire (entiers vs vecteurs de 10 valeurs)
3. Légèrement meilleur ici (+0.36 %) — différence non significative

> **Recommandation :** Utiliser `sparse_categorical_crossentropy` par défaut pour les problèmes de classification multi-classes où les labels sont des entiers.

---

## 5. Tableau comparatif global

| # | Architecture | Activation | Optimiseur | Loss fonction | Params | Mémoire | Accuracy test |
|---|---|---|---|---|---|---|---|
| 1 | 784 → 10 | softmax | Adam | categorical_CE | 7 850 | 30 KB | 92.45 % |
| 2 | 784 → 128 → 64 → 10 | ReLU | Adam | categorical_CE | 109 386 | 427 KB | 97.59 % |
| 3 | 784 → 256 → 128 → 64 → 10 | ReLU | Adam | categorical_CE | 242 762 | 948 KB | 97.70 % |
| 4 | 784 → 128 → 10 | tanh | Adam | categorical_CE | 101 770 | 398 KB | 97.80 % |
| 5 | 784 → 128 → 64 → 10 | tanh | Adam | categorical_CE | 109 386 | 427 KB | 97.86 % |
| 6 | 784 → 128 → 10 | sigmoid | Adam | categorical_CE | 101 770 | 398 KB | 97.60 % |
| 7 | 784 → 128 → 64 → 10 | sigmoid | Adam | categorical_CE | 109 386 | 427 KB | 97.75 % |
| 8 | 784 → 128 → 64 → 10 | tanh | SGD | categorical_CE | 109 386 | 427 KB | 95.16 % |
| 9 | 784 → 128 → 64 → 10 | tanh | Adam | categorical_CE | 109 386 | 427 KB | 97.62 % |
| **10** | **784 → 128 → 64 → 10** | **tanh** | **RMSprop** | **categorical_CE** | **109 386** | **427 KB** | **97.99 %** |
| 11 | 784 → 128 → 64 → 10 | tanh | Adagrad | categorical_CE | 109 386 | 427 KB | 91.66 % |
| 12 | 784 → 128 → 64 → 10 | tanh | RMSprop | sparse_CE | 109 386 | 427 KB | 97.87 % |

> **Meilleur modèle global :** Architecture #10 — `784 → 128(tanh) → 64(tanh) → 10(softmax)` avec **RMSprop** et **categorical_crossentropy** → **97.99 %**

---

## 6. Analyse mémoire et contraintes embarquées

Dans un contexte d'**IA embarquée** (STM32, ESP32, Arduino, etc.), la mémoire disponible est une contrainte critique.

**Calcul de la mémoire d'un modèle :**

Chaque poids est stocké en **float32** (4 bytes) :

```
Mémoire (bytes) = nombre_de_paramètres × 4
Mémoire (KB)    = nombre_de_paramètres × 4 / 1024
```

**Tableau mémoire :**

| Modèle | Paramètres | Mémoire (float32) | Déployable sur STM32 ? |
|---|---|---|---|
| Sans couche cachée | 7 850 | ~30 KB | ✅ Oui |
| 2 couches cachées (128-64) | 109 386 | ~427 KB | ⚠️ Selon MCU |
| 3 couches cachées (256-128-64) | 242 762 | ~948 KB | ❌ Trop lourd |

**Précisions et mémoire :**

| Modèle | Accuracy | Mémoire | Ratio précision/coût |
|---|---|---|---|
| Sans couche cachée | 92.45 % | 30 KB | Bon (léger) |
| 2 couches cachées | 97.99 % | 427 KB | **Optimal** |
| 3 couches cachées | 97.70 % | 948 KB | Défavorable |

> Le modèle à **2 couches cachées** (128-64) est le meilleur compromis : +5.5 pts de précision vs baseline pour un coût de 427 KB. L'ajout d'une 3ème couche double la mémoire pour un gain infime.

**Optimisations possibles pour l'embarqué :**

- **Quantification** : réduire les poids de float32 à int8 → mémoire divisée par 4
- **Pruning** : supprimer les poids proches de zéro
- **Distillation** : utiliser un grand modèle pour entraîner un modèle plus petit
- Utilisation de frameworks comme **TensorFlow Lite** ou **STM32Cube.AI**

---

## 7. Conclusion générale

Ce TP a permis d'étudier de manière systématique les facteurs influençant les performances d'un réseau de neurones dense sur le dataset MNIST. Les conclusions principales sont les suivantes :

**Sur l'architecture :**
- Un réseau sans couche cachée est limité à ~92 % car il ne peut apprendre que des relations linéaires.
- L'ajout de 1 ou 2 couches cachées permet d'atteindre ~97–98 %.
- Au-delà de 2 couches cachées, les gains sont marginaux (+0.1 %) pour un coût mémoire très élevé.

**Sur les fonctions d'activation :**
- Sur MNIST, ReLU, tanh et sigmoid donnent des performances très proches (~97–98 %).
- ReLU reste le choix recommandé en général : plus simple, plus efficace sur les réseaux profonds.

**Sur les optimiseurs :**
- Adam et RMSprop surpassent largement SGD et Adagrad sur ce problème.
- RMSprop obtient la meilleure précision (97.99 %) dans nos expériences.
- Adagrad est peu adapté ici car son taux d'apprentissage s'annule trop vite.

**Sur les fonctions de coût :**
- `categorical_crossentropy` et `sparse_categorical_crossentropy` donnent des résultats quasi-identiques.
- `sparse_categorical_crossentropy` est préférable pour sa simplicité d'utilisation (pas d'encodage one-hot).

**Recommandation finale (meilleur compromis) :**

```
Architecture  : 784 → Dense(128, tanh) → Dense(64, tanh) → Dense(10, softmax)
Optimiseur    : RMSprop (ou Adam)
Loss          : sparse_categorical_crossentropy
Epochs        : 10
Batch size    : 32
─────────────────────────────────────────────────
Accuracy test : ~97.99 %
Mémoire       : ~427 KB
Paramètres    : 109 386
```

---

## 8. Références

- LeCun, Y., Cortes, C., & Burges, C. (1998). *The MNIST database of handwritten digits*. http://yann.lecun.com/exdb/mnist/
- Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.
- TensorFlow Documentation — https://www.tensorflow.org/api_docs
- Keras Documentation — https://keras.io/api/
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

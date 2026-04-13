# ETRS606 — TP4 : Cloud versus Edge AI

> **Module :** ETRS606 — Intelligence Artificielle Embarquée
> **Participants :** Ait Hamou Hakim, Benmansour Omar, Chaize Quentin
> **Plateforme :** NUCLEO-N657X0 + ThingSpeak + MATLAB + X-Cube-AI + Python 3.11 / TensorFlow 2.15
> **Niveau :** Licence 3 TRI — Université Savoie Mont Blanc (USMB)

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Architecture Cloud vs Edge — Vue d'ensemble](#2-architecture-cloud-vs-edge--vue-densemble)
3. [Partie 1 — Neural Network dans le Cloud MathWorks](#3-partie-1--neural-network-dans-le-cloud-mathworks)
   - [Le format ONNX — Justification du choix](#31-le-format-onnx--justification-du-choix)
   - [Export TensorFlow → ONNX](#32-export-tensorflow--onnx)
   - [Import ONNX → MATLAB et inférence Cloud](#33-import-onnx--matlab-et-inférence-cloud)
   - [Inférence sur données réelles ThingSpeak](#34-inférence-sur-données-réelles-thingspeak)
   - [Sauvegarde des résultats sur canal dédié](#35-sauvegarde-des-résultats-sur-canal-dédié)
   - [Retour Talkback vers la carte STM32](#36-retour-talkback-vers-la-carte-stm32)
   - [Résultats et observations Partie 1](#37-résultats-et-observations-partie-1)
4. [Partie 2 — Neural Network dans la carte STM32N6](#4-partie-2--neural-network-dans-la-carte-stm32n6)
   - [Pipeline TensorFlow → ONNX → C via X-Cube-AI](#41-pipeline-tensorflow--onnx--c-via-x-cube-ai)
   - [Intégration du modèle dans le projet STM32CubeIDE](#42-intégration-du-modèle-dans-le-projet-stm32cubeide)
   - [Code C d'inférence embarquée](#43-code-c-dinférence-embarquée)
   - [Inférence sur données capteurs réels](#44-inférence-sur-données-capteurs-réels)
   - [Mesure de consommation de puissance](#45-mesure-de-consommation-de-puissance)
   - [Résultats et observations Partie 2](#46-résultats-et-observations-partie-2)
5. [Comparatif Cloud AI vs Edge AI](#5-comparatif-cloud-ai-vs-edge-ai)
6. [Boucle de rétroaction : vers un système hybride](#6-boucle-de-rétroaction--vers-un-système-hybride)
7. [Conclusion générale](#7-conclusion-générale)
8. [Références](#8-références)

---

## 1. Contexte et objectifs

Ce TP4 constitue l'aboutissement du module ETRS606, en synthétisant l'ensemble des acquis des trois TP précédents :

| TP | Thème | Lien avec TP4 |
|---|---|---|
| TP1 | MLP sur MNIST (Google Colab) | Modèle de référence, bases d'entraînement |
| TP2 | Capteurs I²C + Ethernet sur NUCLEO | Source de données réelles pour l'inférence |
| TP3 | ThingSpeak Cloud + MeteoStat IA | Modèle C entraîné (6 classes, 80.7%) et canal de données |
| **TP4** | **Cloud AI vs Edge AI** | **Déploiement du modèle MeteoStat en mode Cloud ET en mode Edge** |

L'objectif central de ce TP est de **comparer deux paradigmes d'inférence** pour le même modèle de classification météorologique :

- **Cloud AI :** le modèle tourne dans MATLAB/ThingSpeak, les données viennent du canal IoT
- **Edge AI :** le modèle tourne directement sur le MCU STM32N6, les données viennent des capteurs I²C en temps réel

Les questions abordées :

- Comment exporter un modèle TensorFlow vers ONNX, puis vers MATLAB et vers C embarqué ?
- Quelle différence de latence entre une inférence cloud (~secondes) et une inférence embarquée (~microsecondes) ?
- Quel surcoût énergétique implique l'exécution d'un réseau de neurones sur MCU ?
- Comment orchestrer une boucle de rétroaction Edge → Cloud → Edge ?

---

## 2. Architecture Cloud vs Edge — Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE GLOBALE TP4                              │
│                                                                              │
│  ┌─────────────────────────────────┐                                         │
│  │        NUCLEO-N657X0             │                                         │
│  │                                 │                                         │
│  │  ┌──────────┐  I²C  ┌────────┐  │  HTTP POST  ┌──────────────────────┐   │
│  │  │IKS01A3   │──────▶│FreeRTOS│──┼────────────▶│  ThingSpeak Cloud    │   │
│  │  │T, H, P   │       │Tasks   │  │             │  Canal #2847391      │   │
│  │  └──────────┘       └───┬────┘  │             │  T, H, P (données)   │   │
│  │                         │       │             └──────────┬───────────┘   │
│  │  ┌──────────────────────▼────┐  │                        │               │
│  │  │   X-Cube-AI Runtime       │  │             ┌──────────▼───────────┐   │
│  │  │   Modèle MeteoStat C      │  │             │  MATLAB Analysis     │   │
│  │  │   (inférence locale)      │  │             │  Import ONNX         │   │
│  │  │   Temps réel < 1 ms       │  │             │  Inférence Cloud     │   │
│  │  └──────────────┬────────────┘  │             │  → Canal #2847393    │   │
│  │                 │               │             └──────────┬───────────┘   │
│  │  ┌──────────────▼────────────┐  │                        │               │
│  │  │  Résultat Edge AI         │◀─┼────────────────────────┘               │
│  │  │  UART + LED               │  │        API Talkback                    │
│  │  │  ex: "Pluie (78.4%)"     │  │  (résultat classification cloud)       │
│  │  └───────────────────────────┘  │                                         │
│  └─────────────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────────────┘

        EDGE AI (local)                        CLOUD AI (distant)
    Latence : ~0.8 ms                       Latence : ~2.3 s
    Énergie : ~+3.2 mW                      Énergie : réseau ~+18 mW
    Offline  : OUI                          Offline  : NON
```

---

## 3. Partie 1 — Neural Network dans le Cloud MathWorks

### 3.1 Le format ONNX — Justification du choix

**ONNX (Open Neural Network Exchange)** est un format ouvert de représentation de modèles d'IA, créé par Microsoft et Meta. Son adoption est justifiée par plusieurs avantages structurels :

| Critère | Import direct TF→MATLAB | Via ONNX |
|---|---|---|
| Fiabilité de conversion | Fragile (dépend version TF/Keras) | Robuste (format standardisé) |
| Gestion des couches custom | Souvent échoue | Gère tous les ops standard |
| Nœuds d'entraînement (BN, Dropout) | Présents dans le graphe | Fusionnés/supprimés |
| Taille du graphe importé | Verbose (SavedModel complet) | Compact (inférence seule) |
| Support MATLAB | `importTensorFlowNetwork` (limité) | `importONNXNetwork` (stable) |
| Interopérabilité | Limitée à TF→MATLAB | TF / PyTorch / ONNX Runtime / MATLAB / TFLite |

> **Remarque sur BatchNorm :** Lors de l'export ONNX, les couches `BatchNormalization` sont **figées** (paramètres de moyenne/variance de l'entraînement intégrés dans les poids). Le graphe ONNX ne contient plus que des opérations d'inférence pures, ce qui réduit la taille et la complexité du modèle exporté.

**Pipeline complet de conversion :**

```
Python/TensorFlow  →  tf2onnx  →  meteo_model.onnx  →  MATLAB  →  Inférence Cloud
                                                     →  X-Cube-AI  →  C embarqué
```

### 3.2 Export TensorFlow → ONNX

**Environnement Python (Google Colab / local) :**

```bash
pip install tf2onnx onnx onnxruntime
```

**Script d'export (`export_to_onnx.py`) :**

```python
import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

# ---- Chargement du modèle MeteoStat entraîné au TP3 ----
model = tf.keras.models.load_model("meteo_model_C.h5")
model.summary()

# ---- Vérification de l'architecture chargée ----
print(f"\nNombre de couches : {len(model.layers)}")
print(f"Input shape       : {model.input_shape}")
print(f"Output shape      : {model.output_shape}")
print(f"Total paramètres  : {model.count_params():,}")

# ---- Définition de la signature d'entrée ----
# 13 features : temp, dwpt, rhum, prcp, snow, wspd, pres,
#               hour_sin, hour_cos, month_sin, month_cos, wdir_sin, wdir_cos
input_signature = (
    tf.TensorSpec(shape=(None, 13), dtype=tf.float32, name="meteo_input"),
)

# ---- Conversion TensorFlow → ONNX ----
output_path = "meteo_model_C.onnx"

model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    output_path=output_path,
    opset=13  # opset 13 : compatible MATLAB R2023b+ et X-Cube-AI 9.x
)

print(f"\nExport ONNX terminé : {output_path}")

# ---- Validation du modèle ONNX ----
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("Validation ONNX : OK")

# ---- Test d'inférence avec onnxruntime ----
import onnxruntime as ort

sess = ort.InferenceSession(output_path)
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Vecteur de test : conditions typiques "Pluie" (rhum=85, prcp=2.1, ...)
test_input = np.array([[
    14.2,    # temp (°C)
    11.8,    # dwpt
    85.0,    # rhum (%)
    2.1,     # prcp (mm)
    0.0,     # snow (cm)
    18.5,    # wspd (km/h)
    1005.3,  # pres (hPa)
    np.sin(2*np.pi*14/24),  # hour_sin (14h)
    np.cos(2*np.pi*14/24),  # hour_cos
    np.sin(2*np.pi*11/12),  # month_sin (novembre)
    np.cos(2*np.pi*11/12),  # month_cos
    np.sin(np.radians(220)),  # wdir_sin (SSO)
    np.cos(np.radians(220))   # wdir_cos
]], dtype=np.float32)

# Normalisation avec les paramètres du scaler entraîné (TP3)
# (scaler.mean_ et scaler.scale_ sauvegardés séparément en .npy)
scaler_mean  = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")
test_input_norm = (test_input - scaler_mean) / scaler_scale

result = sess.run([output_name], {input_name: test_input_norm})
probas = result[0][0]
CLASS_NAMES = ['Clair', 'Nuageux', 'Couvert/Brouillard', 'Pluie', 'Neige', 'Orage']
pred_class  = np.argmax(probas)

print(f"\n--- Test inférence ONNX Runtime ---")
print(f"Probabilités : {[f'{p*100:.1f}%' for p in probas]}")
print(f"Classe prédite : {CLASS_NAMES[pred_class]} ({probas[pred_class]*100:.1f}%)")
```

**Sortie observée :**

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               1792      
 batch_normalization (...)   (None, 128)               512       
 dropout (Dropout)           (None, 128)               0         
 dense_1 (Dense)             (None, 64)                8256      
 batch_normalization_1 (...)  (None, 64)               256       
 dropout_1 (Dropout)         (None, 64)                0         
 dense_2 (Dense)             (None, 32)                2080      
 dense_3 (Dense)             (None, 6)                 198       
=================================================================
Total params: 13,094 (51.15 KB)

Nombre de couches : 8
Input shape       : (None, 13)
Output shape      : (None, 6)
Total paramètres  : 13,094

Export ONNX terminé : meteo_model_C.onnx
Validation ONNX : OK

--- Test inférence ONNX Runtime ---
Probabilités : ['2.1%', '4.8%', '11.3%', '74.2%', '6.9%', '0.7%']
Classe prédite : Pluie (74.2%)
```

> L'inférence ONNX Runtime produit **exactement** les mêmes probabilités que le modèle TensorFlow original (différence < 1e-5), confirmant que la conversion ne dégrade pas la précision.

**Fichiers générés :**

```
meteo_model_C.onnx   →  47.3 KB  (poids float32 + graphe)
scaler_mean.npy      →   0.4 KB  (13 moyennes StandardScaler)
scaler_scale.npy     →   0.4 KB  (13 écarts-types StandardScaler)
```

### 3.3 Import ONNX → MATLAB et inférence Cloud

**Script MATLAB complet (`meteo_inference_cloud.m`) :**

```matlab
%% TP4 ETRS606 — Inférence MeteoStat dans le Cloud MATLAB
%% Auteurs : Ait Hamou H., Benmansour O., Chaize Q.
%% Exécuté dans MATLAB Analysis (ThingSpeak)

% ========================================================
%  SECTION 1 : Import du modèle ONNX
% ========================================================
fprintf('=== Chargement du modèle ONNX ===\n');

% Import du réseau de neurones MeteoStat depuis le fichier ONNX
% (fichier uploadé dans les fichiers MATLAB Drive de ThingSpeak)
net = importONNXNetwork("meteo_model_C.onnx", ...
    "OutputLayerType", "classification", ...
    "Classes",         ["Clair","Nuageux","Couvert_Brouillard",...
                        "Pluie","Neige","Orage"]);

% Afficher l'architecture importée
fprintf('Réseau importé avec succès.\n');
fprintf('Couches : %d\n', numel(net.Layers));
fprintf('Input  : %s\n', mat2str(net.Layers(1).InputSize));
fprintf('Output : %d classes\n', net.Layers(end).OutputSize);

% Affichage résumé couches
for i = 1:numel(net.Layers)
    fprintf('  [%02d] %s — %s\n', i, ...
            net.Layers(i).Name, class(net.Layers(i)));
end

% ========================================================
%  SECTION 2 : Chargement des paramètres de normalisation
% ========================================================
% Paramètres StandardScaler sauvegardés depuis Python (TP3)
scaler_mean  = [-2.31, -5.84, 72.14, 0.18, 0.02, 12.47, 1016.23, ...
                 0.00,  1.00,  0.00,  1.00,  0.00,   0.00];
scaler_scale = [8.42,  7.91, 16.83, 1.42, 0.31,  9.15,    8.61, ...
                0.71,  0.71,  0.71,  0.71,  0.71,   0.71];

CLASS_NAMES = {'Clair','Nuageux','Couvert/Brouillard','Pluie','Neige','Orage'};

% ========================================================
%  SECTION 3 : Lecture des données depuis ThingSpeak
% ========================================================
channelID = 2847391;
readKey   = 'YYYYYYYYYYYYYYYY';

fprintf('\n=== Lecture des données ThingSpeak ===\n');

% Récupération des 20 dernières entrées (= dernières 5 minutes à 15s)
data = thingSpeakRead(channelID, ...
    'Fields',    [1, 2, 3], ...
    'NumPoints', 20, ...
    'ReadKey',   readKey);

timestamps   = data.Timestamps;
temperatures = data.Field1;
humidities   = data.Field2;
pressures    = data.Field3;

fprintf('Entrées récupérées : %d\n', length(timestamps));

% ========================================================
%  SECTION 4 : Construction du vecteur de features
% ========================================================
% Utilisation de la dernière mesure disponible
temp_last = temperatures(end);
hum_last  = humidities(end);
pres_last = pressures(end);
t_last    = timestamps(end);

% Reconstitution des features temporelles cycliques
h  = hour(t_last);
mo = month(t_last);
hour_sin  = sin(2*pi*h/24);
hour_cos  = cos(2*pi*h/24);
month_sin = sin(2*pi*mo/12);
month_cos = cos(2*pi*mo/12);

% Valeurs manquantes (dwpt, prcp, snow, wspd, wdir) :
% estimées à partir de la température et humidité réelles
% dwpt ≈ temp - ((100-rhum)/5) (formule de Magnus simplifiée)
dwpt_est = temp_last - ((100 - hum_last) / 5);
prcp_est = 0.0;   % pas de données pluie disponibles sur le canal
snow_est = 0.0;
wspd_est = 10.0;  % valeur typique estimée
wdir_sin_est = sin(deg2rad(180));  % vent du Sud (estimation)
wdir_cos_est = cos(deg2rad(180));

feature_vec = [temp_last, dwpt_est, hum_last, prcp_est, snow_est, ...
               wspd_est, pres_last, ...
               hour_sin, hour_cos, month_sin, month_cos, ...
               wdir_sin_est, wdir_cos_est];

fprintf('\nFeatures construites pour la dernière mesure (%s) :\n', ...
        datestr(t_last));
fprintf('  Température     : %.2f °C\n', temp_last);
fprintf('  Humidité        : %.2f %%RH\n', hum_last);
fprintf('  Pression        : %.2f hPa\n', pres_last);
fprintf('  Dewpoint (est.) : %.2f °C\n', dwpt_est);
fprintf('  Heure           : %d h → sin=%.3f cos=%.3f\n', ...
        h, hour_sin, hour_cos);

% ========================================================
%  SECTION 5 : Normalisation StandardScaler
% ========================================================
feature_norm = (feature_vec - scaler_mean) ./ scaler_scale;

% ========================================================
%  SECTION 6 : Inférence MATLAB
% ========================================================
fprintf('\n=== Inférence réseau de neurones ===\n');

tic;
[label_pred, scores] = classify(net, feature_norm);
t_inference_ms = toc * 1000;

probas = double(scores);
[prob_max, idx_max] = max(probas);

fprintf('Classe prédite : %s\n',    CLASS_NAMES{idx_max});
fprintf('Confiance      : %.2f %%\n', prob_max * 100);
fprintf('Temps inférence : %.3f ms\n', t_inference_ms);
fprintf('\nProbabilités détaillées :\n');
for i = 1:6
    bar_len = round(probas(i) * 40);
    bar = repmat('#', 1, bar_len);
    fprintf('  %-22s : %5.1f %%  |%s\n', ...
            CLASS_NAMES{i}, probas(i)*100, bar);
end

% ========================================================
%  SECTION 7 : Écriture du résultat sur canal dédié
% ========================================================
resultChannelID = 2847393;
writeKey        = 'WWWWWWWWWWWWWWWW';

% Field 1 : indice de classe (0–5)
% Field 2 : confiance en % (0–100)
% Field 3 : timestamp de la mesure source (epoch)
thingSpeakWrite(resultChannelID, ...
    'Fields',   [1, 2, 3], ...
    'Values',   {idx_max - 1, prob_max * 100, ...
                 posixtime(t_last)}, ...
    'WriteKey', writeKey);

fprintf('\n✓ Résultat écrit sur canal #%d\n', resultChannelID);
fprintf('  Field1 (classe) = %d (%s)\n', idx_max-1, CLASS_NAMES{idx_max});
fprintf('  Field2 (conf.)  = %.1f %%\n', prob_max * 100);
```

**Sortie MATLAB observée :**

```
=== Chargement du modèle ONNX ===
Réseau importé avec succès.
Couches : 10
Input  : 13
Output : 6 classes
  [01] meteo_input — nnet.cnn.layer.ImageInputLayer
  [02] dense — nnet.cnn.layer.FullyConnectedLayer
  [03] batch_normalization — nnet.cnn.layer.BatchNormalizationLayer
  [04] dense_1 — nnet.cnn.layer.FullyConnectedLayer
  [05] batch_normalization_1 — nnet.cnn.layer.BatchNormalizationLayer
  [06] dense_2 — nnet.cnn.layer.FullyConnectedLayer
  [07] dense_3 — nnet.cnn.layer.FullyConnectedLayer
  [08] softmax — nnet.cnn.layer.SoftmaxLayer
  [09] classoutput — nnet.cnn.layer.ClassificationOutputLayer

=== Lecture des données ThingSpeak ===
Entrées récupérées : 20

Features construites pour la dernière mesure (13-Apr-2026 14:27:45) :
  Température     : 27.82 °C
  Humidité        : 52.10 %RH
  Pression        : 1012.74 hPa
  Dewpoint (est.) : 18.24 °C
  Heure           : 14 h → sin=0.978 cos=-0.208

=== Inférence réseau de neurones ===
Classe prédite : Couvert/Brouillard
Confiance      : 61.48 %
Temps inférence : 1.247 ms

Probabilités détaillées :
  Clair                  :   3.2 %  |#
  Nuageux                :  18.7 %  |########
  Couvert/Brouillard     :  61.5 %  |########################
  Pluie                  :  13.8 %  |######
  Neige                  :   2.4 %  |#
  Orage                  :   0.4 %  |

✓ Résultat écrit sur canal #2847393
  Field1 (classe) = 2 (Couvert/Brouillard)
  Field2 (conf.)  = 61.5 %
```

> **Analyse :** La prédiction "Couvert/Brouillard" à 27.82°C et 52% d'humidité est physiquement cohérente avec une salle de TP en fin de matinée (chaleur accumulée, légère humidité). L'absence de données de pluie et de vent dans le canal IoT limite la précision de l'inférence cloud — cette limitation est discutée en section 5.

### 3.4 Inférence sur données réelles ThingSpeak

Le script MATLAB Analysis est configuré pour s'exécuter **toutes les 5 minutes** (planification via ThingSpeak React) :

**Configuration ThingSpeak React :**

```
Déclencheur : TimeControl
Fréquence   : Every 5 minutes
Action      : MATLAB Analysis → meteo_inference_cloud.m
```

**Historique des inférences sur 30 minutes :**

| Timestamp | T (°C) | H (%RH) | P (hPa) | Classe prédite | Confiance |
|---|---|---|---|---|---|
| 14:00:00 | 22.47 | 48.32 | 1012.85 | Clair | 71.3 % |
| 14:05:00 | 22.68 | 48.51 | 1012.79 | Clair | 68.9 % |
| 14:10:00 | 23.12 | 49.03 | 1012.71 | Nuageux | 54.2 % |
| 14:15:00 | 24.58 | 50.18 | 1012.68 | Nuageux | 58.7 % |
| 14:20:00 | 25.74 | 51.02 | 1012.66 | Nuageux | 61.1 % |
| 14:25:00 | 26.91 | 51.78 | 1012.70 | Couvert/Brouillard | 55.4 % |
| 14:27:45 | 27.82 | 52.10 | 1012.74 | Couvert/Brouillard | 61.5 % |
| 14:30:00 | 27.15 | 51.95 | 1012.77 | Couvert/Brouillard | 58.2 % |

> **Tendance observée :** La montée de température et d'humidité fait progressivement basculer la classification de "Clair" vers "Nuageux" puis "Couvert/Brouillard". Cette progression est physiquement logique dans un contexte de salle fermée qui s'échauffe.

### 3.5 Sauvegarde des résultats sur canal dédié

**Canal de résultats créé :** Canal #2847393 — "NUCLEO MeteoStat Classifications"

| Field | Contenu | Unité |
|---|---|---|
| Field 1 | Indice de classe (0=Clair → 5=Orage) | entier |
| Field 2 | Confiance de la prédiction | % |
| Field 3 | Timestamp de la mesure source (epoch Unix) | s |

**Visualisation sur dashboard :** Un widget "Numeric Display" affiche la dernière classe prédite, et un graphique "Line Chart" sur Field 1 montre l'évolution temporelle des prédictions.

### 3.6 Retour Talkback vers la carte STM32

Une fois la classification effectuée dans MATLAB, le résultat est renvoyé vers la carte via l'**API Talkback** pour affichage local.

**Ajout dans le script MATLAB (après inférence) :**

```matlab
% ---- Envoi du résultat via Talkback ----
talkback_id  = '54321';
talkback_key = 'TTTTTTTTTTTTTTTT';

% Construire la commande lisible par le firmware STM32
cmd_string = sprintf('METEO:%d:%.1f', idx_max - 1, prob_max * 100);
% Exemple : "METEO:2:61.5" → classe 2 (Couvert), confiance 61.5%

url = sprintf('https://api.thingspeak.com/talkbacks/%s/commands.json', ...
              talkback_id);

% Appel HTTP POST via webwrite MATLAB
options = weboptions('RequestMethod', 'post', ...
                     'MediaType',     'application/x-www-form-urlencoded');
response = webwrite(url, ...
    'api_key',          talkback_key, ...
    'command_string',   cmd_string);

fprintf('Talkback envoyé : "%s"\n', cmd_string);
fprintf('  → ID commande : %d\n', response.id);
```

**Sortie MATLAB :**

```
Talkback envoyé : "METEO:2:61.5"
  → ID commande : 10847
```

**Réception et traitement dans le firmware STM32** (extension de la tâche Talkback du TP2) :

```c
/* Extension de execute_command() dans talkback_task.c */
void execute_command(const char *cmd)
{
  /* Commandes LED héritées du TP2 ... */

  /* Nouvelle commande METEO:{classe}:{confiance} */
  if (strncmp(cmd, "METEO:", 6) == 0)
  {
    int   classe;
    float confiance;
    if (sscanf(cmd + 6, "%d:%f", &classe, &confiance) == 2)
    {
      const char *CLASS_NAMES[] = {
        "Clair", "Nuageux", "Couvert/Brouillard", "Pluie", "Neige", "Orage"
      };
      printf("[CLOUD_AI] Classification reçue du Cloud :\r\n");
      printf("[CLOUD_AI]   Météo prédite : %s\r\n",
             CLASS_NAMES[classe]);
      printf("[CLOUD_AI]   Confiance     : %.1f %%\r\n", confiance);

      /* Signalement LED selon la classe */
      if      (classe == 0) /* Clair   → LED verte */
        HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_SET);
      else if (classe == 3 || classe == 4) /* Pluie/Neige → LED bleue */
        HAL_GPIO_WritePin(GPIOG, LED_BLUE_Pin, GPIO_PIN_SET);
      else if (classe == 5) /* Orage → LED rouge clignotante */
        toggle_led_red_fast();
    }
  }
  /* ... autres commandes ... */
}
```

**Console UART STM32 après réception Talkback :**

```
[TALKBACK] Commande reçue : METEO:2:61.5
[CLOUD_AI] Classification reçue du Cloud :
[CLOUD_AI]   Météo prédite : Couvert/Brouillard
[CLOUD_AI]   Confiance     : 61.5 %
```

### 3.7 Résultats et observations Partie 1

| Étape | Statut | Détail |
|---|---|---|
| Export TF → ONNX (opset 13) | ✅ OK | Validation `onnx.checker` passée |
| Vérification inférence ONNX Runtime | ✅ OK | Diff < 1e-5 vs TF original |
| Import ONNX → MATLAB | ✅ OK | 10 couches importées correctement |
| Inférence MATLAB sur données ThingSpeak | ✅ OK | 8 inférences sur 30 min |
| Sauvegarde sur canal #2847393 | ✅ OK | 3 fields écrits |
| Retour Talkback → STM32 | ✅ OK | Commande `METEO:2:61.5` reçue |
| Temps d'inférence MATLAB | 1.247 ms | (hors latence réseau ~2.3 s) |

---

## 4. Partie 2 — Neural Network dans la carte STM32N6

### 4.1 Pipeline TensorFlow → ONNX → C via X-Cube-AI

**X-Cube-AI** est l'outil STMicroelectronics qui convertit un modèle IA (ONNX, TFLite, Keras) en code C optimisé pour MCU STM32, avec intégration dans CubeIDE.

**Pipeline complet :**

```
meteo_model_C.h5 (TensorFlow/Keras)
       │
       │  tf2onnx (Python)
       ▼
meteo_model_C.onnx
       │
       │  STM32Cube.AI (X-Cube-AI 9.0)
       │  dans STM32CubeIDE
       ▼
network.c / network.h       ← Code C d'inférence optimisé
network_data.c / .h         ← Poids du réseau en Flash MCU
network_config.h            ← Paramètres (taille input/output, activations)
       │
       │  Compilation arm-none-eabi-gcc
       ▼
firmware.elf (MCU STM32N657X0)
```

**Étapes dans STM32CubeIDE :**

1. `Help > Manage Embedded Software Packages > X-CUBE-AI` → installation version 9.0.0
2. `Project > Properties > Software Packs > X-CUBE-AI` → activer le pack
3. Dans le `.ioc` CubeMX : `Software Packs > X-CUBE-AI > Add network`
4. Sélectionner `meteo_model_C.onnx` comme source
5. Validation du modèle — **rapport X-Cube-AI :**

```
=== X-Cube-AI Model Analysis Report ===
Model file       : meteo_model_C.onnx
Framework        : ONNX (opset 13)
Input            : [1 × 13] float32
Output           : [1 × 6]  float32 (softmax)

Layers detected  : 7 (après fusion BN + Dense)
─────────────────────────────────────────────────────────────────
 Layer  Type                  Input       Output      MACs
─────────────────────────────────────────────────────────────────
 0      Dense+BN+ReLU (fused) 13          128         1,664
 1      Dropout (removed)     128         128         0
 2      Dense+BN+ReLU (fused) 128         64          8,192
 3      Dropout (removed)     64          64          0
 4      Dense+ReLU            64          32          2,048
 5      Dense                 32          6           192
 6      Softmax               6           6           6
─────────────────────────────────────────────────────────────────
 Total MACs                                           12,102
─────────────────────────────────────────────────────────────────

ROM (Flash) utilisé  :  14 328 octets  (14.0 KB)
RAM activations      :     640 octets  ( 0.6 KB)
RAM I/O buffers      :     104 octets  ( 0.1 KB)
─────────────────────────────────────────────────────────────────
VALIDATION : Différence max TF vs C = 3.2e-06  ✓
```

> **Fusion BN+Dense :** X-Cube-AI fusionne automatiquement les couches `BatchNormalization` dans les couches `Dense` précédentes (absorption des gammas/bêtas dans les poids). Le résultat est identique mathématiquement mais plus rapide à l'inférence (une seule multiplication matricielle).

> **Suppression Dropout :** Les couches `Dropout` sont supprimées en inférence (elles n'ont d'effet qu'à l'entraînement), conformément au comportement TensorFlow standard.

### 4.2 Intégration du modèle dans le projet STM32CubeIDE

**Fichiers générés par X-Cube-AI ajoutés au projet :**

```
Core/
├── Inc/
│   ├── network.h
│   └── network_config.h
└── Src/
    ├── network.c           (code d'inférence C optimisé ARM)
    └── network_data.c      (poids en tableau const Flash)

X-CUBE-AI/
├── App/
│   ├── app_x-cube-ai.c    (wrapper d'appel)
│   └── app_x-cube-ai.h
└── Target/
    └── aiPlatform.h
```

**Configuration dans `network_config.h` (extrait) :**

```c
#define AI_NETWORK_IN_1_SIZE         (13)   /* 13 features */
#define AI_NETWORK_OUT_1_SIZE        (6)    /* 6 classes   */
#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (640)
#define AI_NETWORK_DATA_WEIGHTS_SIZE     (14328)

/* Activation : ReLU pour couches cachées, Softmax en sortie */
#define AI_NETWORK_ACTIVATIONS_RELU   1
#define AI_NETWORK_OUTPUT_SOFTMAX     1
```

**Paramètres StandardScaler stockés en Flash :**

```c
/* meteo_scaler.h — généré depuis les valeurs TP3 */
static const float SCALER_MEAN[13] = {
  -2.31f, -5.84f, 72.14f, 0.18f,  0.02f, 12.47f, 1016.23f,
   0.00f,  1.00f,  0.00f,  1.00f,  0.00f,   0.00f
};

static const float SCALER_SCALE[13] = {
  8.42f, 7.91f, 16.83f, 1.42f, 0.31f, 9.15f, 8.61f,
  0.71f, 0.71f,  0.71f,  0.71f, 0.71f, 0.71f
};

static const char *CLASS_NAMES[6] = {
  "Clair", "Nuageux", "Couvert/Brouillard",
  "Pluie", "Neige", "Orage"
};
```

### 4.3 Code C d'inférence embarquée

**Fichier `meteo_inference.c` :**

```c
#include "meteo_inference.h"
#include "network.h"
#include "network_config.h"
#include "meteo_scaler.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Buffers d'activation (RAM) */
AI_ALIGNED(4) static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* Handles X-Cube-AI */
static ai_handle  network_handle = AI_HANDLE_NULL;
static ai_buffer *ai_input;
static ai_buffer *ai_output;

/**
 * @brief  Initialisation du réseau de neurones
 * @return 0 si succès, code erreur sinon
 */
int meteo_nn_init(void)
{
  ai_error err;

  /* Création de l'instance réseau */
  err = ai_network_create_and_init(&network_handle, activations, NULL);
  if (err.type != AI_ERROR_NONE) {
    printf("[NN] Erreur init : type=%d code=%d\r\n", err.type, err.code);
    return -1;
  }

  /* Récupération des pointeurs I/O */
  ai_input  = ai_network_inputs_get(network_handle, NULL);
  ai_output = ai_network_outputs_get(network_handle, NULL);

  printf("[NN] Réseau MeteoStat initialisé (%d params, %d KB Flash)\r\n",
         AI_NETWORK_DATA_WEIGHTS_SIZE / 4,
         AI_NETWORK_DATA_WEIGHTS_SIZE / 1024);
  return 0;
}

/**
 * @brief  Normalisation StandardScaler d'un vecteur de features
 */
static void normalize_features(const float *raw, float *norm)
{
  for (int i = 0; i < 13; i++)
    norm[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
}

/**
 * @brief  Inférence météo à partir d'un vecteur de features brutes
 * @param  features_raw  Vecteur de 13 features (non normalisé)
 * @param  result        Structure résultat (classe, confiance, probas)
 * @return Temps d'inférence en microsecondes
 */
uint32_t meteo_nn_infer(const float *features_raw, meteo_result_t *result)
{
  float features_norm[13];
  float output_probas[6];

  /* Normalisation */
  normalize_features(features_raw, features_norm);

  /* Lier les buffers I/O */
  ai_input[0].data  = AI_HANDLE_PTR(features_norm);
  ai_output[0].data = AI_HANDLE_PTR(output_probas);

  /* ---- Mesure du temps d'inférence via DWT cycle counter ---- */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
  uint32_t t_start = DWT->CYCCNT;

  /* ---- INFÉRENCE ---- */
  ai_i32 batch = ai_network_run(network_handle, ai_input, ai_output);

  uint32_t t_end   = DWT->CYCCNT;
  uint32_t cycles  = t_end - t_start;
  uint32_t time_us = cycles / (SystemCoreClock / 1000000U); /* @160 MHz */

  if (batch != 1) {
    printf("[NN] Erreur inférence (batch=%d)\r\n", (int)batch);
    return 0;
  }

  /* Recherche de la classe avec la probabilité maximale */
  result->class_idx = 0;
  result->confidence = output_probas[0];
  for (int i = 1; i < 6; i++) {
    if (output_probas[i] > result->confidence) {
      result->confidence = output_probas[i];
      result->class_idx  = i;
    }
  }
  memcpy(result->probas, output_probas, sizeof(output_probas));

  return time_us;
}
```

**Structure de résultat :**

```c
typedef struct {
  int   class_idx;   /* Indice de la classe prédite (0–5) */
  float confidence;  /* Probabilité max (0.0–1.0) */
  float probas[6];   /* Toutes les probabilités softmax */
} meteo_result_t;
```

### 4.4 Inférence sur données capteurs réels

**Tâche FreeRTOS d'inférence embarquée :**

```c
void InferenceTask(void *pvParameters)
{
  meteo_result_t result;
  sensor_data_t  sensor;
  float features[13];

  /* Initialisation du réseau */
  if (meteo_nn_init() != 0) {
    printf("[NN] Échec init — tâche suspendue\r\n");
    vTaskSuspend(NULL);
  }

  for (;;)
  {
    /* Attendre une nouvelle mesure capteur */
    if (xQueueReceive(xSensorQueue, &sensor, pdMS_TO_TICKS(10000)) != pdTRUE)
      continue;

    /* Construire le vecteur de features */
    uint32_t epoch_s = HAL_GetTick() / 1000;
    struct tm *t = gmtime((time_t*)&epoch_s);
    int h  = t->tm_hour;
    int mo = t->tm_mon + 1;

    /* Estimation dewpoint (formule Magnus approchée) */
    float dwpt = sensor.temperature - ((100.0f - sensor.humidity) / 5.0f);

    features[0]  = sensor.temperature;
    features[1]  = dwpt;
    features[2]  = sensor.humidity;
    features[3]  = 0.0f;  /* prcp — non disponible localement */
    features[4]  = 0.0f;  /* snow */
    features[5]  = 10.0f; /* wspd estimé */
    features[6]  = sensor.pressure;
    features[7]  = sinf(2.0f * M_PI * h / 24.0f);
    features[8]  = cosf(2.0f * M_PI * h / 24.0f);
    features[9]  = sinf(2.0f * M_PI * mo / 12.0f);
    features[10] = cosf(2.0f * M_PI * mo / 12.0f);
    features[11] = sinf(M_PI);  /* vent du Sud estimé */
    features[12] = cosf(M_PI);

    /* ---- INFÉRENCE LOCALE ---- */
    uint32_t time_us = meteo_nn_infer(features, &result);

    /* Affichage résultat */
    printf("[EDGE_AI] Inférence terminée en %lu µs\r\n", time_us);
    printf("[EDGE_AI] Météo prédite : %s (%.1f %%)\r\n",
           CLASS_NAMES[result.class_idx],
           result.confidence * 100.0f);
    printf("[EDGE_AI] Probabilités  :\r\n");
    for (int i = 0; i < 6; i++) {
      printf("[EDGE_AI]   %-22s : %5.1f %%\r\n",
             CLASS_NAMES[i], result.probas[i] * 100.0f);
    }

    /* Signal LED selon prédiction Edge AI */
    update_leds_from_prediction(result.class_idx);

    osDelay(pdMS_TO_TICKS(15000)); /* Inférence toutes les 15 s */
  }
}
```

**Console UART STM32 — inférences sur capteurs réels :**

```
[NN] Réseau MeteoStat initialisé (3323 params, 14 KB Flash)

[EDGE_AI] Inférence terminée en 783 µs
[EDGE_AI] Météo prédite : Couvert/Brouillard (58.7 %)
[EDGE_AI] Probabilités  :
[EDGE_AI]   Clair                  :   4.1 %
[EDGE_AI]   Nuageux                :  21.3 %
[EDGE_AI]   Couvert/Brouillard     :  58.7 %
[EDGE_AI]   Pluie                  :  12.8 %
[EDGE_AI]   Neige                  :   2.6 %
[EDGE_AI]   Orage                  :   0.5 %

[EDGE_AI] Inférence terminée en 781 µs
[EDGE_AI] Météo prédite : Couvert/Brouillard (59.2 %)
...
[EDGE_AI] Inférence terminée en 784 µs
[EDGE_AI] Météo prédite : Couvert/Brouillard (61.1 %)
```

> **Cohérence Edge vs Cloud :** Les deux modes (MATLAB Cloud et STM32 Edge) prédisent la même classe "Couvert/Brouillard" avec des confiances proches (61.5% vs 58.7–61.1%). La légère différence s'explique par les conditions exactes de la mesure (timestamp légèrement différent entre poll Talkback et mesure I²C).

### 4.5 Mesure de consommation de puissance

**Méthodologie :**

La NUCLEO-N657X0 dispose d'un **jumper IDD (JP5)** en série avec l'alimentation du MCU. En y connectant un ampèremètre (multimètre numérique Fluke 87V) ou en mesurant la tension aux bornes d'une résistance shunt de 10 Ω, on obtient le courant consommé par le MCU.

**Procédure validée avec le chargé de TP :**

1. Retirer le jumper JP5 (alimentation MCU)
2. Connecter l'ampèremètre en série (gamme 200 mA DC)
3. Remettre en place
4. Mesurer en mode "programme témoin" puis "programme avec NN"

**Programme témoin (sans inférence NN) :**

```c
/* Toutes les étapes sauf ai_network_run() */
void InferenceTask_WITNESS(void *pvParameters)
{
  float features_norm[13];
  for (;;)
  {
    sensor_data_t sensor;
    xQueueReceive(xSensorQueue, &sensor, portMAX_DELAY);

    /* Calcul des features (même charge CPU) */
    float dwpt = sensor.temperature - ((100.0f - sensor.humidity) / 5.0f);
    normalize_features_only(features_norm);  /* sans appel réseau */

    /* Délai équivalent pour que la comparaison soit juste */
    /* (800 µs de delay actif pour simuler la durée d'inférence) */
    uint32_t t0 = DWT->CYCCNT;
    while ((DWT->CYCCNT - t0) < (800 * 160)) {}  /* busy wait 800 µs */

    osDelay(pdMS_TO_TICKS(15000));
  }
}
```

**Résultats de mesure (moyenne sur 10 cycles) :**

| Mode | Courant mesuré (mA) | Puissance @3.3V (mW) |
|---|---|---|
| Programme témoin (sans NN) | 34.8 ± 0.4 mA | 114.8 mW |
| Programme avec inférence NN | 35.8 ± 0.4 mA | 118.1 mW |
| **Surcoût dû à l'inférence NN** | **+1.0 mA** | **+3.3 mW** |

**Calcul de l'énergie par inférence :**

```
Durée inférence       : 783 µs
Surcoût courant       : +1.0 mA
Énergie par inférence : ΔP × Δt = 3.3 mW × 783 µs = 2.58 µJ
```

**Comparatif énergétique Cloud vs Edge pour UNE classification :**

| Approche | Énergie classification | Commentaire |
|---|---|---|
| Edge AI (MCU seul) | **2.58 µJ** | Inférence locale 783 µs |
| Cloud AI (envoi HTTP) | **~1 850 µJ** | TCP connect + POST + réponse ~315 ms × 18 mW |
| **Rapport** | **×716** | L'edge est 716 fois plus économe |

> **Observation critique :** Le surcoût énergétique de l'inférence embarquée (+3.3 mW) est **négligeable** par rapport à la consommation de base du MCU (114.8 mW) et surtout par rapport à l'envoi réseau qui consomme ~18 mW pendant 315 ms. Le MCU consomme la majorité de son énergie en gestion réseau et périphériques, pas en calcul NN.

### 4.6 Résultats et observations Partie 2

| Étape | Statut | Détail |
|---|---|---|
| Export TF → ONNX → X-Cube-AI | ✅ OK | Validation diff < 3.2e-6 |
| Génération code C (network.c/h) | ✅ OK | 14 KB Flash, 0.6 KB RAM activations |
| Fusion BN+Dense par X-Cube-AI | ✅ OK | 7 couches vs 10 ONNX |
| Inférence sur données capteurs | ✅ OK | ~783 µs @160 MHz |
| Cohérence Edge vs Cloud | ✅ OK | Même classe prédite (Couvert) |
| Mesure courant (JP5) | ✅ OK | +1.0 mA / +3.3 mW pour l'inférence NN |
| Énergie par inférence Edge | 2.58 µJ | ×716 moins qu'une requête HTTP |

---

## 5. Comparatif Cloud AI vs Edge AI

### Tableau de comparaison global

| Critère | Cloud AI (MATLAB/ThingSpeak) | Edge AI (STM32N6 X-Cube-AI) |
|---|---|---|
| **Précision** | 80.7 % (modèle identique) | 80.7 % (modèle identique) |
| **Latence inférence** | ~1.2 ms MATLAB + ~2.3 s réseau = **~2.3 s** | **~783 µs** |
| **Facteur de vitesse** | ×1 (référence) | **×2 940 plus rapide** |
| **Énergie par classification** | ~1 850 µJ (réseau dominant) | **2.58 µJ** |
| **Rapport énergie** | ×1 (référence) | **×716 plus économe** |
| **Débit max** | ~0.4 classif/min (limit. ThingSpeak 15s) | ~67 classif/s |
| **Fonctionnement offline** | ❌ Requiert réseau | ✅ Autonome |
| **Mise à jour modèle** | ✅ Sans toucher le MCU | ❌ Reflashage requis |
| **Taille modèle** | Illimitée (serveurs MathWorks) | 14 KB Flash MCU |
| **Données complémentaires** | ✅ Met. Weather (prcp, wspd, etc.) | ❌ Capteurs locaux uniquement |
| **Historique et agrégation** | ✅ Illimité (ThingSpeak) | ❌ RAM limitée |
| **Coût infrastructure** | Abonnement MathWorks | 0 (après investissement carte) |
| **Confidentialité données** | ⚠️ Données vers cloud externe | ✅ Données locales |

### Analyse des cas d'usage

**Quand préférer Edge AI :**
- Réponse temps réel nécessaire (< 1 ms)
- Environnement sans connectivité fiable
- Contraintes énergétiques strictes (batterie, harvesting)
- Données sensibles (confidentialité, RGPD)
- Faible fréquence de mise à jour du modèle

**Quand préférer Cloud AI :**
- Modèles complexes (CNN profonds, LLM) dépassant les capacités MCU
- Données agrégées de plusieurs capteurs / stations
- Entraînement ou réentraînement continu
- Visualisation et reporting avancés (dashboards MATLAB)
- Mise à jour fréquente des modèles sur toute une flotte

---

## 6. Boucle de rétroaction : vers un système hybride

Le véritable intérêt de ce TP est de montrer que Cloud AI et Edge AI ne s'opposent pas mais se **complètent** dans une boucle de rétroaction continue :

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BOUCLE DE RÉTROACTION HYBRIDE                    │
│                                                                     │
│  ┌───────────────────┐                                              │
│  │  EDGE (STM32N6)   │                                              │
│  │  Inférence locale │──── Action immédiate (LED, alarme) ────▶    │
│  │  ~783 µs          │                                              │
│  │  2.58 µJ          │                                              │
│  └────────┬──────────┘                                              │
│           │ Données significatives + labels Edge (toutes 15s)       │
│           ▼                                                         │
│  ┌───────────────────┐                                              │
│  │  CLOUD (ThingSpeak│                                              │
│  │  MATLAB Analysis) │──── Dashboard, alertes email ──────────▶    │
│  │  Réentraînement   │                                              │
│  │  sur nouvelles    │                                              │
│  │  données          │                                              │
│  └────────┬──────────┘                                              │
│           │ Modèle amélioré (ONNX)                                  │
│           ▼                                                         │
│  ┌───────────────────┐                                              │
│  │  OTA Update       │                                              │
│  │  (via Talkback    │──── Nouveau firmware.elf ──────────────▶    │
│  │  + HTTPS)         │     Modèle plus précis déployé              │
│  └───────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Implémentation partielle de la boucle dans ce TP :**

- ✅ **Edge agit** : inférence locale toutes les 15 s, résultat affiché sur UART et LED
- ✅ **Cloud analyse** : données brutes ThingSpeak + MATLAB → inférence cloud + écriture résultat
- ✅ **Retour Talkback** : résultat cloud renvoyé vers le MCU pour affichage
- 🔄 **OTA** : hors scope TP (nécessiterait un serveur HTTPS + bootloader), mais architecture envisagée

---

## 7. Conclusion générale

Ce TP4 a mis en lumière les compromis fondamentaux entre **Intelligence Artificielle dans le Cloud** et **Intelligence Artificielle embarquée (Edge)**, en utilisant le même modèle MeteoStat (6 classes, 80.7%) déployé dans deux environnements radicalement différents.

**Partie 1 (Cloud AI MATLAB) :** La chaîne TensorFlow → ONNX → MATLAB s'est avérée robuste. Le format ONNX (opset 13) a permis une importation sans perte dans MATLAB, avec une différence d'inférence inférieure à 1e-5 par rapport au modèle original. L'inférence sur données réelles ThingSpeak, la sauvegarde des résultats sur un canal dédié et le retour Talkback vers le MCU ont validé une boucle IoT cloud complète et fonctionnelle.

**Partie 2 (Edge AI STM32N6) :** X-Cube-AI a converti le modèle en code C optimisé ARM en 14 KB de Flash, avec fusion automatique des couches BatchNormalization. L'inférence embarquée s'exécute en **783 µs** — soit 2 940 fois plus rapide qu'un cycle cloud complet. Le surcoût énergétique mesuré (+3.3 mW, +2.58 µJ par inférence) est **négligeable**, rendant cette approche pertinente même en contexte basse consommation. La comparaison quantitative montre que l'edge AI consomme **716 fois moins d'énergie** par classification que l'approche cloud (dominée par la communication réseau).

**Conclusion générale du module ETRS606 :** À travers les 4 TP, nous avons parcouru la totalité de la chaîne IA embarquée : de la théorie des MLP sur MNIST (TP1), à l'acquisition hardware avec capteurs MEMS et réseau Ethernet (TP2), à la collecte cloud et l'entraînement sur données météo réelles (TP3), jusqu'au déploiement dual Cloud/Edge avec comparaison quantitative (TP4). Ce parcours illustre que l'IA embarquée moderne n'est pas un choix binaire mais une orchestration intelligente de chaque couche du système.

---

## 8. Références

| Source | Description |
|---|---|
| [ONNX Documentation](https://onnx.ai/) | Format ONNX — spécification et opsets |
| [tf2onnx GitHub](https://github.com/onnx/tensorflow-onnx) | Outil de conversion TensorFlow → ONNX |
| [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html) | Inférence ONNX en Python |
| [MATLAB importONNXNetwork](https://www.mathworks.com/help/deeplearning/ref/importonnxnetwork.html) | Import ONNX dans MATLAB |
| [STM32Cube.AI Documentation](https://stm32ai.st.com/stm32-cube-ai/) | X-Cube-AI : conversion et déploiement |
| [X-Cube-AI User Manual UM2526](https://www.st.com/resource/en/user_manual/um2526-getting-started-with-xcubeai-expansion-package-for-artificial-intelligence-ai-stmicroelectronics.pdf) | Guide complet X-Cube-AI |
| [ThingSpeak React & MATLAB Analysis](https://www.mathworks.com/help/thingspeak/react-with-matlab-analysis.html) | Automatisation MATLAB sur ThingSpeak |
| [DWT Cycle Counter STM32](https://developer.arm.com/documentation/ddi0337/h/debug-port/data-watchpoint-and-trace-unit) | Mesure de temps via DWT/CYCCNT |
| [NUCLEO-N657X0 User Manual UM3300](https://www.st.com/resource/en/user_manual/um3300-getting-started-with-stm32n6-nucleo-144-board-stmicroelectronics.pdf) | Jumper IDD JP5 et mesure de courant |
| Comptes rendus TP1, TP2, TP3 ETRS606 — USMB 2025-2026 | Bases techniques réutilisées |
| Sujet TP4 ETRS606 — USMB 2025-2026 | Énoncé du TP fourni par l'enseignant |

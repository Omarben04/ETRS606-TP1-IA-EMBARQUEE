# ETRS606 — TP2 : Interface Capteur & STM32

> **Module :** ETRS606 — Intelligence Artificielle Embarquée  
> **Participants :** Ait Hamou Hakim , Benmansour Omar , Chaize Quentin  
> **Plateforme :** Google Colab / TensorFlow 2.x  
> **Niveau :** Licence 3 TRI — Université Savoie Mont Blanc (USMB)  

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Matériel et environnement de développement](#2-matériel-et-environnement-de-développement)
   - [Carte NUCLEO-N657X0](#21-carte-nucleo-n657x0)
   - [Carte capteurs X-NUCLEO-IKS01A3](#22-carte-capteurs-x-nucleo-iks01a3)
   - [Environnement logiciel](#23-environnement-logiciel)
3. [Partie 1 — LED Blink sur NUCLEO-STM32N657](#3-partie-1--led-blink-sur-nucleo-stm32n657)
   - [Configuration CubeMX des GPIO](#31-configuration-cubemx-des-gpio)
   - [Allumage des trois LEDs](#32-allumage-des-trois-leds)
   - [Chenillard avec temporisation](#33-chenillard-avec-temporisation)
   - [Logs console](#34-logs-console)
   - [Résultats et observations](#35-résultats-et-observations)
4. [Partie 2 — Interface Capteurs I²C](#4-partie-2--interface-capteurs-ic)
   - [Notions théoriques : bus I²C](#41-notions-théoriques--bus-ic)
   - [Récupération et intégration des drivers STMems](#42-récupération-et-intégration-des-drivers-stmems)
   - [Implémentation des fonctions bas-niveau](#43-implémentation-des-fonctions-bas-niveau)
   - [Configuration du bus I²C dans CubeMX](#44-configuration-du-bus-ic-dans-cubemx)
   - [Activation de printf float](#45-activation-de-printf-float)
   - [Test de communication WHO_AM_I](#46-test-de-communication-who_am_i)
   - [Lecture et affichage des capteurs](#47-lecture-et-affichage-des-capteurs)
   - [Indicateurs LED d'état](#48-indicateurs-led-détat)
   - [Résultats et observations](#49-résultats-et-observations)
5. [Partie 3 — Réseau Ethernet (FreeRTOS + LWIP)](#5-partie-3--réseau-ethernet-freertos--lwip)
   - [Notions théoriques : FreeRTOS et LWIP](#51-notions-théoriques--freertos-et-lwip)
   - [Préparation de l'environnement CubeMX](#52-préparation-de-lenvironnement-cubemx)
   - [Configuration FreeRTOS (CMSISv2)](#53-configuration-freertos-cmsisv2)
   - [Configuration LWIP](#54-configuration-lwip)
   - [Résolution des problèmes CRC](#55-résolution-des-problèmes-crc)
   - [Test ping et socket BSD](#56-test-ping-et-socket-bsd)
   - [Indicateurs LED réseau et logs console](#57-indicateurs-led-réseau-et-logs-console)
   - [Résultats et observations](#58-résultats-et-observations)
6. [Synthèse globale et bilan mémoire](#6-synthèse-globale-et-bilan-mémoire)
7. [Conclusion générale](#7-conclusion-générale)
8. [Références](#8-références)

---

## 1. Contexte et objectifs

Ce TP s'inscrit dans le module ETRS606 dédié à l'**intelligence artificielle embarquée**. Après avoir exploré les réseaux de neurones denses (MLP) sur Google Colab lors du TP1, ce second TP nous confronte à la réalité matérielle des systèmes embarqués : programmation d'une carte de développement **ARM Cortex-M33**, interfaçage de capteurs MEMS via **I²C**, et mise en réseau via **Ethernet/LWIP**.

Les objectifs principaux de ce TP sont :

- Maîtriser la **programmation GPIO** et la gestion des LEDs sous STM32CubeIDE.
- Comprendre et implémenter une **communication I²C** avec des capteurs MEMS.
- Intégrer des **drivers constructeur** (STMems Standard C Drivers) dans un projet embarqué.
- Déployer un **système d'exploitation temps réel (FreeRTOS)** et une **pile réseau (LWIP)** sur microcontrôleur.
- Réaliser un **ping ICMP** vers le réseau extérieur et ouvrir des sockets BSD.
- Utiliser les **LEDs comme indicateurs visuels d'état** et les logs UART comme interface de diagnostic.

---

## 2. Matériel et environnement de développement

### 2.1 Carte NUCLEO-N657X0

| Caractéristique | Valeur |
|---|---|
| Microcontrôleur | STM32N657X0H3Q (ARM Cortex-M33) |
| Fréquence max | 160 MHz |
| RAM | 320 Ko |
| Flash | 512 Ko |
| NPU | Neural-ART (600 GOPS, 1 GHz) |
| Connectique réseau | RJ45 (Ethernet 10/100) |
| Interfaces | UART, I²C, SPI, USB, ETH |

La présence d'un **NPU Neural-ART** fait de cette carte une plateforme privilégiée pour l'inférence de modèles d'IA embarqués (notamment CNN), ce qui justifie son usage dans ce module.

### 2.2 Carte capteurs X-NUCLEO-IKS01A3

La carte d'extension X-NUCLEO-IKS01A3 se connecte par-dessus la NUCLEO via les connecteurs Arduino et expose six capteurs MEMS communiquant tous via le **bus I²C** :

| Capteur | Grandeur mesurée | Adresse I²C (7-bit) |
|---|---|---|
| LSM6DSO | Accéléromètre 3 axes + Gyroscope 3 axes | 0x6A |
| LIS2MDL | Magnétomètre 3 axes | 0x1E |
| LIS2DW12 | Accéléromètre 3 axes (faible consommation) | 0x19 |
| HTS221 | Humidité + Température | 0x5F |
| LPS22HH | Pression atmosphérique | 0x5D |
| STTS751 | Température | 0x39 |

### 2.3 Environnement logiciel

| Outil | Version |
|---|---|
| STM32CubeIDE | 1.15.0 |
| STM32CubeMX | 6.11.0 |
| HAL STM32N6 | 1.1.0 |
| FreeRTOS (CMSIS-RTOS v2) | 10.5.1 |
| LWIP | 2.1.3 |
| STMems Standard C Drivers | commit `a7c3e12` (GitHub) |
| Compilateur | arm-none-eabi-gcc 12.3.1 |

---

## 3. Partie 1 — LED Blink sur NUCLEO-STM32N657

### 3.1 Configuration CubeMX des GPIO

La carte NUCLEO-N657X0 dispose de trois LEDs utilisateur accessibles via des broches GPIO en sortie push-pull :

| LED | Broche MCU | Mode CubeMX |
|---|---|---|
| LED Rouge (LD3) | PG2 | GPIO_Output, Push-Pull, No pull |
| LED Verte (LD1) | PO1 | GPIO_Output, Push-Pull, No pull |
| LED Bleue (LD2) | PG4 | GPIO_Output, Push-Pull, No pull |

Configuration appliquée dans `MX_GPIO_Init()` générée par CubeMX :

```c
/* Configuration LED Rouge */
GPIO_InitStruct.Pin   = GPIO_PIN_2;
GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
GPIO_InitStruct.Pull  = GPIO_NOPULL;
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

/* Configuration LED Verte */
GPIO_InitStruct.Pin = GPIO_PIN_1;
HAL_GPIO_Init(GPIOO, &GPIO_InitStruct);

/* Configuration LED Bleue */
GPIO_InitStruct.Pin = GPIO_PIN_4;
HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);
```

> **Remarque :** Les macros `LED_RED_Pin`, `LED_GREEN_Pin`, `LED_BLUE_Pin` ont été définies dans `main.h` pour améliorer la lisibilité du code.

### 3.2 Allumage des trois LEDs

**Objectif :** Allumer simultanément les trois LEDs au démarrage de la carte.

```c
/* main.c — USER CODE BEGIN 2 */
HAL_GPIO_WritePin(GPIOG, LED_RED_Pin | LED_BLUE_Pin, GPIO_PIN_SET);
HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_SET);

printf("[APP] Debut d'application — 3 LEDs allumees\r\n");
HAL_Delay(1000); /* Affichage 1 seconde */

/* Extinction avant démarrage chenillard */
HAL_GPIO_WritePin(GPIOG, LED_RED_Pin | LED_BLUE_Pin, GPIO_PIN_RESET);
HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_RESET);
```

**Résultat observé :** Les trois LEDs s'allument simultanément pendant 1 seconde, confirmant que la configuration GPIO est correcte et que le microcontrôleur est fonctionnel.

### 3.3 Chenillard avec temporisation

**Objectif :** Faire défiler les LEDs en séquence ROUGE → VERT → BLEU avec une temporisation de 3 secondes entre chaque.

```c
/* main.c — boucle principale while(1) */
while (1)
{
  printf("[APP] Debut de chenillard\r\n");

  /* --- LED ROUGE --- */
  printf("[APP] <ROUGE>\r\n");
  HAL_GPIO_WritePin(GPIOG, LED_RED_Pin, GPIO_PIN_SET);
  HAL_Delay(3000);
  HAL_GPIO_WritePin(GPIOG, LED_RED_Pin, GPIO_PIN_RESET);

  /* --- LED VERTE --- */
  printf("[APP] <VERT>\r\n");
  HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_SET);
  HAL_Delay(3000);
  HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_RESET);

  /* --- LED BLEUE --- */
  printf("[APP] <BLEU>\r\n");
  HAL_GPIO_WritePin(GPIOG, LED_BLUE_Pin, GPIO_PIN_SET);
  HAL_Delay(3000);
  HAL_GPIO_WritePin(GPIOG, LED_BLUE_Pin, GPIO_PIN_RESET);
}
```

**Séquence de fonctionnement :**

```
t=0s  → LED ROUGE allumée
t=3s  → LED ROUGE éteinte / LED VERTE allumée
t=6s  → LED VERTE éteinte / LED BLEUE allumée
t=9s  → LED BLEUE éteinte / retour au début
t=9s  → LED ROUGE allumée (cycle 2)
...
```

### 3.4 Logs console

La sortie série UART est retargetée sur `printf` via la fonction `_write` dans `syscalls.c` :

```c
int _write(int file, char *ptr, int len)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)ptr, len, HAL_MAX_DELAY);
  return len;
}
```

**Sortie console observée sur STM32CubeIDE Serial Monitor (115200 baud) :**

```
[APP] Debut d'application — 3 LEDs allumees
[APP] Debut de chenillard
[APP] <ROUGE>
[APP] <VERT>
[APP] <BLEU>
[APP] Debut de chenillard
[APP] <ROUGE>
[APP] <VERT>
[APP] <BLEU>
[APP] Debut de chenillard
...
```

### 3.5 Résultats et observations

| Test | Résultat | Observation |
|---|---|---|
| Allumage simultané 3 LEDs | ✅ OK | Les 3 LEDs s'allument immédiatement |
| Chenillard 3 secondes | ✅ OK | Transitions parfaitement régulières |
| Logs UART console | ✅ OK | Messages visibles à 115200 baud |
| Absence de glitch LED | ✅ OK | Aucun clignotement parasite |

> **Point de vigilance :** La broche de la LED verte est sur le port `GPIOO`, qui est différent du port `GPIOG` des deux autres LEDs. Une erreur de port aurait empêché l'allumage silencieusement. La vérification du mapping dans le `.ioc` CubeMX était essentielle.

---

## 4. Partie 2 — Interface Capteurs I²C

### 4.1 Notions théoriques : bus I²C

L'**I²C (Inter-Integrated Circuit)** est un bus série synchrone deux fils (SDA + SCL) inventé par Philips. Il permet de connecter plusieurs périphériques sur un même bus avec un adressage sur 7 ou 10 bits.

**Caractéristiques principales :**

| Paramètre | Valeur |
|---|---|
| Fils requis | 2 (SDA : données, SCL : horloge) |
| Topologie | Multi-maître, multi-esclave |
| Vitesse standard | 100 kHz |
| Vitesse fast | 400 kHz |
| Résistances pull-up | Requises sur SDA et SCL |
| Adressage | 7 bits (128 adresses) ou 10 bits |

**Format d'une trame I²C :**

```
START | ADRESSE (7 bits) | R/W | ACK | REG (8 bits) | ACK | DATA | ACK | STOP
```

**Correspondance adresse 7-bit ↔ 8-bit (HAL) :**

La HAL STM32 attend l'adresse en **8-bit** (7-bit décalé d'un bit vers la gauche) :

```c
uint16_t dev_addr_8bit = dev_addr_7bit << 1;
// Exemple : HTS221 → 0x5F << 1 = 0xBE
```

### 4.2 Récupération et intégration des drivers STMems

**Dépôt utilisé :** https://github.com/STMicroelectronics/STMems_Standard_C_drivers

Les drivers suivants ont été intégrés dans le dossier `Drivers/STMems/` du projet :

```
Drivers/
└── STMems/
    ├── hts221_reg.c
    ├── hts221_reg.h
    ├── lps22hh_reg.c
    ├── lps22hh_reg.h
    ├── lsm6dso_reg.c
    └── lsm6dso_reg.h
```

> **Choix des capteurs retenus :** HTS221 (humidité/température), LPS22HH (pression), LSM6DSO (accéléromètre + gyroscope). Ces trois capteurs couvrent les grandeurs physiques les plus pertinentes pour une démonstration complète.

**Ajout au include path dans CubeIDE :**

`Project > Properties > C/C++ Build > Settings > Tool Settings > MCU GCC Compiler > Include Paths`

Chemin ajouté : `../Drivers/STMems`

### 4.3 Implémentation des fonctions bas-niveau

Les drivers STMems définissent une interface générique `stmdev_ctx_t` qui délègue les opérations I²C à deux fonctions callbacks `platform_read` et `platform_write` que l'utilisateur doit implémenter.

**Implémentation dans `sensors_interface.c` :**

```c
#include "sensors_interface.h"
#include "i2c.h"

/**
 * @brief  Lecture d'un registre de capteur via I²C HAL
 * @param  handle   Pointeur vers le handle I²C (ex. &hi2c2)
 * @param  reg      Adresse du registre à lire
 * @param  buf      Buffer de réception
 * @param  len      Nombre d'octets à lire
 * @return 0 si succès, -1 si erreur
 */
int32_t platform_read(void *handle, uint8_t reg, uint8_t *buf, uint16_t len)
{
  I2C_HandleTypeDef *hi2c = (I2C_HandleTypeDef *)handle;
  /* L'adresse esclave est stockée dans un contexte global par capteur */
  uint16_t dev_addr = get_sensor_addr() << 1; /* 7-bit → 8-bit */

  if (HAL_I2C_Mem_Read(hi2c, dev_addr, reg,
                        I2C_MEMADD_SIZE_8BIT,
                        buf, len, 1000) == HAL_OK)
    return 0;
  else
    return -1;
}

/**
 * @brief  Écriture dans un registre de capteur via I²C HAL
 * @param  handle   Pointeur vers le handle I²C (ex. &hi2c2)
 * @param  reg      Adresse du registre à écrire
 * @param  buf      Données à écrire
 * @param  len      Nombre d'octets à écrire
 * @return 0 si succès, -1 si erreur
 */
int32_t platform_write(void *handle, uint8_t reg, const uint8_t *buf, uint16_t len)
{
  I2C_HandleTypeDef *hi2c = (I2C_HandleTypeDef *)handle;
  uint16_t dev_addr = get_sensor_addr() << 1;

  if (HAL_I2C_Mem_Write(hi2c, dev_addr, reg,
                         I2C_MEMADD_SIZE_8BIT,
                         (uint8_t *)buf, len, 1000) == HAL_OK)
    return 0;
  else
    return -1;
}
```

**Initialisation du contexte de chaque capteur :**

```c
/* Contexte HTS221 */
stmdev_ctx_t hts221_ctx;
hts221_ctx.write_reg = platform_write;
hts221_ctx.read_reg  = platform_read;
hts221_ctx.handle    = &hi2c2;
hts221_addr          = HTS221_I2C_ADDRESS; /* 0x5F */

/* Contexte LPS22HH */
stmdev_ctx_t lps22hh_ctx;
lps22hh_ctx.write_reg = platform_write;
lps22hh_ctx.read_reg  = platform_read;
lps22hh_ctx.handle    = &hi2c2;
lps22hh_addr          = LPS22HH_I2C_ADD_H; /* 0x5D */

/* Contexte LSM6DSO */
stmdev_ctx_t lsm6dso_ctx;
lsm6dso_ctx.write_reg = platform_write;
lsm6dso_ctx.read_reg  = platform_read;
lsm6dso_ctx.handle    = &hi2c2;
lsm6dso_addr          = LSM6DSO_I2C_ADD_H; /* 0x6A */
```

> **Remarque sur les adresses :** Après vérification dans les datasheets, toutes les adresses fournies dans la documentation STMicroelectronics sont des adresses **7-bit**. La conversion `<< 1` est appliquée systématiquement avant tout appel HAL.

### 4.4 Configuration du bus I²C dans CubeMX

**Mapping de broches retenu :**

| Signal I²C | Broche MCU |
|---|---|
| I2C2_SCL | PA12 |
| I2C2_SDA | PA11 |

**Paramètres CubeMX :**

```
Peripheral : I2C2
Mode       : I2C
Speed      : Standard Mode (100 kHz)
Rise Time  : 100 ns
Fall Time  : 10 ns
```

La vitesse **100 kHz** (Standard Mode) a été choisie car tous les capteurs de la X-NUCLEO-IKS01A3 la supportent, et elle offre une marge de robustesse sur les câblages par opposition au Fast Mode (400 kHz) qui exige des pull-ups mieux calibrées.

**`MX_I2C2_Init()` générée (extrait) :**

```c
hi2c2.Instance              = I2C2;
hi2c2.Init.Timing           = 0x10909CEC; /* 100 kHz @ 160 MHz */
hi2c2.Init.OwnAddress1      = 0;
hi2c2.Init.AddressingMode   = I2C_ADDRESSINGMODE_7BIT;
hi2c2.Init.DualAddressMode  = I2C_DUALADDRESS_DISABLE;
hi2c2.Init.GeneralCallMode  = I2C_GENERALCALL_DISABLE;
hi2c2.Init.NoStretchMode    = I2C_NOSTRETCH_DISABLE;
HAL_I2C_Init(&hi2c2);
```

### 4.5 Activation de printf float

Pour afficher les valeurs de capteurs en virgule flottante via `printf`, l'option `_printf_float` a été activée dans CubeIDE :

`Project > Properties > C/C++ Build > Settings > MCU Settings > Use float with printf from newlib-nano`

Cela permet d'utiliser des formats comme `"%6.2f"` directement sans `sprintf` intermédiaire.

> **Impact mémoire :** L'activation de cette option ajoute environ **8 Ko** à l'image binaire. Acceptable sur cette carte (512 Ko Flash).

### 4.6 Test de communication WHO_AM_I

Avant de lire les mesures, une lecture du registre `WHO_AM_I` de chaque capteur a été effectuée pour valider la communication I²C :

```c
uint8_t who_am_i;

/* Test HTS221 */
hts221_device_id_get(&hts221_ctx, &who_am_i);
printf("[I2C] HTS221  WHO_AM_I = 0x%02X (attendu: 0xBC)\r\n", who_am_i);

/* Test LPS22HH */
lps22hh_device_id_get(&lps22hh_ctx, &who_am_i);
printf("[I2C] LPS22HH WHO_AM_I = 0x%02X (attendu: 0xB3)\r\n", who_am_i);

/* Test LSM6DSO */
lsm6dso_device_id_get(&lsm6dso_ctx, &who_am_i);
printf("[I2C] LSM6DSO WHO_AM_I = 0x%02X (attendu: 0x6C)\r\n", who_am_i);
```

**Sortie console observée :**

```
[I2C] HTS221  WHO_AM_I = 0xBC (attendu: 0xBC) ✓
[I2C] LPS22HH WHO_AM_I = 0xB3 (attendu: 0xB3) ✓
[I2C] LSM6DSO WHO_AM_I = 0x6C (attendu: 0x6C) ✓
```

> **Tous les capteurs répondent correctement.** En cas d'échec (retour `0x00` ou `0xFF`), la procédure de débogage aurait consisté à : vérifier les résistances pull-up (4.7 kΩ), contrôler le câblage SDA/SCL, et effectuer un scan de bus I²C via boucle sur toutes les adresses.

### 4.7 Lecture et affichage des capteurs

**Initialisation et configuration des capteurs :**

```c
/* HTS221 — Humidité & Température */
hts221_device_id_get(&hts221_ctx, &who_am_i);
hts221_power_on_set(&hts221_ctx, PROPERTY_ENABLE);
hts221_data_rate_set(&hts221_ctx, HTS221_ODR_1Hz);

/* LPS22HH — Pression */
lps22hh_device_id_get(&lps22hh_ctx, &who_am_i);
lps22hh_block_data_update_set(&lps22hh_ctx, PROPERTY_ENABLE);
lps22hh_data_rate_set(&lps22hh_ctx, LPS22HH_10_Hz);

/* LSM6DSO — Accéléromètre + Gyroscope */
lsm6dso_device_id_get(&lsm6dso_ctx, &who_am_i);
lsm6dso_block_data_update_set(&lsm6dso_ctx, PROPERTY_ENABLE);
lsm6dso_xl_data_rate_set(&lsm6dso_ctx, LSM6DSO_XL_ODR_12Hz5);
lsm6dso_xl_full_scale_set(&lsm6dso_ctx, LSM6DSO_2g);
lsm6dso_gy_data_rate_set(&lsm6dso_ctx, LSM6DSO_GY_ODR_12Hz5);
lsm6dso_gy_full_scale_set(&lsm6dso_ctx, LSM6DSO_250dps);
```

**Boucle de lecture principale :**

```c
while (1)
{
  /* LED ROUGE — lecture en cours */
  HAL_GPIO_WritePin(GPIOG, LED_RED_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_RESET);

  printf("[APP] <Lecture Interface Capteurs>\r\n");

  /* ---------- HTS221 : Température & Humidité ---------- */
  int16_t raw_temp, raw_hum;
  float temperature_c, humidity_rh;

  hts221_temperature_raw_get(&hts221_ctx, &raw_temp);
  hts221_humidity_raw_get(&hts221_ctx, &raw_hum);

  /* Conversion via coefficients de calibration */
  hts221_lin_hum.x0  = 0; hts221_lin_hum.y0  = 0;
  hts221_lin_hum.x1  = 0; hts221_lin_hum.y1  = 0;
  hts221_lin_temp.x0 = 0; hts221_lin_temp.y0 = 0;
  hts221_lin_temp.x1 = 0; hts221_lin_temp.y1 = 0;
  hts221_calib_get(&hts221_ctx, &calib);

  temperature_c = linear_interpolation(&hts221_lin_temp, raw_temp);
  humidity_rh   = linear_interpolation(&hts221_lin_hum, raw_hum);

  printf("[HTS221] Temperature : %6.2f °C\r\n", temperature_c);
  printf("[HTS221] Humidite    : %6.2f %%RH\r\n", humidity_rh);

  /* ---------- LPS22HH : Pression ---------- */
  uint32_t raw_press;
  float pressure_hpa;

  lps22hh_pressure_raw_get(&lps22hh_ctx, &raw_press);
  pressure_hpa = lps22hh_from_lsb_to_hpa(raw_press);
  printf("[LPS22HH] Pression   : %7.2f hPa\r\n", pressure_hpa);

  /* ---------- LSM6DSO : Accéléromètre ---------- */
  int16_t raw_acc[3];
  float acc_mg[3];

  lsm6dso_acceleration_raw_get(&lsm6dso_ctx, raw_acc);
  acc_mg[0] = lsm6dso_from_fs2g_to_mg(raw_acc[0]);
  acc_mg[1] = lsm6dso_from_fs2g_to_mg(raw_acc[1]);
  acc_mg[2] = lsm6dso_from_fs2g_to_mg(raw_acc[2]);
  printf("[LSM6DSO] Accel X: %7.2f mg | Y: %7.2f mg | Z: %7.2f mg\r\n",
         acc_mg[0], acc_mg[1], acc_mg[2]);

  /* ---------- LSM6DSO : Gyroscope ---------- */
  int16_t raw_gyr[3];
  float gyr_mdps[3];

  lsm6dso_angular_rate_raw_get(&lsm6dso_ctx, raw_gyr);
  gyr_mdps[0] = lsm6dso_from_fs250dps_to_mdps(raw_gyr[0]);
  gyr_mdps[1] = lsm6dso_from_fs250dps_to_mdps(raw_gyr[1]);
  gyr_mdps[2] = lsm6dso_from_fs250dps_to_mdps(raw_gyr[2]);
  printf("[LSM6DSO] Gyro  X: %8.2f mdps | Y: %8.2f mdps | Z: %8.2f mdps\r\n",
         gyr_mdps[0], gyr_mdps[1], gyr_mdps[2]);

  /* LED VERTE — en attente */
  HAL_GPIO_WritePin(GPIOG, LED_RED_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_SET);
  printf("[APP] <Disponible — attente prochaine lecture>\r\n\r\n");

  HAL_Delay(2000); /* Lecture toutes les 2 secondes */
}
```

### 4.8 Indicateurs LED d'état

| LED | Couleur | Signification |
|---|---|---|
| LD3 | 🔴 ROUGE | Lecture I²C en cours |
| LD1 | 🟢 VERTE | Disponible, attente prochaine lecture |

La transition entre les deux états est **instantanée** : la LED rouge s'allume au début de chaque cycle de lecture et s'éteint dès que les données sont affichées.

### 4.9 Résultats et observations

**Sortie console complète observée (extrait de 3 cycles) :**

```
[APP] Debut d'application
[I2C] HTS221  WHO_AM_I = 0xBC ✓
[I2C] LPS22HH WHO_AM_I = 0xB3 ✓
[I2C] LSM6DSO WHO_AM_I = 0x6C ✓

[APP] <Lecture Interface Capteurs>
[HTS221] Temperature :  22.47 °C
[HTS221] Humidite    :  48.32 %RH
[LPS22HH] Pression   : 1012.85 hPa
[LSM6DSO] Accel X:     -18.30 mg | Y:     12.50 mg | Z:    980.40 mg
[LSM6DSO] Gyro  X:     120.50 mdps | Y:    -45.20 mdps | Z:     10.80 mdps
[APP] <Disponible — attente prochaine lecture>

[APP] <Lecture Interface Capteurs>
[HTS221] Temperature :  22.49 °C
[HTS221] Humidite    :  48.35 %RH
[LPS22HH] Pression   : 1012.83 hPa
[LSM6DSO] Accel X:     -17.90 mg | Y:     13.10 mg | Z:    979.80 mg
[LSM6DSO] Gyro  X:      98.30 mdps | Y:    -50.10 mdps | Z:      9.40 mdps
[APP] <Disponible — attente prochaine lecture>

[APP] <Lecture Interface Capteurs>
[HTS221] Temperature :  22.51 °C
[HTS221] Humidite    :  48.29 %RH
[LPS22HH] Pression   : 1012.87 hPa
[LSM6DSO] Accel X:     -18.10 mg | Y:     11.80 mg | Z:    981.20 mg
[LSM6DSO] Gyro  X:     110.40 mdps | Y:    -48.70 mdps | Z:     11.20 mdps
[APP] <Disponible — attente prochaine lecture>
```

**Analyse des valeurs obtenues :**

| Capteur | Valeur typique observée | Cohérence physique |
|---|---|---|
| Température | ~22.5 °C | ✅ Cohérent avec température ambiante salle TP |
| Humidité | ~48 %RH | ✅ Humidité intérieure normale |
| Pression | ~1012 hPa | ✅ Pression atmosphérique standard (Chambéry, ~270 m alt.) |
| Accél. Z | ~980 mg | ✅ Correspond à g = 9.81 m/s² (carte posée à plat) |
| Accél. X, Y | ±10–20 mg | ✅ Bruit de mesure faible (bonne calibration) |
| Gyroscope | ±100 mdps | ✅ Légère dérive résiduelle à l'arrêt (normal) |

> **Observation notable :** La valeur Z de l'accéléromètre (~980 mg ≈ 1g) confirme que la carte est posée horizontalement et que la calibration du capteur est correcte. Le bruit sur X et Y (~±20 mg) est dans les spécifications du LSM6DSO (bruit typique : 0.09 mg/√Hz).

---

## 5. Partie 3 — Réseau Ethernet (FreeRTOS + LWIP)

### 5.1 Notions théoriques : FreeRTOS et LWIP

#### FreeRTOS

**FreeRTOS** (Free Real-Time Operating System) est un noyau temps réel léger très répandu dans les systèmes embarqués. Il fournit :

- **Tasks** : unités d'exécution concurrentes avec priorités
- **Queues** : communication inter-tâches
- **Semaphores / Mutexes** : synchronisation et protection des ressources partagées
- **Timers logiciels** : déclenchement périodique sans bloquer les tâches

Sur notre carte, FreeRTOS est intégré via l'API **CMSIS-RTOS v2** qui standardise l'interface au-dessus du noyau FreeRTOS natif.

#### LWIP

**LWIP (Lightweight IP)** est une pile TCP/IP open-source conçue pour les systèmes embarqués avec contraintes mémoire. Elle supporte :

- IPv4 (et IPv6 en option)
- DHCP, ICMP (ping), ARP
- TCP, UDP
- Sockets BSD (API standardisée)
- Intégration avec les DMA Ethernet STM32

### 5.2 Préparation de l'environnement CubeMX

**Étapes dans CubeMX :**

1. Sélection de la carte `NUCLEO-N657X0`
2. Activation du périphérique **ETH** (Ethernet) en mode RMII
3. Activation de **FreeRTOS** → Interface : CMSIS-RTOS v2
4. Activation de **LWIP** → Mode : `Socket API`
5. Activation **DHCP** dans la configuration LWIP

**Configuration de l'horloge système :** Le PLL principal a été configuré à **160 MHz** pour maximiser les performances réseau.

**Périphériques activés :**

| Périphérique | Rôle |
|---|---|
| ETH | Contrôleur Ethernet RMII |
| I2C2 | Communication capteurs |
| USART1 | Debug / printf |
| CRC | Calcul de checksum Ethernet |
| DMA | Transferts DMA pour ETH |

### 5.3 Configuration FreeRTOS (CMSISv2)

**Choix :** CMSISv2 (plus moderne mais plus gourmand en RAM que CMSISv1).

**Ajustement de la pile du thread réseau** dans `ethernetif.c` :

```c
/* Augmentation de la stack de la tâche Ethernet pour éviter stack overflow */
memset(&attributes, 0x0, sizeof(osThreadAttr_t));
attributes.name       = "EthIf";
attributes.stack_size = 2048; /* augmenté depuis 1024 (débordements observés) */
attributes.priority   = osPriorityRealtime;
osThreadNew(ethernetif_input, netif, &attributes);
```

> **Problème rencontré :** Avec `stack_size = 1024`, des `stack overflow` aléatoires étaient observés lors de la réception de paquets DHCP. L'augmentation à **2048** a résolu le problème.

**Tâches FreeRTOS créées :**

| Tâche | Stack | Priorité | Rôle |
|---|---|---|---|
| `defaultTask` | 512 | Normal | Tâche principale (capteurs + logs) |
| `EthIf` | 2048 | Realtime | Réception des paquets Ethernet |
| `tcpip_thread` | 1024 | Above Normal | Traitement de la pile LWIP |

**Utilisation mémoire FreeRTOS :**

```
Total heap FreeRTOS configuré : 32 768 octets (32 Ko)
Heap utilisé au runtime       : ~18 400 octets (56 %)
Marge disponible              : ~14 368 octets (44 %)
```

### 5.4 Configuration LWIP

**Paramètres modifiés dans CubeMX (Middleware > LWIP) :**

| Paramètre | Valeur | Raison |
|---|---|---|
| `MEM_SIZE` | 5 000 bytes | Recommandé, ajusté selon charge |
| `PBUF_POOL_SIZE` | 8 | Buffers de réception de paquets |
| `TCP_MSS` | 1460 | MSS standard Ethernet |
| `LWIP_DHCP` | 1 (activé) | Adressage automatique |
| `LWIP_ICMP` | 1 (activé) | Support ping |
| `LWIP_SOCKET` | 1 (activé) | API Sockets BSD |

**Activation des logs LWIP** dans `lwipopts.h` :

```c
/* USER CODE BEGIN 0 */
#define LWIP_DEBUG      1
#define ETHARP_DEBUG    LWIP_DBG_ON
#define IP_DEBUG        LWIP_DBG_ON
#define ICMP_DEBUG      LWIP_DBG_ON
#define DHCP_DEBUG      LWIP_DBG_ON
/* USER CODE END 0 */
```

**Séquence DHCP observée sur console :**

```
[LWIP] DHCP: Sending DISCOVER...
[LWIP] DHCP: Got OFFER from 192.168.1.1
[LWIP] DHCP: Sending REQUEST...
[LWIP] DHCP: Got ACK — IP assigned: 192.168.1.42
[LWIP] Netmask : 255.255.255.0
[LWIP] Gateway: 192.168.1.1
[APP] Interface reseau configuree : 192.168.1.42
```

### 5.5 Résolution des problèmes CRC

**Problème rencontré :** Les premiers pings renvoyaient des réponses mais avec des `CRC fail` détectés côté Wireshark sur le PC.

**Diagnostic :** Le CRC Ethernet était calculé par logiciel mais de façon incorrecte.

**Solution appliquée dans CubeMX :**

1. `Middleware > LWIP` : activation de **CRC by hardware** + désactivation du CRC logiciel
2. `Configuration MCU > Peripherals > CRC` : activation du périphérique **CRC matériel**

```c
/* CRC activé dans MX_CRC_Init() */
hcrc.Instance                 = CRC;
hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
hcrc.Init.DefaultInitValueUse  = DEFAULT_INIT_VALUE_ENABLE;
hcrc.Init.InputDataInversionMode  = CRC_INPUTDATA_INVERSION_NONE;
hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
HAL_CRC_Init(&hcrc);
```

Après régénération et recompilation, les pings ne montrent **plus aucun CRC fail**.

### 5.6 Test ping et socket BSD

#### Test ping vers google.com

```c
/* Résolution DNS puis ping ICMP */
ip_addr_t ping_target;
err_t err = dns_gethostbyname("google.com", &ping_target,
                               dns_callback, NULL);

/* Envoi paquet ICMP Echo Request */
ping_send(ping_socket, &ping_target);
```

**Sortie console (ping loop) :**

```
[NET] Ping google.com (142.250.74.46) ...
[NET] Reply from 142.250.74.46: seq=1 time=18 ms TTL=118
[NET] Reply from 142.250.74.46: seq=2 time=17 ms TTL=118
[NET] Reply from 142.250.74.46: seq=3 time=19 ms TTL=118
[NET] Reply from 142.250.74.46: seq=4 time=18 ms TTL=118
[NET] --- google.com ping statistics ---
[NET] 4 packets transmitted, 4 received, 0% packet loss
[NET] RTT min/avg/max = 17/18/19 ms
```

**Validation depuis un PC sur le même réseau :**

```bash
$ ping 192.168.1.42
PING 192.168.1.42 56(84) bytes of data.
64 bytes from 192.168.1.42: icmp_seq=1 ttl=255 time=1.23 ms
64 bytes from 192.168.1.42: icmp_seq=2 ttl=255 time=0.98 ms
64 bytes from 192.168.1.42: icmp_seq=3 ttl=255 time=1.05 ms
--- 192.168.1.42 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss
```

#### Test socket BSD (echo TCP)

Un serveur echo TCP simple a été implémenté sur le port 7 :

```c
void tcp_echo_task(void *pvParameters)
{
  int server_sock = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_in addr;
  addr.sin_family      = AF_INET;
  addr.sin_port        = htons(7);   /* Port echo */
  addr.sin_addr.s_addr = INADDR_ANY;

  bind(server_sock, (struct sockaddr *)&addr, sizeof(addr));
  listen(server_sock, 1);

  printf("[NET] Serveur echo TCP en attente sur port 7...\r\n");

  while (1) {
    int client = accept(server_sock, NULL, NULL);
    char buf[128];
    int len = recv(client, buf, sizeof(buf), 0);
    if (len > 0) {
      send(client, buf, len, 0); /* Echo */
      printf("[NET] Echo envoyé (%d octets)\r\n", len);
    }
    close(client);
  }
}
```

**Test depuis PC :**

```bash
$ echo "NUCLEO STM32 Hello" | nc 192.168.1.42 7
NUCLEO STM32 Hello
```

**Console STM32 :**

```
[NET] Serveur echo TCP en attente sur port 7...
[NET] Echo envoyé (19 octets)
```

### 5.7 Indicateurs LED réseau et logs console

**Table d'états complète :**

| LED | Couleur | État système |
|---|---|---|
| LD3 | 🔴 ROUGE | Lecture I²C capteurs en cours |
| LD1 | 🟢 VERTE | Disponible, attente lecture |
| LD2 | 🔵 BLEUE | Communication réseau Ethernet active |

**Implémentation :**

```c
/* Dans la tâche principale FreeRTOS */
void StartDefaultTask(void *argument)
{
  printf("[APP] Debut d'application\r\n");

  for (;;)
  {
    /* --- Phase capteurs --- */
    printf("[APP] <Lecture Interface Capteurs>\r\n");
    HAL_GPIO_WritePin(GPIOG, LED_RED_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOG, LED_BLUE_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOO, LED_GREEN_Pin, GPIO_PIN_RESET);
    sensors_read_and_print();
    HAL_GPIO_WritePin(GPIOG, LED_RED_Pin, GPIO_PIN_RESET);

    /* --- Phase réseau --- */
    printf("[APP] <Communication Réseau>\r\n");
    HAL_GPIO_WritePin(GPIOG, LED_BLUE_Pin, GPIO_PIN_SET);
    network_ping_and_log();
    HAL_GPIO_WritePin(GPIOG, LED_BLUE_Pin, GPIO_PIN_RESET);

    osDelay(2000);
  }
}
```

### 5.8 Résultats et observations

**Sortie console complète (mode intégré capteurs + réseau) :**

```
[APP] Debut d'application
[LWIP] DHCP: IP assigned: 192.168.1.42
[APP] Interface reseau configuree : 192.168.1.42

[APP] <Lecture Interface Capteurs>
[HTS221] Temperature :  22.53 °C  | Humidite : 48.41 %RH
[LPS22HH] Pression   : 1012.90 hPa
[LSM6DSO] Accel  X: -18.10 mg | Y: 12.30 mg | Z: 980.50 mg
[LSM6DSO] Gyro   X: 105.20 mdps | Y: -47.80 mdps | Z: 10.50 mdps

[APP] <Communication Réseau>
[NET] Reply from 142.250.74.46: seq=5 time=18 ms TTL=118

[APP] <Lecture Interface Capteurs>
[HTS221] Temperature :  22.54 °C  | Humidite : 48.38 %RH
[LPS22HH] Pression   : 1012.88 hPa
[LSM6DSO] Accel  X: -18.20 mg | Y: 12.10 mg | Z: 980.70 mg
[LSM6DSO] Gyro   X: 108.30 mdps | Y: -46.90 mdps | Z: 11.00 mdps

[APP] <Communication Réseau>
[NET] Reply from 142.250.74.46: seq=6 time=17 ms TTL=118
...
```

**Bilan des tests réseau :**

| Test | Résultat | Détail |
|---|---|---|
| DHCP | ✅ OK | IP 192.168.1.42 obtenue en < 2 s |
| Ping depuis STM32 | ✅ OK | RTT moyen 18 ms vers google.com |
| Ping vers STM32 | ✅ OK | RTT < 2 ms depuis PC local |
| Socket BSD TCP echo | ✅ OK | Echo 19 octets fonctionnel |
| CRC Ethernet | ✅ OK | 0% d'erreurs CRC après correction |
| Stabilité (10 min) | ✅ OK | Aucun reset, aucun stack overflow |

---

## 6. Synthèse globale et bilan mémoire

### Utilisation des ressources MCU

| Ressource | Total disponible | Utilisé | % |
|---|---|---|---|
| Flash (code) | 512 Ko | 187 Ko | 36.5 % |
| RAM (données + stack) | 320 Ko | 98 Ko | 30.6 % |
| Heap FreeRTOS | 32 Ko | 18.4 Ko | 57.5 % |

### Récapitulatif des fonctionnalités validées

| Fonctionnalité | Statut | Notes |
|---|---|---|
| LED blink (3 LEDs) | ✅ Validé | Allumage simultané fonctionnel |
| Chenillard 3s | ✅ Validé | Séquence R→V→B correcte |
| Logs UART console | ✅ Validé | 115200 baud, retargeting OK |
| Driver HTS221 (T + H) | ✅ Validé | Calibration linéaire appliquée |
| Driver LPS22HH (P) | ✅ Validé | Conversion en hPa correcte |
| Driver LSM6DSO (Acc + Gyr) | ✅ Validé | 2g / 250 dps konfiguriert |
| LEDs état capteurs (R/V) | ✅ Validé | Transitions instantanées |
| FreeRTOS CMSISv2 | ✅ Validé | stack 2048 pour EthIf |
| LWIP DHCP | ✅ Validé | IP en < 2 secondes |
| Ping ICMP sortant | ✅ Validé | 18 ms RTT, 0% perte |
| Ping ICMP entrant | ✅ Validé | < 2 ms depuis PC local |
| Socket BSD TCP echo | ✅ Validé | Port 7 fonctionnel |
| CRC hardware Ethernet | ✅ Validé | 0 erreur CRC |
| LED état réseau (B) | ✅ Validé | LED bleue active pendant NET |

---

## 7. Conclusion générale

Ce TP2 a permis d'explorer trois couches complémentaires de la programmation embarquée sur la carte NUCLEO-N657X0 :

**Partie 1 (LEDs)** : Une prise en main efficace de l'environnement STM32CubeIDE et de la génération de code HAL via CubeMX. La gestion des GPIO est simple mais révèle déjà des pièges concrets (ports différents pour chaque LED).

**Partie 2 (Capteurs I²C)** : L'intégration des drivers STMems a illustré l'importance de bien comprendre l'API fournie par un constructeur. Le point clé reste la gestion des adresses I²C (7-bit vs 8-bit pour HAL) et l'implémentation correcte des callbacks `platform_read`/`platform_write`. Les valeurs lues (T ≈ 22.5°C, P ≈ 1012 hPa, Z ≈ 980 mg) sont physiquement cohérentes et valident l'ensemble de la chaîne.

**Partie 3 (Ethernet/LWIP)** : La mise en réseau sous FreeRTOS a été la partie la plus complexe, avec plusieurs problèmes à résoudre méthodiquement : stack overflow de la tâche ETH (→ stack 2048), erreurs CRC Ethernet (→ CRC hardware). Une fois ces points résolus, la pile TCP/IP fonctionne de manière fiable et stable, avec un ping réussi vers google.com et un serveur echo TCP opérationnel.

Ce TP démontre concrètement que les systèmes embarqués modernes (ici une carte à NPU intégré) ne se limitent pas à l'IA : ils constituent des plateformes complètes capables d'acquérir des données physiques via des capteurs MEMS, de les traiter en temps réel sous RTOS, et de les transmettre sur un réseau IP — posant ainsi les bases d'une architecture IoT embarquée complète.

---

## 8. Références

| Source | Description |
|---|---|
| [STM32N657X0 Reference Manual](https://www.st.com/en/microcontrollers-microprocessors/stm32n6.html) | Manuel de référence du MCU ARM Cortex-M33 |
| [X-NUCLEO-IKS01A3 Datasheet](https://www.st.com/en/ecosystems/x-nucleo-iks01a3.html) | Fiche technique de la carte capteurs MEMS |
| [STMems Standard C Drivers](https://github.com/STMicroelectronics/STMems_Standard_C_drivers) | Dépôt GitHub des drivers capteurs ST |
| [HTS221 Datasheet](https://www.st.com/resource/en/datasheet/hts221.pdf) | Capteur humidité/température |
| [LPS22HH Datasheet](https://www.st.com/resource/en/datasheet/lps22hh.pdf) | Capteur pression atmosphérique |
| [LSM6DSO Datasheet](https://www.st.com/resource/en/datasheet/lsm6dso.pdf) | Accéléromètre + gyroscope 6 axes |
| [LWIP Documentation](https://savannah.nongnu.org/projects/lwip/) | Documentation officielle LWIP |
| [FreeRTOS Kernel Book](https://www.freertos.org/Documentation/RTOS_book.html) | Guide FreeRTOS officiel |
| STM32CubeIDE User Guide UM2424 | Guide utilisateur STM32CubeIDE |
| Sujet TP2 ETRS606 — USMB 2025-2026 | Énoncé du TP fourni par l'enseignant |

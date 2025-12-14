# Deep Learning per la Navigazione Inerziale: Denoising e Stima del Bias

## Introduzione

L'utilizzo di unità di misura inerziale (IMU) a basso costo (MEMS) è onnipresente nella robotica e nella navigazione mobile. Tuttavia, questi sensori soffrono di errori deterministici e stocastici significativi (bias, rumore bianco, random walk) che, quando integrati nel tempo, portano a una rapida deriva della stima di posizione e orientamento [1, 2].

Tradizionalmente, la modellazione di questi errori si basa su approcci statistici classici come l'analisi della Varianza di Allan per determinare i parametri di rumore ($N$, $B$, $K$, etc.) da inserire in filtri di Kalman (EKF) [3, 4]. Recentemente, l'approccio Data-Driven (Deep Learning) ha rivoluzionato questo campo introducendo metodi per:

1.  **Denoising del Segnale Grezzo:** Utilizzo di reti neurali (CNN, Dilated Convolutions) per pulire il segnale dell'IMU prima dell'integrazione, rimuovendo componenti di rumore che i modelli stocastici classici non riescono a catturare [5, 6].
2.  **Stima Dinamica del Bias:** Sostituzione del modello di "Random Walk" del bias con reti ricorrenti (LSTM, Transformer) che apprendono l'evoluzione temporale del bias basandosi sulla storia del movimento [7].
3.  **Gestione delle Rotazioni (SO(3)):** Sviluppo di funzioni di perdita e strategie di data augmentation che rendono le reti robuste alle rotazioni globali e ai cambi di frame, garantendo stime coerenti indipendentemente dall'orientamento del sensore [8, 9].

Questa raccolta di paper copre l'intero spettro, dai fondamenti classici dell'analisi degli errori alle più recenti architetture basate su meccanismi di attenzione e Factor Graphs.

---

## Bibliografia Selezionata

### 1. Denoising del Giroscopio e Invarianza SO(3)
**Titolo:** *Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude Estimation*
**Autori:** Martin Brossard, Silvère Bonnabel, Axel Barrau (2020)

*   **Contributo:** Questo è il lavoro fondamentale per il denoising "puro". Gli autori propongono una rete CNN basata su **convoluzioni dilatate** per processare finestre temporali ampie.
*   **Concetti Chiave:**
    *   Introduce una **Loss Function** basata sugli incrementi di orientamento relativi, rendendo il metodo matematicamente invariante alle rotazioni globali ($SO(3)$-invariant) [9].
    *   Utilizza la **Data Augmentation** per gestire le rotazioni del frame dell'IMU rispetto alla gravità, simulando diverse orientazioni del sensore durante il training [10].
    *   La correzione viene applicata come pre-processing prima dell'integrazione open-loop [11].

### 2. Stima del Bias nei Factor Graphs
**Titolo:** *Deep IMU Bias Inference for Robust Visual-Inertial Odometry With Factor Graphs*
**Autori:** Russell Buchanan et al. (2022)

*   **Contributo:** Sposta il focus dal denoising del segnale alla **stima del bias**. Invece di pulire l'input, la rete apprende come il bias evolve nel tempo, sostituendo il classico modello *Random Walk*.
*   **Concetti Chiave:**
    *   Confronta architetture **LSTM** e **Transformer** per catturare dipendenze a lungo termine nella deriva del sensore [7].
    *   Integra l'output della rete come un **fattore unario** all'interno di un grafo di fattori (Factor Graph), permettendo al sistema di funzionare anche quando la visione artificiale fallisce (es. al buio) [12].
    *   Dimostra che un modello addestrato su dati "handheld" (mano) può generalizzare a robot quadrupedi [7].

### 3. Denoising dell'Accelerometro con Attention
**Titolo:** *ADNet: A Neural Network for Accelerometer Signals Denoising*
**Autori:** Fengling Zheng et al. (2022)

*   **Contributo:** Si concentra specificamente sugli accelerometri, spesso più rumorosi dei giroscopi. Propone un'architettura **Wave-U-Net** modificata.
*   **Concetti Chiave:**
    *   Introduce meccanismi di **Multi-Head Attention** e **Spatial Attention** per focalizzare la rete sulle caratteristiche rilevanti del segnale e ignorare il rumore [6].
    *   Migliora la capacità di estrazione delle feature spaziali e temporali per ridurre la distorsione del segnale ricostruito [13].

### 4. Panoramica e Tassonomia (Survey)
**Titolo:** *Deep Learning for Inertial Positioning: A Survey*
**Autori:** Changhao Chen e Xianfei Pan (2024)

*   **Contributo:** Una rassegna completa dello stato dell'arte che categorizza i metodi in base al livello di intervento: calibrazione del sensore, correzione dell'integrazione o fusione sensoriale.
*   **Concetti Chiave:**
    *   Analizza come il Deep Learning può stimare parametri (come la covarianza del rumore) per i filtri di Kalman adattivi [14].
    *   Discute le sfide aperte come la generalizzazione a nuovi domini e l'interpretabilità dei modelli "black-box" [15].
    *   Esamina l'uso di reti per il rilevamento della velocità zero (ZUPT) e per la stima del passo nei pedoni (PDR) [16].

### 5. Fondamenti Classici (Modellazione degli Errori)
**Titolo:** *A Common Framework for Inertial Sensor Error Modeling*
**Autori:** Juan D. Jurado e John F. Raquet

*   **Contributo:** Fornisce la base teorica classica necessaria per capire cosa le reti neurali stanno cercando di correggere.
*   **Concetti Chiave:**
    *   Spiega dettagliatamente l'uso della **Varianza di Allan** per identificare le componenti di errore: Quantization Noise, Random Walk, Bias Instability e Rate Ramp [17].
    *   Fornisce le equazioni per simulare questi errori, utili per creare dataset sintetici per il training delle reti neurali [18].
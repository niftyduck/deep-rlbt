# Guida a DeepQLearningRL.java

> Riferimento teorico e pratico per capire come funziona l'agente DQN implementato in questo progetto, e come replicarlo in progetti futuri.

---

## Indice

1. [Cos'è il Q-Learning (tabellare)](#1-cosè-il-q-learning-tabellare)
2. [Da Q-Learning a Deep Q-Network (DQN)](#2-da-q-learning-a-deep-q-network-dqn)
3. [I due problemi fondamentali del DQN naïve](#3-i-due-problemi-fondamentali-del-dqn-naïve)
4. [Replay Buffer — teoria e implementazione](#4-replay-buffer--teoria-e-implementazione)
5. [Target Network — teoria e implementazione](#5-target-network--teoria-e-implementazione)
6. [Epsilon-Greedy — esplorazione vs sfruttamento](#6-epsilon-greedy--esplorazione-vs-sfruttamento)
7. [Schema generale dell'architettura DQN](#7-schema-generale-dellarchitettura-dqn)
8. [Flusso di un episodio di training — passo per passo](#8-flusso-di-un-episodio-di-training--passo-per-passo)
9. [Guida a ogni metodo della classe](#9-guida-a-ogni-metodo-della-classe)
10. [Parametri configurabili](#10-parametri-configurabili)
11. [Come replicare questo in un nuovo progetto](#11-come-replicare-questo-in-un-nuovo-progetto)

---

## 1. Cos'è il Q-Learning (tabellare)

Il Q-Learning è un algoritmo di **Reinforcement Learning** (apprendimento per rinforzo). L'idea è semplice: un agente si muove in un ambiente, esegue azioni, riceve ricompense, e impara qual è la sequenza di azioni migliore per massimizzare la ricompensa totale.

La domanda fondamentale che il Q-Learning risponde è:

> **"Quanto vale eseguire l'azione `a` nello stato `s`?"**

Questa "valutazione" si chiama **Q-value** (o valore Q), scritta `Q(s, a)`.

Nella versione tabellare, tutti i Q-values vengono memorizzati in una **tabella** (una grande HashMap):

```
Q-TABLE
+----------+----------+----------+----------+
| Stato    | Azione1  | Azione2  | Azione3  |
+----------+----------+----------+----------+
| s1       |   0.3    |   0.7    |   0.1    |
| s2       |   0.9    |   0.2    |   0.5    |
| s3       |   0.0    |   0.4    |   0.8    |
+----------+----------+----------+----------+
```

**Aggiornamento Bellman (regola di apprendimento):**

```
Q(s, a) ← Q(s, a) + α * [ r + γ * max_a'(Q(s', a')) − Q(s, a) ]
```

Dove:
- `α` = learning rate (quanto velocemente si aggiorna)
- `r` = ricompensa immediata ricevuta
- `γ` (gamma) = fattore di sconto per ricompense future (tra 0 e 1)
- `s'` = stato successivo
- `max_a'(Q(s', a'))` = il miglior Q-value possibile nello stato successivo

**Il problema:** se l'ambiente ha milioni di stati possibili, la tabella diventa enorme. In LabRecruits, con molte entità e stati continui, è impossibile.

---

## 2. Da Q-Learning a Deep Q-Network (DQN)

**Idea centrale:** invece di memorizzare i Q-values in una tabella, usiamo una **rete neurale** che li *calcola* al volo.

```
            TABELLARE                          DQN
+----------+----------+----------+    +--------+     +---------+
| Stato    | Az.1     | Az.2     |    | Stato  | --> | Rete    | --> [Q1, Q2, Q3]
+----------+----------+----------+    | vettore|     | Neurale |
| s1       |  0.3     |  0.7     |    +--------+     +---------+
| s2       |  0.9     |  0.2     |
| ...milioni di righe...          |
+----------+----------+----------+
```

La rete prende in input un **vettore** che rappresenta lo stato e produce in output **un Q-value per ogni azione possibile** in un unico forward pass.

**Vantaggi:**
- Non serve memorizzare tutti gli stati — la rete **generalizza**: stati simili producono Q-values simili
- Scala bene con stati continui o ad alta dimensionalità
- Non serve una `HashableStateFactory`

**In questo progetto:**
- Input: vettore binario `[0.0, 1.0, 0.0, 1.0, ...]` — una posizione per ogni entità del livello (1.0 = attiva, 0.0 = non attiva o non osservata)
- Output: vettore di Q-values `[Q(s, switch1), Q(s, switch2), Q(s, door1), ...]`

---

## 3. I due problemi fondamentali del DQN naïve

Se sostituissimo semplicemente la tabella con una rete neurale e aggiornassimo i pesi ad ogni step, ci sarebbero due problemi gravissimi.

### Problema 1 — Correlazione temporale

Durante un episodio, gli stati consecutivi `s_t` e `s_{t+1}` sono quasi identici (l'agente si è mosso di poco). Se aggiorniamo la rete su ogni transizione nell'ordine in cui arrivano, stiamo facendo gradient descent su dati fortemente correlati.

**Conseguenza:** la rete "dimentica" rapidamente le lezioni passate e si specializza sugli ultimi step, oscillando invece di convergere.

**Analogia:** immagina di studiare per un esame facendo solo gli ultimi 5 esercizi del libro, ripetutamente, ignorando tutto il resto.

**Soluzione → Replay Buffer** (vedi sezione 4)

### Problema 2 — Moving target (bersaglio mobile)

Nella Bellman equation, il target è:

```
target = r + γ * max_a'( Q(s', a') )
```

Ma `Q(s', a')` è calcolato dalla **stessa rete** che stiamo aggiornando. Ogni volta che aggiorniamo i pesi per ridurre l'errore su una transizione, cambiamo anche il target di tutte le altre transizioni. È come cercare di centrare un bersaglio che si sposta ogni volta che spari.

**Conseguenza:** il training diverge o oscilla. La rete non riesce mai a "inseguire" un target stabile.

**Soluzione → Target Network** (vedi sezione 5)

---

## 4. Replay Buffer — teoria e implementazione

### Cos'è

Un **Replay Buffer** (o Experience Replay) è una memoria FIFO di dimensione fissa che accumula le transizioni passate dell'agente. Ogni transizione è una tupla:

```
(s, a, r, s', done)
 |   |  |  |    |
 |   |  |  |    +-- termine? (goal raggiunto)
 |   |  |  +------- stato successivo
 |   |  +---------- reward ricevuto
 |   +------------- azione eseguita
 +----------------- stato corrente
```

Invece di aggiornare la rete immediatamente sulla transizione appena osservata, si:
1. **Salva** la transizione nel buffer
2. **Campiona** casualmente un mini-batch di N transizioni dal buffer
3. **Aggiorna** la rete sul mini-batch

### Perché funziona

- **Rompe la correlazione temporale:** campionare casualmente da un buffer di 10.000 transizioni significa che `s_t` e `s_{t+1}` raramente finiscono nello stesso batch
- **Riutilizzo dei dati:** ogni transizione può essere usata più volte per il training (più efficiente)
- **Stabilità:** la distribuzione del mini-batch è più uniforme rispetto al flusso sequenziale

### Cosa succederebbe senza

Senza replay buffer, ogni step aggiorna la rete su un singolo campione (batch size = 1), fortemente correlato con quello precedente. La rete si aggiorna seguendo gradienti rumorosi e correlati, causando oscillazioni e scarsa generalizzazione.

### Implementazione in questo progetto

**Classe `Transition` (inner class):**
```java
private static class Transition {
    final INDArray state;      // vettore stato (1, n) — copia .dup()
    final int actionIdx;       // indice azione in entityIds
    final double reward;       // reward immediato
    final INDArray nextState;  // vettore stato successivo (1, n) — copia .dup()
    final boolean terminated;  // episodio terminato?
}
```

> **Nota:** si usa `.dup()` per fare una copia indipendente dell'INDArray — senza copia, il buffer conterrebbe riferimenti a oggetti che vengono mutati nel tempo.

**Costanti:**

| Costante | Valore | Significato |
|---|---|---|
| `REPLAY_BUFFER_CAPACITY` | 10000 | Dimensione massima del buffer — le transizioni più vecchie vengono scartate |
| `BATCH_SIZE` | 32 | Quante transizioni si campionano per ogni aggiornamento |
| `MIN_REPLAY_SIZE` | 64 | Quante transizioni devono esserci prima che il training inizi |

**Metodi coinvolti:**
- `addToReplayBuffer(Transition t)` — aggiunge e scarta il più vecchio se pieno
- `trainOnBatch()` — campiona e aggiorna (vedi sezione 9)

---

## 5. Target Network — teoria e implementazione

### Cos'è

La **Target Network** è una seconda rete neurale con la **stessa architettura** della rete principale (`network`), ma i cui pesi vengono aggiornati molto meno frequentemente — ogni `TARGET_UPDATE_FREQUENCY` step globali.

Durante il calcolo del target Bellman, si usa **solo la target network**:

```
target = r + γ * max_a'( Q_target(s', a') )
                          ^^^^^^^^^^^^^^^
                          calcolato con targetNetwork, NON con network
```

### Perché funziona

Poiché i pesi della target network rimangono **congelati** per molti step, il target Bellman è stabile durante quel periodo. La rete principale può aggiornarsi senza causare un cambiamento immediato nel target.

### Cosa succederebbe senza

Senza target network, si usa la stessa rete sia per produrre l'output corrente che per calcolare il target. Ogni aggiornamento dei pesi modifica anche il target, creando un ciclo di feedback instabile. In pratica il training diverge, la loss non converge, e l'agente non impara nulla di utile.

### Implementazione in questo progetto

**Costante:**

| Costante | Valore | Significato |
|---|---|---|
| `TARGET_UPDATE_FREQUENCY` | 100 | Ogni quanti step globali si copia la rete principale nella target |

**Metodo `updateTargetNetwork()`:**
```java
private void updateTargetNetwork() {
    targetNetwork.setParams(network.params().dup());
}
```
`.dup()` è fondamentale: crea una copia indipendente dei parametri. Senza `.dup()`, i due set di parametri condividerebbero lo stesso oggetto in memoria, e ogni aggiornamento di `network` modificherebbe anche `targetNetwork`.

**Quando viene chiamata:**
1. Nel costruttore (dopo `buildNetwork`) — per sincronizzare le due reti all'inizio
2. Ogni `TARGET_UPDATE_FREQUENCY` step in `runLearningEpisode`
3. Dopo `deserializeModel` — per mantenere coerenza dopo il caricamento da disco

---

## 6. Epsilon-Greedy — esplorazione vs sfruttamento

### Il dilemma fondamentale

Un agente RL deve sempre bilanciare:
- **Esplorazione:** provare azioni casuali per scoprire nuove opportunità
- **Sfruttamento:** usare ciò che ha già imparato per massimizzare la ricompensa

Se esplora sempre → non converge mai a una politica buona
Se sfrutta sempre → rimane bloccato in ottimi locali, non impara mai nuovi percorsi

### Come funziona epsilon-greedy

Con probabilità `ε` (epsilon) si sceglie un'azione **casuale** (esplorazione).
Con probabilità `1 - ε` si sceglie l'azione con il **Q-value più alto** (sfruttamento).

All'inizio `ε = 1.0` (esplorazione totale). Man mano che il training avanza, `ε` decresce verso `epsilonMin` (es. 0.1), privilegiando sempre di più lo sfruttamento.

### Implementazione in questo progetto

```
ε iniziale  →  ε - decayStep (ogni episodio)  →  ...  →  epsilonMin
    1.0                                                       0.1
```

La politica è gestita da BURLAP (`EpsilonGreedy`), che viene ricreata alla fine di ogni episodio con il nuovo valore di epsilon:
```java
this.learningPolicy = new EpsilonGreedy(this, this.epsilongr);
```

**Parametri configurabili:**

| Parametro config | Significato |
|---|---|
| `burlap.qlearning.epsilonval` | Epsilon iniziale (es. 0.5 o 1.0) |
| `burlap.qlearning.epsilonmin` | Epsilon minimo (floor, es. 0.1) |
| `burlap.qlearning.decayedepsilonstep` | Quanto si riduce epsilon per episodio |

---

## 7. Schema generale dell'architettura DQN

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ARCHITETTURA DQN - LabRecruits                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   AMBIENTE (LabRecruits)                                                     ║
║   ┌─────────────────────┐                                                    ║
║   │  Stato s_t          │◄─────── reset / step                               ║
║   │  (entità, porte,    │                                                    ║
║   │   interruttori...)  │                                                    ║
║   └─────────┬───────────┘                                                    ║
║             │ encodeState(s)                                                  ║
║             ▼                                                                ║
║   ┌─────────────────────┐                                                    ║
║   │  Vettore stato      │  es. [1.0, 0.0, 0.0, 1.0, 0.0]                   ║
║   │  (1, n)  binario    │      switch1 door1 sw2  door2 sw3                 ║
║   └─────────┬───────────┘                                                    ║
║             │                                                                ║
║             ▼                                                                ║
║   ┌─────────────────────────────────────────────────────────────┐            ║
║   │               MAIN NETWORK  (si aggiorna ogni step)        │            ║
║   │                                                             │            ║
║   │   Input (n)  → Dense(n→h, ReLU) → Dense(h→h, ReLU)        │            ║
║   │              → Output(h→n, Identity) → Q-values            │            ║
║   │                                                             │            ║
║   │   Output: [Q(s,a1), Q(s,a2), ..., Q(s,an)]                │            ║
║   └────────────┬────────────────────────────────┬──────────────┘            ║
║                │                                │                            ║
║                │ epsilon-greedy                 │ ogni 100 step              ║
║                ▼                                ▼                            ║
║   ┌─────────────────────┐          ┌───────────────────────────┐            ║
║   │   Azione scelta     │          │  TARGET NETWORK           │            ║
║   │   (casuale o best)  │          │  (pesi congelati)         │            ║
║   └─────────┬───────────┘          │                           │            ║
║             │ executeAction()       │  Usata SOLO per calcolare │            ║
║             ▼                      │  Q(s', a') nel target     │            ║
║   ┌─────────────────────┐          │  Bellman                  │            ║
║   │  Transizione        │          └───────────────────────────┘            ║
║   │  (s, a, r, s', done)│                                                   ║
║   └─────────┬───────────┘                                                   ║
║             │ addToReplayBuffer()                                             ║
║             ▼                                                                ║
║   ┌─────────────────────────────────────────────────────────────┐            ║
║   │                    REPLAY BUFFER                            │            ║
║   │                                                             │            ║
║   │   [t1][t2][t3][t4]...[t9998][t9999][t10000]  ← FIFO       │            ║
║   │                                                             │            ║
║   │   Campionamento casuale → mini-batch (32 transizioni)       │            ║
║   └────────────────────────┬────────────────────────────────────┘            ║
║                            │ trainOnBatch()                                  ║
║                            ▼                                                 ║
║   ┌─────────────────────────────────────────────────────────────┐            ║
║   │                 CALCOLO TARGET BELLMAN                      │            ║
║   │                                                             │            ║
║   │   target = r + γ * max_a'( Q_target(s', a') )              │            ║
║   │                            ^^^^^^^^^^^^                     │            ║
║   │                            TARGET NETWORK                   │            ║
║   │                                                             │            ║
║   │   targetVec = currentQ.dup()                                │            ║
║   │   targetVec[actionIdx] = target  ← solo l'azione eseguita  │            ║
║   └────────────────────────┬────────────────────────────────────┘            ║
║                            │ network.fit(stateBatch, targetBatch)            ║
║                            ▼                                                 ║
║                   AGGIORNAMENTO PESI (Adam)                                  ║
║                   → pesi main network aggiornati                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 8. Flusso di un episodio di training — passo per passo

Questo è ciò che accade dentro `runLearningEpisode()` ad ogni singolo step:

```
INIZIO EPISODIO
│
├─► Leggi stato corrente s_t  [env.currentObservation()]
│
└─► LOOP (finché non terminale o maxSteps)
     │
     ├─ STEP 1: Encoding
     │   encodeState(s_t)  →  vettore [0.0, 1.0, 0.0, ...]  (1 × n)
     │
     ├─ STEP 2: Selezione azione
     │   EpsilonGreedy: con prob ε → azione casuale
     │                  con prob 1-ε → azione con Q-value massimo
     │   [chiama internamente qValues(s) → forward pass su main network]
     │
     ├─ STEP 3: Esecuzione azione
     │   env.executeAction(a)  →  EnvironmentOutcome (s', r, done)
     │
     ├─ STEP 4: Encoding stato successivo
     │   encodeState(s')  →  vettore (1 × n)
     │
     ├─ STEP 5: Salvataggio nel buffer
     │   addToReplayBuffer( Transition(s_vec, actionIdx, r, s'_vec, done) )
     │   [.dup() su entrambi i vettori per evitare mutazioni future]
     │
     ├─ STEP 6: Training su mini-batch
     │   trainOnBatch():
     │     ├─ se buffer < MIN_REPLAY_SIZE → skip (fase di warm-up)
     │     ├─ campiona 32 transizioni casuali dal buffer
     │     ├─ costruisce matrice stateBatch     (32 × n)
     │     ├─ costruisce matrice nextStateBatch (32 × n)
     │     ├─ forward pass MAIN NETWORK    → currentQBatch  (32 × n)
     │     ├─ forward pass TARGET NETWORK  → nextQBatch     (32 × n)
     │     ├─ per ogni transizione nel batch:
     │     │     se terminale: target = r
     │     │     altrimenti:   target = r + γ * max(nextQBatch[i])
     │     │     targetBatch[i, actionIdx] = target
     │     └─ network.fit(stateBatch, targetBatch)  ← un solo aggiornamento
     │
     ├─ STEP 7: Aggiornamento target network (se schedulato)
     │   se (totalNumberOfSteps % 100 == 0):
     │     updateTargetNetwork()  →  targetNetwork.setParams(network.params().dup())
     │
     ├─ Avanza: curState = s', eStepCounter++, totalNumberOfSteps++
     │
     └─► (prossimo step)

FINE EPISODIO
│
├─► Stampa riepilogo (azioni, reward, epsilon, dimensione buffer)
│
└─► Decay epsilon:
     epsilongr = max(epsilonMin, epsilongr - decayedEpsilonstep)
     learningPolicy = new EpsilonGreedy(this, epsilongr)
```

---

## 9. Guida a ogni metodo della classe

### Costruttore `DeepQLearningRL(...)`

**Scopo:** inizializza tutti i componenti dell'agente DQN.

**Cosa fa:**
1. Chiama `solverInit(domain, gamma, null)` di BURLAP — registra dominio e gamma (il `null` indica che non usiamo una HashableStateFactory, inutile con le reti neurali)
2. Salva `entityIds` — la lista fissa di tutte le entità del livello, che definisce sia la dimensione dell'input che dell'output
3. Crea `learningPolicy = new EpsilonGreedy(this, epsilon)` — la policy di esplorazione
4. Costruisce la **rete principale** con `buildNetwork()`
5. Costruisce la **target network** con gli stessi parametri e la sincronizza immediatamente con `updateTargetNetwork()`

**Parametri:**

| Parametro | Tipo | Significato |
|---|---|---|
| `domain` | `SADomain` | Il dominio BURLAP (azioni disponibili, modello) |
| `gamma` | `double` | Fattore di sconto (0 = solo reward immediato, 1 = futuro infinito) |
| `entityIds` | `List<String>` | Lista fissa degli ID di tutte le entità del livello |
| `learningRate` | `double` | Learning rate per Adam optimizer |
| `epsilon` | `double` | Epsilon iniziale per esplorazione (tipicamente 0.5–1.0) |
| `decayEpsilonStep` | `double` | Riduzione di epsilon per episodio |
| `maxEpisodeSize` | `int` | Max step per episodio prima del cut-off |
| `epsilonMin` | `double` | Floor per epsilon (mai scende sotto questo valore) |
| `hiddenSize` | `int` | Neuroni per hidden layer (da config `burlap.network.hidden_size`) |

---

### `buildNetwork(inputSize, outputSize, lr, hiddenSize)`

**Scopo:** costruisce e inizializza una rete fully-connected con DL4J.

**Architettura:**
```
Input (n) → Dense(n → h, ReLU) → Dense(h → h, ReLU) → Output(h → n, Identity)
```

- **ReLU** sugli hidden layer: attivazione standard, evita il vanishing gradient
- **Identity** sull'output layer: i Q-values non hanno un range fisso, quindi NON si usa sigmoid/tanh che limiterebbero i valori
- **Adam** come optimizer: adattivo, gestisce bene i gradienti rumorosi tipici del RL
- **MSE** come loss function: errore quadratico medio tra Q predetti e Q target

**Usato da:** costruttore (due volte — una per `network`, una per `targetNetwork`)

---

### `encodeState(State s)`

**Scopo:** converte uno stato LabRecruits in un vettore numerico adatto alla rete.

**Come funziona:**
- Per ogni entità in `entityIds`, controlla se è osservata e il suo stato
  - `SWITCH` → 1.0 se `isOn`, 0.0 altrimenti
  - `DOOR` → 1.0 se `isOpen`, 0.0 altrimenti
  - Non osservata → 0.0 (default)
- Restituisce un `INDArray` di forma `(1, n)` — una riga per il batch size, n colonne per le features

**Nota importante:** l'ordine è determinato da `entityIds`, che è fisso e identico per tutti gli stati. Questo garantisce che la stessa entità occupi sempre la stessa posizione nel vettore.

**Usato da:** `runLearningEpisode`, `testDeepQLearningAgent`, `qValues`, `qValue`, `value`

---

### `qValues(State s)`

**Scopo:** calcola tutti i Q-values per uno stato dato. Richiesto dall'interfaccia `QProvider` di BURLAP.

**Come funziona:**
1. Esegue un forward pass su `network` con `encodeState(s)`
2. Per ogni azione applicabile, legge il Q-value all'indice corrispondente in `entityIds`
3. Restituisce una lista di `QValue`

**Usato da:** `EpsilonGreedy` internamente per scegliere l'azione (sia casuale che greedy)

---

### `qValue(State s, Action a)`

**Scopo:** Q-value per una coppia specifica `(stato, azione)`.

**Usato da:** BURLAP internamente se necessario; raramente chiamato direttamente.

---

### `value(State s)`

**Scopo:** valore `V*(s) = max_a Q(s, a)` — il miglior Q-value possibile da quello stato.

**Usato da:** BURLAP internamente.

---

### `runLearningEpisode(Environment env, int maxSteps)`

**Scopo:** esegue un intero episodio di training implementando il loop DQN completo.

**Relazioni:**
- Chiama `encodeState` → `learningPolicy.action` → `env.executeAction` → `addToReplayBuffer` → `trainOnBatch` → `updateTargetNetwork`
- È il cuore della classe: tutti gli altri metodi servono questo

**Vedi sezione 8 per il flusso dettagliato.**

---

### `addToReplayBuffer(Transition t)`

**Scopo:** inserisce una nuova transizione nel buffer con logica FIFO.

**Come funziona:**
- Se `replayBuffer.size() >= REPLAY_BUFFER_CAPACITY` → rimuove il primo elemento (il più vecchio)
- Aggiunge il nuovo in fondo

**Nota:** `remove(0)` su una `ArrayList` è O(n). Per buffer grandi (100k+), sarebbe più efficiente usare una `LinkedList` o un array circolare. Per 10.000 elementi è accettabile.

---

### `trainOnBatch()`

**Scopo:** campiona un mini-batch dal buffer e aggiorna i pesi della rete principale.

**Come funziona:**
1. Se il buffer ha meno di `MIN_REPLAY_SIZE` transizioni → esce subito (warm-up)
2. Campiona `BATCH_SIZE` transizioni con rimpiazzo (uniform random)
3. Impila i vettori stato in una matrice `(32, n)` con `Nd4j.vstack()`
4. Forward pass su **main network** → Q-values correnti (base per il target vector)
5. Forward pass su **target network** → Q-values per gli stati successivi (target Bellman)
6. Per ogni transizione nel batch:
   - Se terminale: `target = r`
   - Altrimenti: `target = r + γ * max(Q_target(s'))`
   - Sovrascrive solo la posizione dell'azione eseguita nel vettore target
7. `network.fit(stateBatch, targetBatch)` — un solo aggiornamento su tutto il batch

**Perché si sovrascrive solo una posizione?**
La rete produce Q-values per TUTTE le azioni. Conosciamo il target corretto solo per l'azione che abbiamo eseguito. Le altre posizioni del target vector sono copiate dall'output corrente della rete → errore zero su quelle posizioni → nessun aggiornamento per quelle azioni.

---

### `updateTargetNetwork()`

**Scopo:** copia i pesi della main network nella target network.

**Come funziona:**
```java
targetNetwork.setParams(network.params().dup());
```
`.params()` restituisce tutti i pesi come un unico vettore flat. `.dup()` crea una copia indipendente.

---

### `testDeepQLearningAgent(Environment env, int maxSteps)`

**Scopo:** testa la policy appresa senza modificare nulla nell'agente.

**Differenze da `runLearningEpisode`:**
- Selezione azione: sempre **greedy** (max Q-value, nessuna casualità)
- **Nessun** `network.fit()` → nessun aggiornamento dei pesi
- **Nessun** inserimento nel replay buffer
- **Nessun** decay di epsilon
- **Nessun** aggiornamento di `totalNumberOfSteps`

---

### `getMaxValuedAction(State s, INDArray qValues)`

**Scopo:** trova l'azione con il Q-value più alto — usata solo nel testing.

---

### `serializeModel(String path)` / `deserializeModel(String path)`

**Scopo:** salva/carica il modello su disco.

- `serializeModel`: salva pesi, architettura e stato dell'optimizer Adam (i "momenti" di Adam, utili per riprendere il training)
- `deserializeModel`: carica e poi chiama `updateTargetNetwork()` per mantenere coerenza tra le due reti

---

### `resetSolver()`

**Scopo:** reimposta l'agente a uno stato pulito.

**Cosa resetta:**
- Pesi di `network` (reinizializzazione Xavier)
- Pesi di `targetNetwork` (sincronizzati con main)
- `replayBuffer` (svuotato)
- `eStepCounter` e `totalNumberOfSteps`

**Cosa NON resetta:**
- `epsilongr` — usa `setEpsilongr()` se necessario

---

### `printNetworkSummary(PrintStream ps)`

**Scopo:** stampa un riepilogo dell'architettura della rete, degli entityIds, e dello stato corrente (dimensione buffer, step totali). Sostituisce la stampa della Q-table usata nel Q-Learning tabellare.

---

## 10. Parametri configurabili

Tutti i parametri si trovano in `src/test/resources/configurations/burlap_test.config`.

| Parametro | Default | Significato |
|---|---|---|
| `burlap.qlearning.lr` | 0.25 | Learning rate per Adam optimizer |
| `burlap.qlearning.gamma` | 0.59 | Fattore di sconto per reward futuri |
| `burlap.qlearning.epsilonval` | 0.5 | Epsilon iniziale (esplorazione) |
| `burlap.qlearning.epsilonmin` | 0.1 | Epsilon minimo (floor del decay) |
| `burlap.qlearning.decayedepsilonstep` | 0.95 | Riduzione epsilon per episodio |
| `burlap.num_of_episodes` | 3 | Numero di episodi di training |
| `burlap.network.hidden_size` | 64 | Neuroni per hidden layer |

**Costanti hardcoded** (modificabili nel codice in `DeepQLearningRL.java`):

| Costante | Valore | Dove |
|---|---|---|
| `REPLAY_BUFFER_CAPACITY` | 10000 | Dimensione massima del buffer |
| `BATCH_SIZE` | 32 | Dimensione del mini-batch |
| `MIN_REPLAY_SIZE` | 64 | Warm-up del buffer |
| `TARGET_UPDATE_FREQUENCY` | 100 | Step tra un sync della target network e il successivo |

---

## 11. Come replicare questo in un nuovo progetto

### Checklist per adattare la classe a un nuovo dominio

1. **Definisci il vettore di stato**
   - Identifica le features rilevanti del tuo ambiente
   - Sostituisci `encodeState()` con la tua logica di encoding
   - La dimensione del vettore = `inputSize` della rete

2. **Definisci le azioni**
   - Le azioni devono essere enumerabili e indicizzabili
   - La lista equivalente a `entityIds` determina `outputSize` della rete
   - Ogni azione corrisponde a un neurone di output

3. **Definisci il reward**
   - Il reward arriva dall'ambiente (`eo.r`) — non devi modificare il DQN per questo
   - Se il reward è molto grande o molto piccolo, considera di normalizzarlo (es. clip a [-1, 1])

4. **Regola i parametri**
   - Inizia con `epsilon = 1.0` e un decay lento
   - `gamma` vicino a 1.0 per task con reward ritardati (es. raggiungere un goal lontano)
   - `gamma` più basso (0.5–0.7) per task con reward immediati
   - `hidden_size = 64` è un buon punto di partenza; aumenta a 128 o 256 per stati complessi
   - `BATCH_SIZE = 32` è standard; `64` se hai GPU
   - `TARGET_UPDATE_FREQUENCY = 100` è conservativo; puoi ridurlo a 50 per ambienti veloci

5. **Segnali di convergenza (cosa monitorare)**
   - La reward media per episodio deve crescere nel tempo
   - L'epsilon deve scendere gradualmente
   - La loss di training (non esposta qui, ma loggabile da DL4J) deve diminuire
   - Se la reward oscilla violentemente → aumenta `TARGET_UPDATE_FREQUENCY` o `BATCH_SIZE`
   - Se la reward cresce molto lentamente → riduci `TARGET_UPDATE_FREQUENCY` o aumenta `lr`

6. **Estensioni possibili (per progetti futuri)**
   - **Double DQN:** usa main network per *scegliere* l'azione migliore in s', usa target network per *valutarla* → riduce la sovrastima dei Q-values
   - **Prioritized Experience Replay:** campiona transizioni con errore più alto più frequentemente → convergenza più rapida
   - **Dueling DQN:** divide l'output in valore di stato V(s) + vantaggio A(s,a) → migliore generalizzazione
   - **Reward shaping:** aggiungi reward intermedi per guidare l'esplorazione verso sottobiettivi

### Dipendenze necessarie (Maven)

```xml
<!-- BURLAP - framework RL -->
<dependency>
    <groupId>com.github.jmacglashan</groupId>
    <artifactId>burlap</artifactId>
</dependency>

<!-- DL4J - rete neurale -->
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-core</artifactId>
</dependency>

<!-- ND4J - tensori (backend DL4J) -->
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native-platform</artifactId>
</dependency>
```

---

*Guida generata a partire da `DeepQLearningRL.java` — progetto RLbT, giugno 2026.*
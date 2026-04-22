# Log dei Progressi: Progetto Segmentazione OpenEarthMap

Questo documento tiene traccia cronologica di tutti gli step di sviluppo, delle difficoltà tecniche riscontrate e delle soluzioni ingegneristiche applicate per superarle. Questo report fungerà da documentazione tecnica per la validazione del progetto.

---

## Fase 1: Baseline e Identificazione del "Cap" Prestazionale
**Obiettivo iniziale:** Addestrare una Lightweight U-Net (Swin-Tiny backbone) su OpenEarthMap (8 classi) in ambiente Kaggle.
**Risultato ottenuto:** Il modello convergeva lentamente e si "congelava" irrimediabilmente attorno al **30% di accuratezza (mIoU)**.

### 🕵️ Limitazioni Riscontrate (Analisi Critica)
Dall'analisi del codice sono emerse molteplici colli di bottiglia critici:
1. **Perdita di Dettagli Spaziali (Preprocessing):** L'utilizzo di `A.Resize(224, 224)` su immagini satellitari distruggeva fisicamente la risoluzione nativa (GSD). Strutture piccole (es. automobili o tetti) si tramutavano in artefatti di 1-2 pixel irriconoscibili per i layer convoluzionali.
2. **Metric Miscalculation:** La funzione `evaluate()` calcolava l'accuracy su *tutti* i pixel, incluso il "background sconosciuto" (Classe 0). Essendo la rete addestrata a ignorare la classe 0, l'accuracy globale veniva matematicamente penalizzata verso il basso a prescindere dalla bontà delle predizioni sulle classi valide.
3. **Catastrophic Forgetting (Ciclo di Training):** Il backbone veniva scongelato all'epoca 4 creando *ex-novo* un optimizer Adam. Questo distruggeva completamente il "Momentum" faticosamente acquisito dal decoder nelle prime 3 epoche, causando un riavvio asimmetrico e spike letali nei gradienti.
4. **Fragilità dei Pesi Pre-addestrati (RSP):** Il mapping manuale dei tensori dal checkpoint *Remote Sensing Pretrained* (RSP) all'architettura più recente di `timm`/`smp` falliva parzialmente, lasciando molti layer della Swin-Tiny inizializzati in maniera randomica.

---

## Fase 2: Ristrutturazione della Pipeline (Le Soluzioni)
Per sbloccare il modello, sono state progettate e integrate le seguenti correzioni mirate nel codice sorgente (`transforms.py`, `engine.py`, `main_kaggle.py`, `lightweight_unet.py`):

1. **Random Crop Preservativo:** Abbiamo sostituito il ridimensionamento con `A.RandomCrop(224, 224)` (e `CenterCrop` per la validazione). Questa mossa ha permesso alla rete di processare le frequenze altissime delle immagini senza distorcerle, supportata da normalizzazione canonica ImageNet. Mantenute le essenziali *Horizontal/Vertical Flips e Random Rotations*.
2. **Advanced Metric & Loss Engineering:** 
   - L'Accuracy è stata riscritta per escludere i pixel con target 0.
   - È stata aggiunta formalmente la **metrica F1 Score** calcolata via *Precision e Recall*.
   - La funzione di costo è stata aggiornata implementando una **Focal Loss + Dice Loss** (per iniziare ad aggredire le classi in minoranza).
3. **Differential Learning Rates:** Per salvare il momentum, l'optimizer è stato inizializzato subito (Epoca 0) passando l'intero set di layer, assegnando un *Learning Rate* standard (`1e-4`) al decoder e un LR miscroscopico (`1e-5`) al backbone. Lo sblocco all'epoca 4 ora avviene fluidamente attivando `requires_grad=True` senza sovrascrivere l'optimizer.
4. **Fallback Pesi ImageNet:** Ripristinata la solidità usando in via predefinita i sicuri pesi ImageNet, garantendo una rete performante sin dallo step zero.

---

## Fase 3: Risultati Intermedi e Nuove Difficoltà
**Nuovo Risultato ottenuto:** Sfruttando le GPU T4 di Kaggle su 50 epoche, **la metrica mIoU è salita drasticamente dal 30% a circa il 53%, con Dice e F1 Score al 60%**. Un balzo prestazionale enorme.

### 🕵️ Nuove Limitazioni (Il Muro delle Classi Rare)
Nonostante l'eccellente salute della pipeline, l'ispezione delle predizioni (`samples/`) rivela che alcune classi appaiono molto raramente o vengono classificate come classi adiacenti. 
- **Causa (Extreme Class Imbalance):** Le classi come *Acqua*, *Terreno* o *Alberi* dominano geograficamente il dataset. Al contrario, *Macchine* o *Piccoli Edifici* sono in fortissima minoranza pixel.
- Pur con la Focal Loss attiva, se le immagini fornite al training sono fisicamente poche, il modello semplicemente non può immagazzinare una varianza sufficiente di forme di un veicolo o di un palazzo visti in diverse angolazioni/luci.

---

## Fase 4: Implementazione "Additional Task" — Wavelet-based Strategies
**L'obiettivo:** Migliorare l'IoU esaltando i contorni e arginare lo stallo.
Essendo il dataset nativamente sprovvisto di input multi-modali ausiliari (come mappe di elevazione DSM o bande Infrarosse), abbiamo attuato la task aggiuntiva proposta (ricerca *ISPAMM Lab*): l'uso delle **Trasformate Wavelet (DWT)**.

**Implementazione Tecnica:**
1. Tramite `pywt`, estraiamo le alte frequenze (LL, LH, HH) convertendo l'immagine in scala di grigi per catturare i bordi matematici perfetti delle geometrie urbane e naturali.
2. Abbiamo concatenato questa mappa di bordi come **4° Canale** al tensore RGB originale.
3. Riconfigurata la U-Net (`smp.Unet`) e il sistema di normalizzazione (`A.Normalize`) per elaborare nativamente input a 4 canali.
4. Applicato un fix visivo su `utils/plots.py` per de-normalizzare correttamente solo i primi 3 canali durante il plotting, evitando il collasso di Matplotlib.

**Risultati della Fase 4:** I contorni predetti sono visibilmente più netti (aumento del potenziale sul Dice Score), ma **l'mIoU complessivo si è stabilizzato sempre intorno al 50-53%**. Questo ha confermato l'ipotesi critica: *migliorare la percezione dei bordi non risolve l'assenza intrinseca di dati per le classi rare (Sbilanciamento Estremo)*.

---

## Fase 5: Soluzione Definitiva per Classi Rare — Pseudo-Labeling (In Progress)
**Perché ci serve:** Non è più un problema di rete o di funzione di costo, ma di *Dati*. Se in 50 epoche il modello vede un'automobile 20 volte e una foresta 20.000 volte, non imparerà mai a generalizzare l'automobile. Per superare l'80% di mIoU dobbiamo aumentare drasticamente l'esposizione della rete alle minoranze. Non potendo etichettare a mano milioni di pixel, ci affidiamo al **Semi-Supervised Learning**.

**Cosa andremo a fare (Il Piano Operativo):**
1. **Raccolta Dati Unlabeled:** Prenderemo migliaia di immagini satellitari grezze (senza maschere umane).
2. **Generazione "Coarse":** Passiamo le immagini alla nostra attuale U-Net (addestrata al 50%). La rete indicherà le zone "semanticamente" corrette (es. "qui c'è un edificio"), ma con bordi imprecisi.
3. **Generazione "Crisp" con SAM:** Passiamo le stesse immagini a **Segment Anything Model (SAM)** di Meta in modalità Zero-Shot. SAM non sa dare i nomi agli oggetti, ma genera poligoni vettoriali e ritagli fisici perfetti.
4. **La Fusione (Lo script `pseudo_labeling.py`):** Uniamo i due output. Per ogni maschera perfetta ritagliata da SAM, le assegniamo il nome della classe che la nostra U-Net ha "visto" all'interno di quel poligono.
5. **Re-Training:** Otterremo un enorme dataset pseudo-etichettato ad altissima fedeltà. Riaddestrando il modello da zero su questa mole di dati massiva, l'accuratezza esploderà fisiologicamente verso il target dell'80%+.

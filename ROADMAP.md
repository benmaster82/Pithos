# Roadmap Pithos

Stato al 2026-10-24 (v0.4). Riferimento hardware: i5-10500H, 16 GB, GTX 1650 Max-Q 4 GB.

Un token in generazione costa oggi circa 55–70 ms:

| Voce | ms/token |
|---|---:|
| Esperti non residenti calcolati sulla CPU | 30–45 |
| Attesa della GPU | ~10 |
| Accodamento dei comandi e combine | ~8 |
| Logit | ~3,5 |

Le voci sono in ordine di impatto atteso.

## 1. Esperti sulla CPU (collo di bottiglia attuale)
Il modello (13,5 GB) non sta nella RAM libera (9–10 GB): parte degli esperti viene letta dal disco
durante la generazione.

Gli esperti sulla CPU sono limitati dalla banda della RAM (~660 MB di pesi per token a ~20 GB/s):
il tempo scala con i byte letti.

- [ ] **Esperti in formato più compatto** (3 bit o meno; Q4_0 non serve, occupa quanto Q4_K).
  Migliore candidato: [APEX I-Compact](https://huggingface.co/mudler/LFM2.5-24B-A2B-APEX-GGUF),
  quantizzazione mista per questo modello partita da BF16 con imatrix: 10,4 GiB contro 13,5,
  perplexity +1,6% (16,20 contro 15,94), HellaSwag invariato entro il rumore. Prima verificare dai
  metadati del file quali formati usa (probabilmente IQ4_XS/IQ3, oggi non supportati da Pithos).
  Ricomprimere noi dal Q4_K darebbe risultati peggiori (errori sommati, niente imatrix).
  Variante dinamica (esperti freddi compatti, caldi a 4 bit, in un file affiancato) solo dopo,
  se il file statico regge. Guadagno atteso: da 14–18 a ~17–21 tok/s, più stabile.
- [ ] **Prefetch in VRAM guidato dal predittore**: caricare sulla GPU gli esperti previsti per il
  layer successivo. Guadagno modesto (1–2 caricamenti possibili per layer, ~0,27 ms per esperto).
- [ ] **Più slot esperti in VRAM** recuperando memoria (KV cache int8, vedi punto 4): marginale a
  contesto 8k (~15 esperti, +0,6%), utile solo con contesti lunghi.

## 2. Decodifica speculativa senza modello ausiliario (prompt lookup)
- [ ] Proporre i token successivi cercando n-grammi già presenti nel contesto e verificarli in un
  solo passaggio a batch (gli esperti si condividono fra i token del batch).
  Atteso: +30–80% su testo ripetitivo (agente: nomi di file, codice, risultati dei tool), poco sul
  testo libero.

## 3. Prefill sulla GPU
- [ ] Portare sulla GPU anche i layer densi dell'elaborazione del prompt (oggi a batch sulla CPU,
  36–45 tok/s a caldo, molto meno a freddo). Riduce l'attesa della prima risposta in modalità
  agente, dove il prompt di sistema con i tool è lungo.

## 4. Funzionalità
- [ ] **Avvio a freddo dell'agente**: precalcolare all'avvio il prompt di sistema con i tool.
- [ ] **Output JSON vincolato**: grammatica che garantisce chiamate ai tool ben formate.
- [ ] **Contesto più lungo / KV cache int8**: meno VRAM e RAM per token di contesto.
- [ ] **Richieste parallele nel server** (oggi una alla volta, le altre in coda).
- [ ] **Test su altro hardware**: kernel NEON su ARM64, percorso f32 su GPU AMD e Intel.

## 5. Rifiniture
- [ ] Fondere il combine dei MoE nel primo kernel del layer successivo (residual stream in
  ping-pong): ~1 ms per token.

## Fatto
- v0.4: layer densi in VRAM con kernel fusi (14–18 tok/s su testo reale contro 11–14).
- v0.3: modalità agente con strumenti integrati e client MCP.
- v0.2: server compatibile OpenAI, riuso del prompt, interfaccia web (oggi Pandora).
- v0.1: motore C senza dipendenze, cache degli esperti su GPU via OpenCL.

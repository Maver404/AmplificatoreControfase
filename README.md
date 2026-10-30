# Amplificatore in controfase

Web app che, in tempo reale, prende il segnale di un **microfono di riferimento** posto a monte (verso la
sorgente del rumore), lo rigenera **in controfase** e lo invia a un amplificatore e a una o due casse, così da
attenuare il rumore in una zona da silenziare. È un sistema di cancellazione attiva del rumore (ANC)
**feed-forward** che gira nel browser: microfono → AudioWorklet (DSP) → due uscite indipendenti L/R, con un
**microfono d'errore** opzionale nella zona per misurare e regolare automaticamente il risultato.

Funziona in **Chrome su macOS e Windows** con **un'unica interfaccia audio USB** per ingresso e uscita.
Interfaccia in italiano; stack: Vite + TypeScript, test con Vitest e Playwright, immagine Docker con nginx.

> **Sicurezza prima di tutto.** L'app pilota casse reali. Prima di ogni avvio porta l'amplificatore **al
> minimo** e tieni la mano sulla manopola del volume o sull'interruttore. `Esc` (PANIC) funziona solo se la
> finestra dell'app ha il fuoco. Leggi la sezione [Sicurezza](#sicurezza) prima del primo utilizzo.

## Indice

- [Cosa fa (e cosa no)](#cosa-fa-e-cosa-no)
- [Limiti fisici, con numeri](#limiti-fisici-con-numeri)
- [Layout A e layout B](#layout-a-e-layout-b)
- [Hardware](#hardware)
- [Configurazione di macOS](#configurazione-di-macos)
- [Configurazione di Windows](#configurazione-di-windows)
- [Avvio](#avvio)
- [Configurazione guidata (passi 1–7)](#configurazione-guidata-passi-17)
- [Sicurezza](#sicurezza)
- [Checklist di prova sull'hardware](#checklist-di-prova-sullhardware)
- [Risoluzione problemi](#risoluzione-problemi)
- [Struttura del progetto](#struttura-del-progetto)
- [Cosa è verificato automaticamente e cosa solo sull'hardware](#cosa-è-verificato-automaticamente-e-cosa-solo-sullhardware)

## Cosa fa (e cosa no)

**Catena di elaborazione** (doppia precisione, nessuna allocazione nel ciclo audio):

1. **Neutralizzazione della retroazione** (dopo la calibrazione): sottrae dal riferimento la parte di
   segnale che arriva dalla cassa, con un filtro FIR misurato cassa → microfono.
2. **Condizionamento**: blocco della continua (3 Hz) → passa-alto di protezione (vento e rombo; attivo, 20 Hz
   di default, 10–200 Hz) → compensazione barriera opzionale (passa-basso del 1° ordine che imita una
   parete/finestra) → passa-basso Bessel (500 Hz di default, 100–4000 Hz).
3. **Modalità tonale** (solo per rumori periodici): banco di risonatori sulle armoniche di f0, così si
   riproduce solo il tono e non il rumore a banda larga.
4. **Per ogni uscita L/R**: ritardo frazionario (precisione sotto il campione) → polarità → guadagno → muto e
   avvio graduale → volume master → limitatore con tetto (ceiling) in dBFS, preceduto dal rilevatore di
   innesco.

**Una sola convenzione di fase**: fase totale = polarità (0° o 180°, default 180°) + Δ, con Δ ∈ [−180°, +180°]
e positivo = ritardo, riferita alla frequenza f_ref (a f0 in modalità tonale). Tre modalità:

| Modalità | Quando | Come agisce Δ |
|---|---|---|
| **Ritardo** (default) | Qualunque rumore | Ritardo aggiuntivo Δ/(360°·f_ref), nessuna dispersione |
| **Tonale** | Rumori periodici stabili (motori, compressori, trasformatori); suggerita quando l'app rileva un tono dominante | Ritardo avvolto su periodi interi 1/f0; f0 stimata con precisione sub-bin e inseguita |
| **Hilbert** (avanzata) | Ottimizzazione con mic d'errore e regolazione automatica | Rotazione di fase a larga banda; aggiunge ritardo di gruppo (≈ 2,6 ms a 100 Hz) |

**Con il mic d'errore** l'app misura i percorsi acustici, calcola ritardo/guadagno/polarità ottimali, misura
l'attenuazione reale (confronto A/B con il bypass) e regola automaticamente. **Senza mic d'errore** usa la
geometria (distanze, angolo, temperatura) e una calibrazione a due posizioni con l'unico microfono, ma il
risultato **non è misurabile**.

**Cosa non fa**: non è un FxLMS adattivo completo; non usa ASIO su Windows; non funziona con Bluetooth o
AirPlay; non crea una "bolla di silenzio" in tutta la stanza.

## Limiti fisici, con numeri

Questi limiti valgono per qualunque sistema di questo tipo; l'app li mostra anche nella guida integrata.

**Latenza** (microfono → cassa, nel browser ben configurato). L'app usa sempre la latenza **misurata**.

| Configurazione | Latenza tipica |
|---|---|
| macOS + interfaccia USB + Chrome | ≈ 15–30 ms (es. 25,7 ms con quanto 64, 28,6 ms con 128) |
| Windows (WASAPI condiviso) | più alta: ≈ 10 ms in più per direzione |
| Impostazioni di default del browser | 55–120 ms |
| Bluetooth / AirPlay | +150–275 ms → ANC impossibile (bloccati dall'app) |

**Causalità (rumore a banda larga).** Il rumore deve impiegare, per andare dal microfono alla zona, almeno il
tempo della catena microfono → computer → cassa → zona:
`(d sorgente→zona − d sorgente→mic) / c ≥ latenza + percorso cassa→zona`.
Servono **≈ 3,4 m di vantaggio acustico ogni 10 ms**. Esempio: 25 ms di latenza + cassa a 1 m dalla zona
(≈ 3 ms) → il microfono deve stare **≈ 9,6 m più vicino alla sorgente** di quanto lo sia la zona, lungo la
direzione di arrivo del rumore.

**Rumore periodico e stabile.** Si allinea su periodi interi, quindi funziona **a qualunque distanza**, purché
f0 sia stabile e stimata con precisione (≈ 0,01 %). Esempio: 0,1 Hz di deriva con 100 ms di ritardo totale
danno 3,6° di errore alla fondamentale e 14,4° alla 4ª armonica.

**Tolleranze.** Residuo `A = 10·log10(1 + g² − 2g·cos Δφ)` (g = rapporto di ampiezza, Δφ = errore di fase):

| Attenuazione voluta | Errore di fase max | Errore di ampiezza max | A 100 Hz |
|---|---|---|---|
| 10 dB | 18° | −3,3 / +2,4 dB | ± 0,5 ms |
| 20 dB | 5,7° | ≈ ± 0,8 dB | ± 159 µs |

**Zona di quiete** locale, dell'ordine di **λ/10** attorno al punto ottimizzato: 50 Hz ≈ 69 cm · 100 Hz ≈ 34 cm
· 200 Hz ≈ 17 cm · 500 Hz ≈ 7 cm.

**Risultato realistico: 10–20 dB su circa un'ottava sotto ≈ 500 Hz, oppure sui toni.** Sopra ≈ 500 Hz la zona
si riduce a pochi centimetri (per questo il passa-basso di default è a 500 Hz).

**Parete o finestra tra microfono e zona**: il rumore che la attraversa viene filtrato (perde gli acuti e
ruota di fase), non solo ritardato, quindi l'attenuazione ottenibile **cala**. La compensazione barriera ne
imita solo l'effetto principale.

**Temperatura**: `c = 331,3·√(1 + T/273,15)` m/s (343 m/s a 20 °C). 10 °C in meno allungano di ≈ 0,5 ms un
percorso di 10 m, cioè ≈ 18° a 100 Hz: se la temperatura cambia molto, ricalibra.

## Layout A e layout B

```
LAYOUT A (consigliato)

  sorgente      mic rif.        parete/        cassa          zona silenziata
    (*)  ))))    [o=     ║░░░░░║ finestra       [|>   - - ->   (  mic err. [o=  )
         ──── rumore ────║░░░░░║─────────────────────────────────▶
                                                anti-rumore ─ ─ ─▶  (stessa direzione)

LAYOUT B (sconsigliato)

  sorgente      mic rif.        parete/        zona silenziata              cassa
    (*)  ))))    [o=     ║░░░░░║ finestra    ( |  |  |  |  | )             <|]
         ──── rumore ────║░░░░░║─────────────────────────────────▶
                                             ◀─ ─ ─ ─ ─ ─ ─ ─ ─ anti-rumore
                   ▲ . . . . . . . . . . . retroazione cassa → mic . . . . .
```

**Layout A (consigliato)**: microfono **all'esterno**, rivolto verso la sorgente (oltre la finestra o la
parete); cassa **all'interno, vicino all'apertura**, che suona verso la zona. Rumore e anti-rumore viaggiano
nello stesso verso, quindi la cancellazione resta coerente lungo il percorso; la parete attenua il ritorno
cassa → microfono (meno retroazione, più guadagno utilizzabile).

**Layout B (supportato con avvisi)**: cassa **oltre la zona**, rivolta verso il microfono. Le due onde
viaggiano in versi opposti: si cancellano solo su **strati sottili distanti λ/2** (a 100 Hz ogni ≈ 1,7 m),
con rinforzi fino a +6 dB in mezzo; inoltre la cassa punta verso il microfono → forte **retroazione** e
rischio di innesco. Usalo solo se il layout A è impossibile.

La guida integrata nell'app mostra gli stessi schemi a colori.

## Hardware

- **Un'unica interfaccia audio USB** per ingresso e uscita (stesso clock: niente deriva tra dispositivi),
  impostata a **48 kHz**.
- **Ingresso 1: microfono di riferimento**; **ingresso 2: microfono d'errore** (opzionale). Chrome su Mac
  consegna al massimo 2 canali d'ingresso: usa gli ingressi 1–2.
- **Microfono di riferimento direzionale** (cardioide o supercardioide) **puntato verso la sorgente**, con la
  cassa nella sua direzione di minima sensibilità. All'aperto: **antivento** (spugna + pelliccia) e riparo
  dalla pioggia; fissato saldamente.
- **Uscite L/R** verso l'amplificatore: una cassa (R spenta) oppure due casse (stessa zona o due zone).
  Per frequenze sotto ≈ 80 Hz serve un subwoofer.
- Amplificatore con **manopola del volume e interruttore a portata di mano**.
- **Mai** Bluetooth o AirPlay; evita il microfono e le casse integrati del computer (bloccati dall'app).
- Cuffie: consentite solo per prove, con tetto d'uscita forzato a −24 dBFS.

## Configurazione di macOS

1. **Configurazione MIDI Audio** (Applicazioni → Utility): seleziona l'interfaccia e imposta il formato a
   **48.000 Hz** sia per l'ingresso sia per l'uscita.
2. **Impostazioni di Sistema → Privacy e sicurezza → Microfono**: abilita **Google Chrome** (riavvia Chrome se
   lo richiede).
3. In Chrome consenti il microfono al sito (icona a sinistra della barra degli indirizzi).
4. Nell'app scegli la **stessa interfaccia** come ingresso e uscita (non serve che sia quella di sistema).
5. Se compare **Modalità microfono** nel Centro di controllo, lascia *Standard* (niente isolamento della voce).
6. Collega l'alimentatore: il risparmio energetico può causare interruzioni dell'audio.

## Configurazione di Windows

1. **Pannello di controllo Audio** (`Win+R` → `mmsys.cpl`), schede **Riproduzione** e **Registrazione**:
   interfaccia → **Proprietà → Avanzate → Formato predefinito**: **lo stesso 48000 Hz** su entrambe.
2. Disattiva **Miglioramenti audio** (Audio enhancements), **Audio spaziale** (Spatial sound) ed
   **Equalizzazione volume** (Loudness Equalization), se presenti.
3. Scheda **Comunicazioni**: **"Non eseguire alcuna operazione"** (Do nothing), così Windows non abbassa i
   volumi durante le "chiamate".
4. **Impostazioni → Privacy e sicurezza → Microfono**: consenti l'accesso alle app desktop e a Chrome.
5. Chrome usa **WASAPI in modalità condivisa**: **ASIO non viene usato** e la latenza è **più alta** che su
   macOS. Alcuni driver hanno un buffer WDM separato da quello ASIO: impostalo al minimo stabile.
6. Piano energetico **Prestazioni elevate**; evita hub USB senza alimentazione.

Con la latenza più alta di Windows la causalità a banda larga richiede più distanza: per rumori periodici
la modalità tonale funziona comunque.

## Avvio

### Sviluppo e test (Node 24)

```sh
nvm install 24        # una volta sola
nvm use 24            # legge .nvmrc
npm ci
npm run dev           # http://localhost:5173
npm test              # test unitari e simulatore ad anello chiuso (Vitest)
npm run typecheck     # tsc -b
npm run build         # sito statico in dist/
npm run test:e2e      # test end-to-end in Chrome con microfono finto (Playwright)
```

I test end-to-end usano **Google Chrome installato nel sistema** (canale `chrome` di Playwright), non il
Chromium scaricato da Playwright. Se Playwright segnala che manca (`Chromium distribution 'chrome' is not
found`), installa Chrome oppure esegui una volta `npx playwright install chrome`.

**Contesto sicuro**: `getUserMedia` (il microfono) funziona solo su **https://** oppure su
**http://localhost** / **http://127.0.0.1**. Un indirizzo IP in http (es. `http://192.168.1.10:8080`) **non**
è un contesto sicuro: il microfono verrà negato.

### Docker

Richiede Docker Desktop (o Docker Engine ≥ 23 con BuildKit e Compose v2).

```sh
docker compose up --build
```

- **http://localhost:8080** — contesto sicuro perché è localhost.
- **https://localhost:8443** — certificato autofirmato generato all'avvio del container.

La build esegue `npm ci` → `npm test` → `npm run build` (un test che fallisce blocca l'immagine), poi copia
`dist/` in `nginx:1.30.6-alpine`. Durante la build vengono verificati anche lo script del certificato e
`nginx -t`. Il container ha un healthcheck su `http://localhost/` (`docker compose ps` mostra `healthy`).

**Uso da un altro dispositivo in LAN** (serve HTTPS): aggiungi l'IP del computer al certificato con
`EXTRA_SAN` (voci `IP:` o `DNS:` separate da virgole) e apri `https://<IP>:8443`.

```sh
# macOS / Linux (IP del Mac: ipconfig getifaddr en0)
EXTRA_SAN="IP:192.168.1.10,DNS:mypc.local" docker compose up --build

# Windows PowerShell (IP: ipconfig)
$env:EXTRA_SAN = "IP:192.168.1.10"; docker compose up --build
```

In alternativa scrivi `EXTRA_SAN=IP:192.168.1.10` in un file `.env` accanto a `docker-compose.yml`. Se
cambi `EXTRA_SAN`, il certificato viene rigenerato al riavvio. Il browser mostrerà un avviso per il
certificato autofirmato: confronta l'impronta SHA-256 stampata nei log
(`docker compose logs controfase | grep SHA-256`) e prosegui (Avanzate → Procedi).

**Certificato tuo** (consigliato per un uso ripetuto, es. con [mkcert](https://github.com/FiloSottile/mkcert)):

```sh
mkcert -install        # una volta: installa la CA locale di mkcert
mkdir -p certs
mkcert -cert-file certs/server.crt -key-file certs/server.key localhost 127.0.0.1 192.168.1.10
```

poi togli il commento alle righe `volumes:` in `docker-compose.yml` (`./certs:/etc/nginx/certs:ro`) e
riavvia (`docker compose up -d --force-recreate`). Gli altri dispositivi della LAN si fidano del
certificato solo se installi anche lì la CA di mkcert (`mkcert -CAROOT` indica dove si trova). Servono entrambi i file, in PEM, con chiave senza passphrase: lo script li controlla e si ferma con
un messaggio chiaro se mancano, non corrispondono o la chiave è cifrata. La cartella `certs/` è esclusa da
git e dal contesto di build. Le righe sono commentate perché, se `./certs` non esiste, Docker creerebbe una
cartella vuota di proprietà di root.

Per pubblicare le porte solo su questo computer usa `"127.0.0.1:8080:80"` e `"127.0.0.1:8443:443"`.

**Cosa fa nginx** (`docker/nginx.conf`, `docker/security-headers.conf`): `mime.types` completo (i file
`.js` come JavaScript, indispensabile per moduli, worker e AudioWorklet), gzip per JS/CSS/SVG/JSON,
`index.html` con `Cache-Control: no-cache`, `/assets/` con cache lunga `immutable` (i nomi contengono
l'hash), 404 per i file mancanti e fallback a `index.html` per le altre rotte, e su ogni risposta
`Permissions-Policy: microphone=(self)`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` e
una `Content-Security-Policy` che consente solo script, worker e worklet dello stesso origine (nessuno
script inline). Verifica rapida:

```sh
curl -I http://localhost:8080/
curl -kI https://localhost:8443/
curl -I http://localhost:8080/assets/<un-file>.js   # Content-Type JavaScript + cache immutable
```

## Configurazione guidata (passi 1–7)

Al primo avvio l'app propone la configurazione guidata e **sblocca l'uscita solo alla fine**. L'avvio è a
step: "Consenti microfono" (uscita a zero) → scelta dispositivi e validazione → "Attiva uscita".

1. **Dispositivi e validazione** — scegli l'interfaccia per ingresso e uscita. Bloccanti: Bluetooth/AirPlay,
   microfono + casse integrati, elaborazioni (cancellazione eco, soppressione rumore, controllo automatico del
   guadagno) rimaste attive. Avviso con "continua comunque": latenza d'uscita > 40 ms, frequenze di
   campionamento diverse. Cuffie: tetto −24 dBFS e conferma.
2. **Livello dell'amplificatore** — con l'ampli al minimo l'app emette rumore rosa in banda a −20 dBFS
   attraverso il limitatore; alza l'ampli fino a circa il livello del rumore nella zona (≤ 85 dBA). Se in
   seguito la calibrazione chiede un guadagno sopra −6 dB, l'app ti dirà di alzare il guadagno d'ingresso
   (preamplificatore) del microfono di riferimento, non l'amplificatore; sotto −30 dB, di abbassare
   l'amplificatore.
3. **Layout e configurazione** — layout A o B, una o due casse (stessa zona o zone separate), parete/finestra
   in mezzo (compensazione barriera), presenza del mic d'errore.
4. **Posizionamento** — distanze per canale, direzione della sorgente θ (avviso se cos θ < 0,7), temperatura;
   verifica di causalità con il vantaggio richiesto in metri.
5. **Misura dei percorsi** — sweep limitati alla banda operativa, prima un'anteprima 20 dB più bassa, sempre
   annullabili con `Esc`: cassa → riferimento (retroazione, polarità e limite di guadagno), cassa → errore,
   riferimento → errore dal rumore ambiente, loopback opzionale per la latenza elettrica.
6. **Allineamento** — con il mic d'errore: risolutore su ritardo, guadagno, Δ e polarità con vincolo di
   stabilità dell'anello; senza: geometria + calibrazione a due posizioni (mic nella zona, poi di nuovo a monte).
7. **Verifica A/B** — confronto anti-rumore attivo/bypass e attenuazione misurata (larga banda e per terzi
   d'ottava, con il mic d'errore).

Con una calibrazione salvata, all'avvio l'app chiede se hai spostato microfoni, casse o amplificatore:

- **«No, usa la calibrazione salvata»**: dispositivi (1) → livello dell'amplificatore (2) → **Controllo rapido**
  (5) → verifica A/B (7). Il controllo rapido fa suonare un breve sweep per ogni cassa attiva sul percorso
  cassa → microfono di riferimento e lo confronta con quello salvato: entro 0,5 campioni e 0,5 dB tiene i valori
  salvati; per piccole differenze (fino a 6 dB o 3 ms: l'amplificatore regolato di nuovo a orecchio, la latenza
  di un nuovo contesto audio) corregge guadagno e ritardo dei canali e sostituisce il percorso salvato (la
  neutralizzazione del feedback lo segue); per differenze maggiori, polarità invertita o firma di un cavo di
  loopback chiede di rifare le misure. Se la misura non riesce i valori salvati restano, ma vanno accettati
  esplicitamente. Una calibrazione misurata con **un'altra interfaccia audio** viene rifiutata (con il pulsante
  per rifare le misure). Uscendo dal passo 2 torna il volume master salvato.
- **«Sì»**: si ripetono posizionamento, misure e allineamento (passi 4–6); oppure si rifà tutta la configurazione.

## Sicurezza

- **Ampli al minimo prima di ogni avvio**, mano sulla manopola del volume o sull'interruttore.
- **PANIC** (sempre visibile) o **`Esc`**: rampa a zero in 10 ms, poi sospensione dell'audio. **`Esc` funziona
  solo se la finestra dell'app ha il fuoco**: se stai usando un'altra finestra, agisci sull'amplificatore.
- **`M`** = muto (rampa 15 ms), **`B`** = bypass A/B.
- L'uscita resta **a zero** finché i dispositivi non sono validati; all'attivazione sale in 1,5 s. Ogni
  aumento di guadagno oltre 6 dB (preset, valore digitato, doppio clic) sale al massimo di 6 dB/s.
- **Cuffie**: tetto d'uscita **forzato a −24 dBFS**. Mai Bluetooth/AirPlay.
- **Rilevatore di innesco** nel worklet (prima del limitatore): silenzia automaticamente e mostra la
  frequenza. Falsi positivi noti e accettati: un tono che sale molto in fretta, oppure un suono periodico
  (tono, macchina) che il limitatore deve ridurre di almeno 0,5 dB senza sosta (se lo sfiora appena, suona).
- **Silenziamento automatico** su sospensione/interruzione dell'audio, cambio o scollegamento dei
  dispositivi, errori del processore, più di 3 interruzioni (underrun) in 10 s o salto di latenza.
- **Cavo di loopback** (misura opzionale della latenza): **solo sull'ingresso libero (il 2, o l'1 se il riferimento
  è sul 2), phantom (48 V) spento, gain al minimo**; l'anti-rumore è muto durante la misura. **Scollegalo subito dopo**: prima di attivare l'uscita
  l'app cerca la firma di un loopback e, se la trova, rifiuta.
- **Preset** validati e riportati nei limiti; lo sblocco del guadagno a +12 dB non viene mai salvato né
  importato; il caricamento avviene con muto → applica → avvio graduale.
- Livelli: nella zona non superare ≈ 85 dBA.

## Checklist di prova sull'hardware

Da eseguire **in quest'ordine**, prima su macOS e poi **ripetuta identica su Windows**. Non passare al punto
successivo finché il precedente non è superato.

| # | Prova | Come | Criterio di superamento |
|---|---|---|---|
| 1 | **Ampli al minimo** | Ampli al minimo (o spento), avvia l'app, consenti il microfono, scegli i dispositivi | Meter del riferimento attivo; meter d'uscita a −∞ prima di "Attiva uscita"; nessun suono |
| 2 | **PANIC / Esc** | Ampli basso, uscita attiva con un segnale udibile; premi `Esc`, poi riattiva e premi PANIC; prova anche `M` | Silenzio percepito immediato (rampa 10 ms) e audio sospeso; meter d'uscita a −∞ entro 0,1 s. Con la finestra **senza fuoco** `Esc` non agisce: è il comportamento atteso, usa l'ampli |
| 3 | **Innesco provocato a basso guadagno** | Ampli basso, mic vicino e rivolto alla cassa (caso peggiore), guadagno del canale alzato a passi di 1 dB | Silenziamento automatico entro ≈ 1 s dall'inizio dell'innesco (≈ 0,4 s da quando il limitatore comincia a intervenire senza sosta), frequenza mostrata, uscita mai oltre il tetto; nessun fischio sostenuto oltre 1 s |
| 4 | **Latenza elettrica L_e ripetibile** | Cavo di loopback uscita → **ingresso 2** (phantom spento, gain minimo). Passo 5 della configurazione guidata: «Misura il loopback» (5 sweep a −30 dBFS), poi «Ripeti 10 dB più forte» (−20 dBFS); annota L_e e **scollega il cavo**. In una sessione successiva (Chrome o audio riavviati) ripeti la misura | L'app mostra L_e in campioni con 3 decimali, la dispersione σ e il numero di misure (es. «−30 dBFS: 1234,567 campioni (25,720 ms), σ 0,012 campioni su 5 misure»): σ ≤ 0,5 campioni; «Differenza tra −30 e −20 dBFS» ≤ 0,5 campioni (oltre: il sistema o il driver elaborano il segnale); L_e della nuova sessione uguale a quella annotata entro 0,5 campioni (se no: latenza variabile al riavvio, tipica di Windows) |
| 5 | **Tono a 100 Hz** | Una cassa separata, **non** pilotata dall'app, fa da sorgente con un tono a 100 Hz (≈ 70–80 dB nella zona); mic d'errore nella zona; passi 5–7 della configurazione guidata | **≥ 10 dB** di attenuazione al mic d'errore a 100 Hz nel confronto A/B |
| 6 | **Stessa sequenza su Windows** | Ripeti 1–5 su Windows con la stessa interfaccia | Stessi criteri; se la causalità non basta, il punto 5 in modalità tonale |

Facoltativo dopo il punto 5: rumore rosa 50–300 Hz come sorgente. L'attenuazione attesa dipende dalla
causalità (vantaggio acustico in metri, vedi sopra) ed è normalmente inferiore a quella sul tono.

## Risoluzione problemi

| Sintomo | Cosa fare |
|---|---|
| Il microfono non viene concesso | Permesso di sistema (macOS: Privacy e sicurezza → Microfono; Windows: Privacy → Microfono) e del sito in Chrome. Serve https:// o http://localhost: un IP in http non basta |
| Nessun livello in ingresso | Ingresso giusto (riferimento sull'1), gain dell'interfaccia, phantom (48 V) per i microfoni a condensatore |
| Mic d'errore disattivato (canale 2 assente, muto o duplicato) | Scegli il dispositivo che espone gli ingressi 1–2 come coppia stereo; su Windows alcune interfacce mostrano ogni coppia come dispositivo separato. Canale 2 identico al canale 1 = il driver duplica un ingresso mono |
| Frequenze di campionamento diverse | 48 kHz ovunque (Configurazione MIDI Audio o Formato predefinito di Windows), poi riavvia Chrome |
| Latenza alta (> 40 ms) | Latenza Normale o Ultra-bassa nell'app, chiudi le altre app audio, niente Bluetooth; su Windows disattiva i miglioramenti audio e prova un'altra porta USB |
| Crepitii o interruzioni | Latenza Normale (128 campioni), chiudi le schede pesanti, collega l'alimentatore. Più di 3 interruzioni in 10 s silenziano l'uscita |
| Innesco o fischio | Abbassa il guadagno, orienta il mic lontano dalla cassa, mic direzionale, layout A; la neutralizzazione del feedback si applica automaticamente dopo la misura dei percorsi (lunghezza: Fase e filtri › Avanzate → Tap del neutralizzatore, 128–1024; con meno tap il guadagno d’anello sale e, oltre il margine di stabilità, i guadagni dei canali vengono ridotti da soli) |
| "L'ANC sta aumentando il rumore" | Qualcosa si è spostato o la temperatura è cambiata: ripeti misura dei percorsi e allineamento; controlla la polarità |
| Attenuazione scarsa | Problema di causalità → allontana il mic verso la sorgente; rumore periodico → modalità tonale; restringi la banda; parete in mezzo → compensazione barriera |
| Silenziamento automatico improvviso | Dispositivo cambiato o scollegato (cavo, hub USB, sospensione): ricontrolla i dispositivi e riattiva |
| Docker: porta 8080/8443 occupata | Cambia la parte sinistra delle porte in `docker-compose.yml` (es. `"9080:80"`) |
| Docker: errore su `COPY --chmod` | Serve BuildKit: Docker Desktop recente o `DOCKER_BUILDKIT=1` |
| Docker: il container si ferma all'avvio | `docker compose logs controfase`: lo script del certificato spiega il problema (`EXTRA_SAN` non valida, `server.crt`/`server.key` mancanti, non corrispondenti o con passphrase) |
| Avviso di certificato su https | Normale con il certificato autofirmato: verifica l'impronta SHA-256 nei log, oppure usa un tuo certificato (mkcert) in `./certs` |
| Microfono negato aprendo l'app da un altro PC | Usa `https://<IP>:8443` con `EXTRA_SAN=IP:<IP>`; `http://<IP>:8080` non è un contesto sicuro |

## Struttura del progetto

```
src/
  main.ts  styles.css
  app/        stato e parametri, regole di validazione dei dispositivi, preset (schema + limiti)
  audio/      protocol.ts (messaggi main ↔ worklet ↔ worker), engine (AudioContext, avvio a step,
              silenziamento automatico), devices (enumerazione, setSinkId, rilevamento Bluetooth/cuffie),
              calibration (orchestrazione delle misure), autotune
  worklet/    antiphase.worklet.ts: adattatore sottile del processore audio
  workers/    analysis.worker.ts: deconvoluzione degli sweep, cross-spettri, stima di f0
  dsp/        TypeScript puro, testabile in Node: dc-blocker, biquad, bessel-lp, fractional-delay,
              smoother, gain-ramp, limiter, howl-detector, feedback-canceller, resonator-bank,
              f0-tracker, hilbert-pair, meters, tap-buffer, fft, sweep, latency-estimator,
              response (H_dsp in forma chiusa), alignment (risolutore), physics, antiphase-engine
  ui/         configurazione guidata (passi 1–7), controlli di precisione, channel strip, pannelli
              dispositivi/filtri/preset, oscilloscopio, spettro, meter, barra di stato, avvisi, guida
tests/unit/   test Vitest per modulo          tests/sim/   impianto acustico simulato ad anello chiuso
e2e/          test Playwright (Chrome con microfono finto da WAV generati)
docs/         CONTRACTS.md: contratti tra i moduli
Dockerfile  docker-compose.yml  docker/ (nginx.conf, security-headers.conf, 05-selfsigned-cert.sh)
```

## Cosa è verificato automaticamente e cosa solo sull'hardware

**Verificato automaticamente** (`npm test`, `npm run test:e2e`, `npm run build`; i test unitari girano anche
nella build Docker):

- DSP: uscita = −x[n − D_min] esatta con condizionamento escluso; ritardi interi e frazionari; guadagno entro
  0,01 dB; limitatore (+12 dBFS in ingresso → mai oltre il tetto, nessun NaN/Inf); salti di parametro senza
  click; risultato indipendente dalla dimensione del blocco; convenzione di fase: con i parametri di default
  180° a f_ref e Δ = +30° → 210°, entro ± 0,1° in tutte e tre le modalità (ritardo, tonale, Hilbert), e il
  motore coincide con la risposta calcolata entro 0,5°; coppia di Hilbert 90° ± 0,75°; risposta H_dsp contro
  la catena reale.
- Sicurezza: rilevatore di innesco (non scatta su tono stabile con avvio graduale e rampe né su rumore a banda
  larga; scatta su innesco saturato dal limitatore); preset fuori range riportati ai limiti; regole di
  validazione dei dispositivi.
- Misure: stima del ritardo su sweep con rumore e riverbero (± 0,2 campioni, primo arrivo e segno corretti),
  risolutore, causalità e geometria.
- Modalità tonale: inseguimento automatico di f0 (un tono che deriva da 100 a 101 Hz in 10 s viene seguito
  entro 0,05 Hz a ogni stima, ogni 2 s); nessun inseguimento oltre il 3 %, con confidenza < 0,5, con il tono
  meno di 15 dB sopra il fondo, con l'inseguimento spento o fuori dalla modalità tonale.
- Simulatore ad anello chiuso con il vero motore DSP: calibrazione → risolutore → attenuazione attesa (anche
  con due casse sulla stessa zona), modalità tonale con f0 stimata su un tono stabile, neutralizzazione della
  retroazione, regolazione automatica con una cassa e con due casse sulla stessa zona (≥ 6 dB di
  miglioramento da un punto di partenza sbagliato, mai oltre il bypass + 1 dB, ≥ 10 dB di attenuazione alla
  fine).
- Innesco nel simulatore, con l'uscita mai oltre il tetto: con guadagno d'anello 1,5 e anelli di 3–100 ms,
  silenziamento entro 1 s dalla chiusura dell'anello; con guadagno d'anello 1,1–2 e anelli fino a 500 ms, entro
  0,4 s da quando il limitatore comincia a ridurre senza sosta. Nessun intervento in 10 s su segnali legittimi:
  rumore forte stazionario, gradino di +20 dB, tono di macchina che si accende, battimenti, avvio graduale,
  aumento di guadagno di 20 dB, impulsi al tetto, parlato, rumore che sale come un veicolo che passa.
- E2E in Chrome con microfono finto: contesto attivo, nessuna uscita prima della validazione, uscita =
  −g·riferimento ritardato (correlazione > 0,99), `Esc` → muto entro 50 ms, cambio dispositivo e traccia
  fermata → silenziamento automatico, nessun errore del processore, clic reali sui pulsanti.
- Build: `tsc -b` e `vite build`; nella build Docker anche lo script del certificato e `nginx -t`.

**Non ancora verificato automaticamente**:

- l'attenuazione nel simulatore **mentre f0 deriva** con l'inseguimento attivo: sono verificati la stima di
  f0, l'aggiornamento dei parametri da parte dell'app e la cancellazione su un tono stabile, non l'insieme.

**Solo sull'hardware reale** (vedi la [checklist](#checklist-di-prova-sullhardware)):

- la cancellazione acustica reale e l'attenuazione in dB nella stanza;
- le latenze reali dell'interfaccia, del sistema operativo e di Chrome;
- il comportamento su **Windows** (WASAPI, driver, miglioramenti audio);
- la retroazione e l'innesco in un ambiente reale;
- la disponibilità del canale 2 con una specifica interfaccia.

**Docker**: l'immagine non è stata eseguita nell'ambiente in cui è stata scritta (Docker non era installato).
La build verifica comunque la configurazione (`nginx -t` e lo script del certificato), ma va provata con
`docker compose up --build` e i `curl` della sezione [Docker](#docker).

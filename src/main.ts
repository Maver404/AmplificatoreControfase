/**
 * Entry point: creates the audio engine and the controller, mounts every view and binds it to the
 * controller. No logic lives here beyond translating controller state into view data; every action
 * goes through the controller (which owns safety: validation gate, presets, PANIC, keyboard).
 */
import './styles.css';
import { AudioEngine } from './audio/audio-engine.ts';
import { feedbackFirs, sanitizeCalibration } from './audio/calibration.ts';
import type { CalibrationData } from './audio/calibration.ts';
import { detectPlatform, isChromium, listDevices, parseDeviceKey } from './audio/devices.ts';
import type { DeviceOption } from './audio/devices.ts';
import { AppController, describeDevice, firstProblem } from './app/controller.ts';
import type { ControllerChange, ControllerState, DeviceProvider, MicPermission } from './app/controller.ts';
import { PRESETS_STORAGE_KEY } from './app/presets.ts';
import { ISSUE } from './app/validation.ts';
import { besselLowpassSections } from './dsp/bessel-lp.ts';
import { biquadResponse } from './dsp/biquad.ts';
import { cMul } from './dsp/complex.ts';
import type { Complex } from './dsp/complex.ts';
import { anyTrim } from './dsp/harmonic-trims.ts';
import { hilbertResponse } from './dsp/hilbert-pair.ts';
import { DEFAULT_PARAMS, RANGES, computeChannelDelay, dbToLin, phaseReferenceHz } from './dsp/params.ts';
import { headStartSeconds, speedOfSound } from './dsp/physics.ts';
import { groupDelaySamples } from './dsp/response.ts';
import { createChannelStrip } from './ui/channel-strip.ts';
import type { ChannelStrip } from './ui/channel-strip.ts';
import { createDevicePanel } from './ui/device-panel.ts';
import type { DeviceOption as PanelDeviceOption } from './ui/device-panel.ts';
import { createFiltersPanel } from './ui/filters-panel.ts';
import { createGuidePanel } from './ui/guide-panel.ts';
import { createMetersView } from './ui/meters-view.ts';
import { createPrecisionControl } from './ui/precision-control.ts';
import { createPresetsPanel } from './ui/presets-panel.ts';
import type { PresetSummary } from './ui/presets-panel.ts';
import { createScopeView, trimmedChannels } from './ui/scope-view.ts';
import { createSpectrumView } from './ui/spectrum-view.ts';
import { createStatusBar, startControl, statusOverride } from './ui/status-bar.ts';
import { formatHz, formatNumber } from './ui/units.ts';
import { createWarnings } from './ui/warnings.ts';
import type { WarningItem, WarningLevel } from './ui/warnings.ts';
import { createWizard } from './ui/wizard/wizard.ts';

const E2E = import.meta.env.MODE === 'e2e';
/** Room temperature for the cm delay unit (the wizard's geometry step measures the real one). */
const ROOM_TEMP_C = 20;

const byId = (id: string): HTMLElement => {
  const e = document.getElementById(id);
  if (!e) throw new Error(`Elemento #${id} mancante in index.html`);
  return e;
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------- engine and controller

async function microphonePermission(): Promise<MicPermission> {
  try {
    if (!navigator.permissions?.query) return 'unknown';
    const st = await navigator.permissions.query({ name: 'microphone' as PermissionName });
    return st.state === 'granted' ? 'granted' : st.state === 'denied' ? 'denied' : 'prompt';
  } catch {
    return 'unknown';
  }
}

const deviceProvider: DeviceProvider = {
  list: listDevices,
  platform: () => detectPlatform(),
  chromium: () => isChromium(),
  permission: microphonePermission,
};

const engine = new AudioEngine();
const controller = new AppController({
  engine,
  devices: deviceProvider,
  // Stored calibration (presets, wizard) → feedback-neutralisation FIRs (length from «Avanzate»).
  firsFromCalibration: (raw, taps) => {
    const cal = sanitizeCalibration(raw);
    return cal ? feedbackFirs(cal, taps) : [null, null];
  },
  // e2e only (fake microphone + the host's speakers, whose latency is often > 40 ms).
  forcedOverrides: E2E ? [ISSUE.outputLatency] : [],
});

// ---------------------------------------------------------------- views

const warnings = createWarnings(byId('warnings'));

const statusBar = createStatusBar(byId('status-bar'), {
  onStartStop: () => {
    const s = controller.state;
    if (s.armed && !s.panic) controller.disarm();
    // Same as «Attiva uscita» (muted while the guided setup holds the lock); the headphones
    // confirmation lives in the output panel.
    else if ((s.armed || s.canArm) && !needsHeadphonesConfirm(s)) controller.arm({ muted: s.ancLock !== null });
    else gate.focus();
  },
  onPanic: () => controller.panic(),
  onMuteToggle: () => controller.toggleMute(),
  onBypassToggle: () => controller.toggleBypass(),
  onMasterChange: (db) => controller.setMaster(db),
  onMessageDismiss: () => controller.dismissMessage(),
});

// Output gate: "Attiva uscita", headphones confirmation, ceiling and the +12 dB unlock.
function createOutputGate(container: HTMLElement): { update(s: ControllerState): void; focus(): void } {
  const root = el('div', 'output-gate');
  const status = el('p', 'gate-status');
  status.setAttribute('role', 'status');

  const confirm = el('label', 'gate-confirm');
  const confirmBox = el('input');
  confirmBox.type = 'checkbox';
  confirmBox.addEventListener('change', () => controller.confirmHeadphones(confirmBox.checked));
  confirm.append(
    confirmBox,
    el('span', undefined, 'Confermo: le cuffie non sono indossate e il volume è al minimo (limite −24 dBFS).'),
  );

  const actions = el('div', 'gate-actions');
  const armBtn = el('button', 'btn btn--primary gate-arm', 'Attiva uscita');
  armBtn.type = 'button';
  armBtn.addEventListener('click', () => {
    const s = controller.state;
    if (s.armed && !s.panic) controller.disarm();
    // While the guided setup holds the anti-noise lock, the output can only be enabled muted.
    else controller.arm({ headphonesConfirmed: confirmBox.checked, muted: s.ancLock !== null });
  });
  const hint = el('p', 'fp-help gate-hint');
  actions.append(armBtn, hint);

  const limits = el('div', 'gate-limits');
  const ceiling = createPrecisionControl(limits, {
    label: 'Limite d’uscita (ceiling del limiter)',
    value: DEFAULT_PARAMS.ceilingDb,
    min: RANGES.ceilingDb.min,
    max: RANGES.ceilingDb.max,
    step: 0.5,
    unit: 'dBFS',
    resetValue: DEFAULT_PARAMS.ceilingDb,
    resetLabel: 'valore predefinito',
    describe: () =>
      controller.state.params.headphones
        ? 'Cuffie: il limite è bloccato a −24 dBFS.'
        : 'Nessun campione d’uscita supera questo livello; gli aumenti salgono a 6 dB/s.',
    onChange: (v) => controller.setCeiling(v),
  });

  const advanced = el('details', 'advanced');
  advanced.append(el('summary', undefined, 'Avanzate'));
  const advBody = el('div', 'advanced-body');
  const unlock = el('button', 'switch');
  unlock.type = 'button';
  unlock.setAttribute('role', 'switch');
  const track = el('span', 'switch-track');
  track.setAttribute('aria-hidden', 'true');
  unlock.append(track, el('span', 'switch-text', 'Sblocca il guadagno dei canali fino a +12 dB'));
  unlock.addEventListener('click', () => controller.setGainUnlocked(!controller.state.params.gainUnlocked));
  advBody.append(
    unlock,
    el(
      'p',
      'fp-help',
      'Lo sblocco non viene mai salvato né importato con i preset. Se serve più livello, alza prima il guadagno d’ingresso ' +
        '(preamplificatore) del microfono di riferimento: il livello dell’amplificatore impostato al passo 2 è il margine di sicurezza.',
    ),
  );
  advanced.append(advBody);

  root.append(status, confirm, actions, limits, advanced);
  container.append(root);

  return {
    focus(): void {
      armBtn.scrollIntoView({ block: 'center', behavior: 'smooth' });
      armBtn.focus({ preventScroll: true });
    },
    update(s: ControllerState): void {
      const problem = firstProblem(s.validation);
      let text: string;
      let state: string;
      if (!s.micOpen) {
        text = s.opening
          ? 'Apertura del microfono in corso…'
          : 'Uscita bloccata: consenti il microfono e scegli i dispositivi. L’uscita resta a zero finché la validazione non è superata e non premi «Attiva uscita».';
        state = 'blocked';
      } else if (s.panic) {
        text = 'PANIC: uscita azzerata e motore sospeso. «Attiva uscita», Avvio o M ripartono con il soft-start.';
        state = 'panic';
      } else if (s.armed && s.ancLock !== null) {
        text = 'Uscita abilitata solo per i suoni di prova e di misura: l’anti-rumore si attiva alla fine della configurazione guidata.';
        state = 'muted';
      } else if (s.armed) {
        text = s.muted
          ? 'Uscita attiva ma muta: premi Mute (M) per riattivare l’anti-rumore con il soft-start.'
          : s.bypass
            ? 'Uscita attiva in bypass: l’anti-rumore è spento per il confronto A/B (B per riattivarlo).'
            : 'Uscita attiva. Tieni la mano sul volume dell’amplificatore; Esc = PANIC.';
        state = s.muted || s.bypass ? 'muted' : 'armed';
      } else if (!s.validation.ok) {
        text = `Uscita bloccata: ${problem ? problem.text : 'validazione non superata.'}`;
        state = 'blocked';
      } else {
        text =
          'Validazione superata. L’uscita è ancora a zero: abbassa l’amplificatore, poi premi «Attiva uscita» (soft-start di 1,5 s).';
        state = 'ready';
      }
      if (status.textContent !== text) status.textContent = text;
      root.dataset['state'] = state;

      const needConfirm = s.validation.headphones;
      confirm.hidden = !needConfirm;
      confirmBox.checked = s.headphonesConfirmed;

      const armedRunning = s.armed && !s.panic;
      armBtn.textContent = armedRunning ? 'Disattiva uscita' : 'Attiva uscita';
      armBtn.classList.toggle('btn--primary', !armedRunning);
      armBtn.disabled = !armedRunning && (!s.canArm || (needConfirm && !s.headphonesConfirmed));
      hint.textContent = armedRunning
        ? 'Disattivare riporta l’uscita a zeri esatti (serve di nuovo «Attiva uscita»).'
        : needConfirm && !s.headphonesConfirmed && s.canArm
          ? 'Spunta la conferma delle cuffie per continuare.'
          : `Master ${formatNumber(s.params.masterDb, 1)} dB · canali ${formatNumber(s.params.channels[0].gainDb, 1)} / ${formatNumber(s.params.channels[1].gainDb, 1)} dB · limite ${formatNumber(s.params.ceilingDb, 1)} dBFS`;

      ceiling.update(s.params.ceilingDb, { max: s.params.headphones ? RANGES.ceilingDb.min : RANGES.ceilingDb.max });
      unlock.setAttribute('aria-checked', String(s.params.gainUnlocked));
    },
  };
}
const gate = createOutputGate(byId('output-gate'));

// Guided setup: opens at start (not in the e2e build, whose tests drive the console directly).
createWizard(byId('wizard'), controller, { autoOpen: !E2E });

const devicePanel = createDevicePanel(byId('devices'), {
  // Synchronous call inside the click: the AudioContext is created before any await.
  onAllowMic: () => void controller.openMic(),
  onInputChange: (id) => void controller.setInput(id),
  onOutputChange: (id) => void controller.setOutput(id),
  onRefChannelChange: (ch) => controller.setRefChannel(ch),
  onErrorChannelChange: (ch) => controller.setErrorChannel(ch),
  onLatencyModeChange: (mode) => void controller.setLatencyMode(mode),
});

// Every strip edit, the harmonic trims included, goes through setChannel, which applies the L/R link.
const strips: [ChannelStrip, ChannelStrip] = [
  createChannelStrip(byId('strip-l'), {
    channel: 'L',
    initial: controller.state.params.channels[0],
    onChange: (next, changed) => controller.setChannel(0, next, changed),
    onClearTrims: () => controller.clearHarmonicTrims(0),
    onLinkChange: (v) => controller.setLinked(v),
  }),
  createChannelStrip(byId('strip-r'), {
    channel: 'R',
    initial: controller.state.params.channels[1],
    onChange: (next, changed) => controller.setChannel(1, next, changed),
    onClearTrims: () => controller.clearHarmonicTrims(1),
    onLinkChange: (v) => controller.setLinked(v),
  }),
];

const filters = createFiltersPanel(byId('filters'), {
  onPhaseModeChange: (mode) => controller.setPhaseMode(mode),
  onFRefChange: (hz) => controller.setFRef(hz),
  onTonalChange: (tonal) => controller.setTonal(tonal),
  onFiltersChange: (f) => controller.setFilters(f),
  onSnapToPeak: () => controller.snapF0ToPeak(),
  // Remembered by the controller, which rebuilds the current calibration's FIRs with the new length.
  onFeedbackTapsChange: (taps) => controller.setFeedbackTaps(taps),
});

const meters = createMetersView(byId('meters'));
const scope = createScopeView(byId('scope'), { fs: 48000, onChannelChange: () => pushAlignment() });
const spectrum = createSpectrumView(byId('spectrum'));
createGuidePanel(byId('guide'));

const presetsPanel = createPresetsPanel(byId('presets'), {
  onSave: (name) => controller.savePreset(name),
  onLoad: (id) => controller.loadPreset(id),
  onDelete: (id) => controller.deletePreset(id),
  onExport: (id) => {
    const out = controller.exportPresets(id);
    if (out) downloadText(out.filename, out.json);
  },
  onImport: (file) => {
    file.text().then(
      (text) => controller.importPresets(text),
      () => controller.importPresets(''),
    );
  },
});

function downloadText(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = el('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

controller.onTap((block, frames) => scope.pushTap(block, frames));

// ---------------------------------------------------------------- state → views

const panelOption = (d: DeviceOption): PanelDeviceOption => {
  const { note, blocked } = describeDevice(d);
  return note !== undefined ? { deviceId: d.deviceId, label: d.label, note, blocked } : { deviceId: d.deviceId, label: d.label, blocked };
};

/** The panel hides "Consenti microfono" when 'granted': it must stay visible until the mic is open. */
const panelPermission = (s: ControllerState): MicPermission =>
  s.micOpen ? 'granted' : s.permission === 'denied' ? 'denied' : 'prompt';

const fsOf = (s: ControllerState): number => s.fs ?? 48000;

const groupDelayMemo = new Map<string, number | null>();
function memo(key: string, f: () => number | null): number | null {
  if (!groupDelayMemo.has(key)) {
    if (groupDelayMemo.size > 64) groupDelayMemo.clear();
    groupDelayMemo.set(key, f());
  }
  return groupDelayMemo.get(key) ?? null;
}

/** Bessel LP group delay in the pass band (at a tenth of the cutoff), ms. */
function lpGroupDelayMs(s: ControllerState): number | null {
  const f = s.params.filters;
  const fs = fsOf(s);
  if (!f.lpEnabled) return null;
  return memo(`lp|${fs}|${f.lpFreq}|${f.lpOrder}`, () => {
    const sections = besselLowpassSections(fs, f.lpFreq, f.lpOrder);
    const h = (hz: number): Complex => sections.reduce<Complex>((acc, c) => cMul(acc, biquadResponse(c, hz, fs)), { re: 1, im: 0 });
    const v = (groupDelaySamples(h, f.lpFreq / 10, fs) / fs) * 1000;
    return Number.isFinite(v) ? v : null;
  });
}

function hilbertGroupDelayMs(s: ControllerState): number | null {
  const fs = fsOf(s);
  const fRef = s.params.fRef;
  return memo(`hb|${fs}|${fRef}`, () => {
    const v = (groupDelaySamples((hz) => hilbertResponse(hz, fs).a, fRef, fs) / fs) * 1000;
    return Number.isFinite(v) ? v : null;
  });
}

let calMemo: { raw: unknown; cal: CalibrationData | null } = { raw: null, cal: null };
function currentCalibration(): CalibrationData | null {
  const raw = controller.calibrationData;
  if (raw !== calMemo.raw) calMemo = { raw, cal: raw === null ? null : sanitizeCalibration(raw) };
  return calMemo.cal;
}

/**
 * Delay mismatch D + τ_S − τ_P (s) spanned by the tonal alignment of channel L, i.e. the whole
 * periods of f0 by which the anti-noise is late on the noise it cancels. With a calibration at this
 * rate: tap delay + measured speaker → zone path − mic → zone head start (geometry), rounded to whole
 * periods. Otherwise tap delay + browser latencies, an upper bound (τ_P ≥ 0 and the speaker → zone
 * air path left out).
 */
function driftAlign(s: ControllerState): { seconds: number; bound: boolean } | null {
  if (!s.micOpen) return null;
  const fs = fsOf(s);
  const tap = computeChannelDelay(s.params, fs, 0).samples / fs;
  const cal = currentCalibration();
  const sec = cal !== null && cal.fs === fs ? cal.secondary[0] : null;
  if (cal !== null && sec !== null && cal.geometry !== null) {
    const mismatch = tap + sec.delaySamples / fs - headStartSeconds(cal.geometry);
    const period = 1 / s.params.tonal.f0;
    return { seconds: Math.abs(Math.round(mismatch / period)) * period, bound: false };
  }
  const lat = [s.stats.baseLatencyMs, s.stats.outputLatencyMs, s.stats.trackLatencyMs].reduce<number>(
    (a, v) => a + (v !== null && Number.isFinite(v) ? v : 0),
    0,
  );
  return { seconds: tap + lat / 1000, bound: true };
}

const needsHeadphonesConfirm = (s: ControllerState): boolean => s.validation.headphones && !s.headphonesConfirmed;

const WARNING_ORDER: readonly WarningLevel[] = ['block', 'error', 'warn', 'info'];

function warningItems(s: ControllerState): WarningItem[] {
  const items: WarningItem[] = [];
  if (!s.micOpen) {
    if (s.micError) {
      items.push({
        id: 'mic-error',
        level: 'error',
        text: s.micError,
        action: s.opening ? undefined : { label: 'Riprova', onClick: () => void controller.openMic() },
      });
    }
    items.push({
      id: 'mic-closed',
      level: 'block',
      text: 'Microfono non ancora aperto: senza il microfono di riferimento l’uscita resta bloccata.',
      action: s.opening || s.micError ? undefined : { label: 'Consenti microfono', onClick: () => void controller.openMic() },
    });
  } else {
    if (s.micError) items.push({ id: 'mic-error', level: 'error', text: s.micError });
    for (const i of s.validation.issues) {
      const item: WarningItem = {
        id: `v-${i.id}`,
        level: i.overridden ? 'warn' : i.level,
        text: i.overridden ? `Ignorato su tua richiesta: ${i.text}` : i.text,
      };
      if (i.overridable) {
        item.action = i.overridden
          ? { label: 'Annulla', onClick: () => controller.setOverride(i.id, false) }
          : { label: 'Continua comunque', onClick: () => controller.setOverride(i.id, true) };
      }
      items.push(item);
    }
  }
  const f0 = s.f0;
  if (s.tonalSuggested && f0) {
    items.push({
      id: 'tonal-suggestion',
      level: 'info',
      text: `Tono dominante a ${formatHz(f0.f0)} (${formatNumber(f0.dominanceDb, 0)} dB sopra il fondo): la modalità Tonale può cancellarlo meglio.`,
      action: {
        label: 'Passa a Tonale',
        onClick: () => {
          const p = controller.state.params;
          controller.applyParams({ ...p, phaseMode: 'tonal', tonal: { ...p.tonal, f0: f0.f0 } });
        },
      },
    });
  }
  if (!s.params.filters.hpEnabled) {
    items.push({
      id: 'hp-off',
      level: 'warn',
      text: 'Passa-alto di protezione disattivato: vento e rombo captati dal microfono esterno arrivano all’amplificatore.',
    });
  }
  if (s.meters && (s.meters.delayClamped[0] || s.meters.delayClamped[1])) {
    items.push({
      id: 'delay-clamped',
      level: 'warn',
      text: 'Ritardo limitato al minimo del DSP su almeno un canale: l’allineamento richiesto non è causale.',
    });
  }
  if (!s.presets.persistent && s.presets.storageError) {
    items.push({ id: 'preset-storage', level: 'warn', text: s.presets.storageError });
  }
  if (s.bypass && s.armed) {
    items.push({ id: 'bypass', level: 'info', text: 'Bypass attivo: l’anti-rumore è spento per il confronto A/B (tasto B).' });
  }
  return items;
}

function presetSummaries(s: ControllerState): PresetSummary[] {
  return s.presets.list.map((p) => {
    const key = parseDeviceKey(p.deviceKey);
    const parts: string[] = [];
    if (key?.input?.label) parts.push(key.input.label);
    if (p.sampleRate > 0) parts.push(`${formatNumber(p.sampleRate / 1000, 1)} kHz`);
    if (p.calibration !== undefined) parts.push('con calibrazione');
    return { id: p.id, name: p.name, savedAt: p.createdAt || null, detail: parts.join(' · ') };
  });
}

function renderStatusBar(s: ControllerState): void {
  const items = warningItems(s);
  const stimulus = s.measuring || s.meters?.measuring === true;
  const problem = s.validation.ok ? null : (firstProblem(s.validation)?.text ?? 'validazione dei dispositivi non superata.');
  const start = startControl({ ...s, problem, headphonesPending: needsHeadphonesConfirm(s) });
  statusBar.update({
    status: s.status,
    // A stimulus playing (also between the controller's measure() calls, from the engine's meters).
    ...statusOverride({ ...s, measuring: stimulus }),
    running: s.armed && !s.panic,
    canStart: start.canStart,
    startHint: start.hint,
    warnings: { count: items.length, level: WARNING_ORDER.find((l) => items.some((i) => i.level === l)) ?? null },
    muted: s.muted,
    bypass: s.bypass,
    masterDb: s.params.masterDb,
    masterLocked: stimulus && s.armed && !s.panic,
    sampleRate: s.stats.sampleRate > 0 ? s.stats.sampleRate : null,
    renderQuantum: s.stats.renderQuantum > 0 ? s.stats.renderQuantum : null,
    baseLatencyMs: s.stats.baseLatencyMs,
    outputLatencyMs: s.stats.outputLatencyMs,
    trackLatencyMs: s.stats.trackLatencyMs,
    underruns: s.stats.underruns ?? 0,
    message: s.message,
  });
}

function renderStrips(s: ControllerState): void {
  const fs = fsOf(s);
  strips.forEach((strip, i) =>
    strip.update(s.params.channels[i as 0 | 1], s.micOpen ? s.meters : null, {
      fs,
      speedOfSound: speedOfSound(ROOM_TEMP_C),
      phaseRefHz: phaseReferenceHz(s.params),
      calibrated: s.calibratedChannels ? s.calibratedChannels[i as 0 | 1] : null,
      phaseMode: s.params.phaseMode,
      gainUnlocked: s.params.gainUnlocked,
      linked: s.linked,
      harmonics: s.params.tonal.harmonics,
      otherTrimmed: anyTrim(s.params.channels[i === 0 ? 1 : 0].harmonicTrims),
    }),
  );
}

function renderMeters(s: ControllerState): void {
  meters.update({
    ref: s.micOpen && s.meters ? s.meters.ref : null,
    err: s.meters ? s.meters.err : null,
    errAvailable: s.micOpen && s.params.errorChannel >= 0 && s.ch2 === 'ok',
    ancOn: s.ancOn,
    attenuationDb: s.attenuationDb,
    attenuationRefAgeS: s.attenuationRefAgeS,
    // The error mic can only be on the input the reference does not use.
    errorInput: s.params.refChannel === 1 ? 1 : 2,
  });
}

/** Expected-output alignment for the scope: total tap delay and gain from cond to the output. */
function pushAlignment(): void {
  const s = controller.state;
  const m = s.meters;
  for (const ch of [0, 1] as const) {
    const c = s.params.channels[ch];
    const silent = !s.armed || s.muted || s.bypass || s.panic || !c.enabled;
    const env = m ? m.softStart : 0;
    scope.setAlignment({
      channel: ch,
      delaySamples: m ? m.delaySamples[ch] : computeChannelDelay(s.params, fsOf(s), ch).samples,
      gainLin: silent ? 0 : dbToLin(c.gainDb) * dbToLin(s.params.masterDb) * env,
      polarityInvert: c.polarityInvert,
    });
  }
}

function renderState(s: ControllerState): void {
  renderStatusBar(s);
  gate.update(s);
  warnings.update(warningItems(s));
  devicePanel.update({
    permission: panelPermission(s),
    inputs: s.inputs.map(panelOption),
    outputs: s.outputs.map(panelOption),
    inputId: s.inputId,
    outputId: s.outputId,
    sinkIdSupported: s.sinkSupported,
    refChannel: s.params.refChannel,
    errorChannel: s.params.errorChannel,
    latencyMode: s.latencyMode,
    ch2: s.ch2,
    trackChannels: s.trackChannels,
    busy: s.opening,
  });
  renderStrips(s);
  filters.update({
    phaseMode: s.params.phaseMode,
    fRef: s.params.fRef,
    tonal: s.params.tonal,
    filters: s.params.filters,
    fs: fsOf(s),
    f0Estimate: s.f0,
    tonalSuggested: s.tonalSuggested,
    driftAlign: driftAlign(s),
    feedbackTaps: s.feedbackTaps,
    feedbackActive: s.feedbackFirActive[0] || s.feedbackFirActive[1],
    lpGroupDelayMs: lpGroupDelayMs(s),
    hilbertGroupDelayMs: hilbertGroupDelayMs(s),
  });
  renderMeters(s);
  scope.update({
    fs: fsOf(s),
    note:
      s.params.phaseMode === 'hilbert'
        ? 'In modalità Hilbert la traccia attesa non include la rotazione di fase: le due tracce non coincidono.'
        : '',
    trimmed: trimmedChannels(s.params),
  });
  pushAlignment();
  presetsPanel.update({
    presets: presetSummaries(s),
    activeId: s.presets.activeId,
    message: s.presets.message,
    disabled: s.measuring,
  });
  document.documentElement.dataset['ancState'] = s.status;
}

// Coalesce updates into one frame: meters arrive at ~30/s, taps faster.
const dirty: Record<ControllerChange, boolean> = { state: true, meters: false, spectrum: false, stats: false };
let raf = 0;
function flush(): void {
  raf = 0;
  const s = controller.state;
  try {
    if (dirty.state) {
      renderState(s);
    } else {
      if (dirty.meters) {
        renderStrips(s);
        renderMeters(s);
        pushAlignment();
      }
      if (dirty.meters || dirty.stats) renderStatusBar(s);
    }
    if (dirty.spectrum) spectrum.update(controller.spectra);
  } finally {
    dirty.state = dirty.meters = dirty.spectrum = dirty.stats = false;
  }
}
controller.subscribe((_s, change) => {
  dirty[change] = true;
  if (raf === 0) raf = window.requestAnimationFrame(flush);
});

controller.bindKeyboard(window);
window.addEventListener('storage', (e) => {
  if (e.key === PRESETS_STORAGE_KEY) controller.refreshPresets();
});

flush();

if (E2E) {
  void import('./app/e2e-hooks.ts').then((m) => m.installE2eHooks(controller, engine));
}

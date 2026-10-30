/**
 * What every wizard step receives from the shell (wizard.ts), and what it gives back.
 */
import type { AppController, ControllerState } from '../../app/controller.ts';
import type { Analyzer } from '../../audio/calibration.ts';
import type { FlowMode, StepId, WizardData } from './flow.ts';

/** One long operation (noise, sweeps, capture, autotune): navigation is disabled while it runs. */
export interface Op {
  readonly signal: AbortSignal;
  /** Ends the operation (no-op if it was aborted or replaced meanwhile). */
  end(): void;
}

export interface StepEnv {
  readonly controller: AppController;
  readonly data: WizardData;
  /** Analysis worker client (created on first use, terminated with the wizard). */
  analyzer(): Analyzer;
  /** Starts the operation; a running one is aborted first. */
  startOp(label: string): Op;
  /** Aborts the running operation and any measurement in the engine. */
  abortOp(): void;
  /** Label of the running operation, null when idle. */
  readonly busy: string | null;
  /** Milliseconds until the engine's rate-limited master has reached the last master set (0 = there). */
  masterReadyInMs(): number;
  /** Re-renders the step and the navigation (after a change of `data`). */
  refresh(): void;
  /** Remembers layout and geometry in this browser. */
  savePrefs(): void;
  goTo(step: StepId): void;
  /** Answer to the start question: rebuilds the step list. */
  setMode(mode: FlowMode): void;
  /** End of the guided setup (lock released, wizard closed). */
  complete(): void;
}

export interface StepView {
  readonly element: HTMLElement;
  /** Called on every controller change (coalesced to one per frame) and after refresh(). */
  update(s: ControllerState): void;
  /** Leaving the step: stop whatever it started. */
  leave(): void;
}

export type StepFactory = (env: StepEnv) => StepView;

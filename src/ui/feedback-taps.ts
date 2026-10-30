/**
 * Choices offered for the length of the feedback-neutralisation FIRs (plan: 256 taps, 128–1024
 * under «Avanzate»). The value itself is controller state (ControllerState.feedbackTaps, set with
 * setFeedbackTaps and remembered with the device choices): main.ts builds the FIRs with it and the
 * wizard passes it to the solvers' loop-gain check.
 */
export const FEEDBACK_TAP_CHOICES = [128, 256, 512, 1024] as const;

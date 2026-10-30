// Applies the saved theme before the first paint (same key as src/ui/status-bar.ts).
// A classic same-origin script: the Content-Security-Policy forbids inline scripts.
(function () {
  try {
    var t = localStorage.getItem('controfase.theme');
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  } catch (e) {
    // Storage blocked: the system theme applies.
  }
})();

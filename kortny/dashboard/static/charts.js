/* Shared ApexCharts helpers for the Kortny dashboard.
   Provides theme-aware palette/options/formatters and auto re-themes
   every registered chart when the light/dark toggle flips. */
(function () {
  const cssVar = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function palette() {
    return {
      dark: document.documentElement.dataset.theme === 'dark',
      accent: cssVar('--accent') || '#198567',
      accentStrong: cssVar('--accent-strong') || '#116c52',
      warning: cssVar('--warning') || '#926315',
      danger: cssVar('--destructive') || '#b42318',
      text: cssVar('--foreground') || '#15181a',
      muted: cssVar('--muted-foreground') || '#5f6861',
      border: cssVar('--border') || '#d8ded8',
      card: cssVar('--card') || '#ffffff',
    };
  }

  const fmt = {
    money: (v) =>
      '$' + Number(v || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
    moneyPrecise: (v) =>
      '$' + Number(v || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }),
    int: (v) => Number(v || 0).toLocaleString(),
    shortDate: (s) => {
      const d = new Date(s);
      return isNaN(d) ? s : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    },
  };

  // Accent-led multi-hue ramp for categorical charts (donuts / distributed bars).
  function categoricalColors(p) {
    return [p.accent, '#0e7490', '#7c3aed', p.warning, '#0891b2', '#be185d', '#15803d', p.danger, '#a16207', '#4f46e5'];
  }

  // Collapse a sorted (desc) label/value pair to the top N, summing the
  // remainder into a single "Others" entry. Keeps categorical charts readable
  // no matter how many users/models/channels exist.
  function topN(labels, values, n, otherLabel) {
    labels = labels || [];
    values = values || [];
    if (labels.length <= n) return { labels: labels.slice(), values: values.slice() };
    const headLabels = labels.slice(0, n);
    const headValues = values.slice(0, n);
    const rest = values.slice(n).reduce((a, b) => a + Number(b || 0), 0);
    if (rest > 0) {
      headLabels.push(otherLabel || 'Others');
      headValues.push(rest);
    }
    return { labels: headLabels, values: headValues };
  }

  function baseOptions(p) {
    return {
      chart: {
        fontFamily: 'Plus Jakarta Sans, ui-sans-serif, sans-serif',
        foreColor: p.muted,
        toolbar: { show: false },
        animations: { enabled: true, speed: 400 },
        background: 'transparent',
      },
      theme: { mode: p.dark ? 'dark' : 'light' },
      grid: { borderColor: p.border, strokeDashArray: 4, padding: { left: 8, right: 8 } },
      tooltip: { theme: p.dark ? 'dark' : 'light' },
      noData: { text: 'No data in this window', style: { color: p.muted, fontSize: '13px' } },
      dataLabels: { enabled: false },
      legend: { labels: { colors: p.muted } },
    };
  }

  const registry = [];

  // Render an ApexChart and register it for auto re-theming.
  function create(el, opts) {
    if (!el || typeof ApexCharts === 'undefined') return null;
    const chart = new ApexCharts(el, opts);
    chart.render();
    registry.push(chart);
    return chart;
  }

  // Re-theme all registered charts when data-theme changes.
  const observer = new MutationObserver(() => {
    const p = palette();
    registry.forEach((chart) => {
      chart.updateOptions(
        {
          theme: { mode: p.dark ? 'dark' : 'light' },
          chart: { foreColor: p.muted },
          grid: { borderColor: p.border },
          tooltip: { theme: p.dark ? 'dark' : 'light' },
        },
        false,
        false
      );
    });
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  window.KortnyCharts = { palette, fmt, categoricalColors, topN, baseOptions, create };
})();

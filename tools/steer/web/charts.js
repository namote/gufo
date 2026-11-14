// Inline SVG charts for the steering lab: bars, lines and a heat map.
// No library; colors come from CSS classes (.s1 .. .s4) so both themes apply.
'use strict';
(function () {
  const NS = 'http://www.w3.org/2000/svg';

  function svg(tag, attrs, ...children) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v !== undefined && v !== null) el.setAttribute(k, String(v));
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
    return el;
  }

  function niceTicks(lo, hi, count) {
    if (!(isFinite(lo) && isFinite(hi))) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    const raw = (hi - lo) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
    }
    return out;
  }

  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '-';
    if (typeof v !== 'number') return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
    if (Number.isInteger(v)) return String(v);
    return a >= 100 ? v.toFixed(1) : a >= 10 ? v.toFixed(2) : v.toFixed(3);
  }

  function legend(series) {
    if (series.length < 2) return null;
    const box = document.createElement('div');
    box.className = 'legend';
    for (const s of series) {
      const item = document.createElement('span');
      const key = document.createElement('i');
      key.className = `key ${s.cls}${s.dashed ? ' dashed' : ''}`;
      item.append(key, document.createTextNode(s.name));
      box.append(item);
    }
    return box;
  }

  // Shared frame: returns {root, plot, x(v), y(v), width, height}.
  function frame(opts, xDomain, yDomain, band) {
    const width = opts.width || 720;
    const height = opts.height || 220;
    const m = { l: 48, r: 12, t: opts.yLabel ? 22 : 10, b: 30 };
    const W = width - m.l - m.r;
    const H = height - m.t - m.b;
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img',
      'aria-label': opts.label || '' });
    const yt = niceTicks(yDomain[0], yDomain[1], opts.yTicks || 4);
    const y0 = Math.min(yDomain[0], yt[0]);
    const y1 = Math.max(yDomain[1], yt[yt.length - 1]);
    const y = (v) => m.t + H - ((v - y0) / (y1 - y0 || 1)) * H;
    let x;
    if (band) {
      const n = xDomain.length || 1;
      const bw = W / n;
      x = (i) => m.l + i * bw;
      x.band = bw;
    } else {
      const [a, b] = xDomain;
      x = (v) => m.l + ((v - a) / (b - a || 1)) * W;
    }
    const grid = svg('g', { class: 'grid' });
    for (const t of yt) {
      grid.append(svg('line', { x1: m.l, x2: m.l + W, y1: y(t), y2: y(t) }));
      grid.append(svg('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end', class: 'tick' }, fmt(t)));
    }
    root.append(grid);
    if (opts.ref !== undefined) {
      root.append(svg('line', { x1: m.l, x2: m.l + W, y1: y(opts.ref), y2: y(opts.ref), class: 'ref' }));
    }
    if (opts.yLabel) {
      root.append(svg('text', { x: 4, y: m.t - 8, class: 'axis-label' }, opts.yLabel));
    }
    if (opts.xLabel) {
      root.append(svg('text', { x: m.l + W, y: height - 4, 'text-anchor': 'end', class: 'axis-label' },
        opts.xLabel));
    }
    return { root, x, y, m, W, H, width, height, y0, y1 };
  }

  function extent(values, include) {
    let lo = Infinity, hi = -Infinity;
    for (const v of values) {
      if (typeof v === 'number' && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    for (const v of include || []) { if (v !== undefined) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    if (lo === Infinity) return [0, 1];
    return [lo, hi];
  }

  // Grouped bars. opts: {categories: [label], series: [{name, cls, values: [num|null]}],
  // highlight: Set(category index), yMin, yMax, ref, yLabel, xLabel, onClick(i)}
  function bars(container, opts) {
    const cats = opts.categories;
    const all = opts.series.flatMap((s) => s.values);
    const dom = extent(all, [opts.yMin ?? 0, opts.yMax]);
    const f = frame(opts, cats, dom, true);
    const k = opts.series.length || 1;
    const inner = f.x.band * 0.8;
    const bw = Math.max(1, inner / k - (k > 1 ? 0.5 : 0));
    const base = f.y(Math.max(f.y0, Math.min(0, f.y1)));
    const g = svg('g');
    cats.forEach((cat, i) => {
      if (opts.highlight && opts.highlight.has(i)) {
        g.append(svg('rect', { x: f.x(i), y: f.m.t, width: f.x.band, height: f.H, class: 'hl' }));
      }
      opts.series.forEach((s, j) => {
        const v = s.values[i];
        if (typeof v !== 'number' || !isFinite(v)) return;
        const top = f.y(v);
        const r = svg('rect', { x: f.x(i) + f.x.band * 0.1 + j * (inner / k), y: Math.min(top, base),
          width: bw, height: Math.max(0.5, Math.abs(base - top)), class: `bar ${s.cls}` },
        svg('title', {}, `${cat}: ${s.name} ${fmt(v)}`));
        if (opts.onClick) { r.addEventListener('click', () => opts.onClick(i)); r.classList.add('clickable'); }
        g.append(r);
      });
    });
    f.root.append(g);
    const every = Math.max(1, Math.ceil(cats.length / Math.max(1, f.W / 28)));
    cats.forEach((cat, i) => {
      if (i % every) return;
      f.root.append(svg('text', { x: f.x(i) + f.x.band / 2, y: f.m.t + f.H + 14, 'text-anchor': 'middle',
        class: 'tick' }, String(cat)));
    });
    container.replaceChildren(f.root);
    const l = legend(opts.series);
    if (l) container.append(l);
  }

  // Lines. opts: {series: [{name, cls, points: [[x, y]], dots, dashed}], xLabel, yLabel, ref}
  function lines(container, opts) {
    const xs = opts.series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = opts.series.flatMap((s) => s.points.map((p) => p[1]));
    if (!xs.length) { container.replaceChildren(empty(opts.empty || 'waiting for data')); return; }
    const xd = extent(xs);
    if (xd[0] === xd[1]) xd[1] = xd[0] + 1;
    const f = frame(opts, xd, extent(ys, opts.ref !== undefined ? [opts.ref] : []), false);
    for (const t of niceTicks(xd[0], xd[1], 6)) {
      if (t < xd[0] || t > xd[1]) continue;
      f.root.append(svg('text', { x: f.x(t), y: f.m.t + f.H + 14, 'text-anchor': 'middle', class: 'tick' }, fmt(t)));
    }
    for (const s of opts.series) {
      const pts = s.points.filter((p) => typeof p[1] === 'number' && isFinite(p[1]));
      if (!pts.length) continue;
      if (pts.length > 1 && !s.dotsOnly) {
        const d = pts.map((p, i) => `${i ? 'L' : 'M'}${f.x(p[0]).toFixed(1)},${f.y(p[1]).toFixed(1)}`).join('');
        f.root.append(svg('path', { d, class: `line ${s.cls}${s.dashed ? ' dashed' : ''}` }));
      }
      if (s.dots || s.dotsOnly || pts.length === 1) {
        for (const p of pts) {
          f.root.append(svg('circle', { cx: f.x(p[0]), cy: f.y(p[1]), r: 2.5, class: `dot ${s.cls}` },
            svg('title', {}, `${s.name} @ ${fmt(p[0])}: ${fmt(p[1])}`)));
        }
      }
    }
    container.replaceChildren(f.root);
    const l = legend(opts.series);
    if (l) container.append(l);
  }

  // Heat map. opts: {matrix: [[v]], rows: [label], cols: [label], min, max, diverging, title}
  function heatmap(container, opts) {
    const rows = opts.matrix.length;
    const cols = rows ? opts.matrix[0].length : 0;
    if (!rows || !cols) { container.replaceChildren(empty('waiting for data')); return; }
    const cell = Math.max(3, Math.min(14, Math.floor(640 / Math.max(rows, cols))));
    const m = { l: 34, t: 8, r: 8, b: 26 };
    const width = m.l + cols * cell + m.r;
    const height = m.t + rows * cell + m.b;
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart heat', role: 'img',
      'aria-label': opts.label || 'heat map' });
    const lo = opts.min ?? -1, hi = opts.max ?? 1;
    const color = (v) => {
      if (typeof v !== 'number' || !isFinite(v)) return 'transparent';
      if (opts.diverging) {
        const t = Math.max(-1, Math.min(1, (2 * (v - lo)) / (hi - lo || 1) - 1));
        const pct = Math.round(Math.abs(t) * 100);
        return `color-mix(in oklab, var(${t >= 0 ? '--pos' : '--neg'}) ${pct}%, var(--heat-mid))`;
      }
      const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1)));
      return `color-mix(in oklab, var(--pos) ${Math.round(t * 100)}%, var(--heat-mid))`;
    };
    const rl = opts.rows || [...Array(rows).keys()];
    const cl = opts.cols || [...Array(cols).keys()];
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = opts.matrix[i][j];
        const r = svg('rect', { x: m.l + j * cell, y: m.t + i * cell, width: cell, height: cell, class: 'cell' },
          svg('title', {}, `${opts.rowName || 'row'} ${rl[i]} x ${opts.colName || 'col'} ${cl[j]}: ${fmt(v)}`));
        r.style.fill = color(v);
        root.append(r);
      }
    }
    const every = Math.max(1, Math.ceil(12 / cell));
    for (let i = 0; i < rows; i += every) {
      root.append(svg('text', { x: m.l - 4, y: m.t + i * cell + cell * 0.8, 'text-anchor': 'end', class: 'tick' },
        String(rl[i])));
    }
    for (let j = 0; j < cols; j += Math.max(every, 2)) {
      root.append(svg('text', { x: m.l + j * cell + cell / 2, y: m.t + rows * cell + 12, 'text-anchor': 'middle',
        class: 'tick' }, String(cl[j])));
    }
    container.replaceChildren(root);
    const scale = document.createElement('div');
    scale.className = 'legend';
    const bar = document.createElement('i');
    bar.className = opts.diverging ? 'ramp diverging' : 'ramp';
    scale.append(document.createTextNode(fmt(lo) + ' '), bar, document.createTextNode(' ' + fmt(hi)));
    container.append(scale);
  }

  function empty(text) {
    const d = document.createElement('div');
    d.className = 'empty';
    d.textContent = text;
    return d;
  }

  window.Charts = { bars, lines, heatmap, fmt, empty };
})();

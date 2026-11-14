// Steering lab: one page over the steer tool's job API (tools/steer/README.md).
// Plain JS, no build step, no network beyond this service.
'use strict';
(function () {
  const C = window.Charts;
  const fmt = C.fmt;

  // Endpoints. The optional ones may be missing on an older service; the page
  // then says so and shows what it has.
  const EP = {
    health: '/health',
    jobs: '/jobs',
    job: (key, since) => `/jobs/${enc(key)}?since=${since || 0}`,
    datasets: '/datasets',
    dataset: (name) => `/datasets/${enc(name)}`,
    records: (name, offset, limit) => `/datasets/${enc(name)}/records?offset=${offset}&limit=${limit}`,
    generations: (key) => `/jobs/${enc(key)}/generations`,
    rows: (key, config, offset, limit) =>
      `/jobs/${enc(key)}/generations/${enc(config)}?offset=${offset}&limit=${limit}`,
    profileFiles: '/profiles',
    props: '/gufo/props',
    profile: (kind, name) => `/profiles/${enc(kind)}/${enc(name)}`,
  };
  const FINISHED = ['done', 'failed', 'cancelled'];
  const POLL_MS = 1500;

  function enc(s) { return encodeURIComponent(String(s)); }

  // append/replaceChildren take nested lists and skip null, undefined and false,
  // as h() does, so optional parts can be written inline.
  for (const m of ['append', 'replaceChildren']) {
    const orig = Element.prototype[m];
    Element.prototype[m] = function (...c) {
      return orig.apply(this, c.flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false));
    };
  }

  // DOM ---------------------------------------------------------------------
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'style') Object.assign(el.style, v);
      else if (k.startsWith('on')) { if (typeof v === 'function') el.addEventListener(k.slice(2), v); }
      else if (k === 'value') el.value = v;
      else if (k === 'checked') el.checked = !!v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  const $ = (sel, root) => (root || document).querySelector(sel);

  let toastTimer = null;
  function toast(message, error) {
    const t = $('#toast');
    t.textContent = message;
    t.className = 'toast' + (error ? ' error' : '');
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, error ? 9000 : 3500);
  }

  function time(t) {
    if (!t) return '-';
    const d = new Date(t * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }
  function duration(a, b) {
    if (!a) return '-';
    const s = Math.max(0, (b || Date.now() / 1000) - a);
    if (s < 90) return `${s.toFixed(0)} s`;
    if (s < 5400) return `${(s / 60).toFixed(1)} min`;
    return `${(s / 3600).toFixed(2)} h`;
  }
  function json(v) { return JSON.stringify(v, null, 1); }
  function short(key) { return String(key).slice(0, 8); }
  function copyButton(text) {
    return h('button', { class: 'ghost small', type: 'button', onclick: () => {
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject(new Error('no clipboard')))
        .then(() => toast('copied'), () => toast('select and copy the text', true));
    } }, 'Copy');
  }

  // API ---------------------------------------------------------------------
  class ApiError extends Error {
    constructor(status, code, message) { super(message); this.status = status; this.code = code; }
  }
  async function api(method, path, body) {
    const init = { method, headers: {} };
    if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
    let response;
    try { response = await fetch(path, init); } catch (e) { throw new ApiError(0, 'unreachable', 'the steer service does not answer'); }
    let reply = null;
    try { reply = await response.json(); } catch (e) { reply = null; }
    if (!response.ok) {
      const err = (reply && reply.error) || {};
      throw new ApiError(response.status, err.code || 'http_' + response.status, err.message || response.statusText);
    }
    return reply;
  }
  // GET an optional endpoint: null when this service does not have it.
  const missing = new Set();
  async function optional(path) {
    try { return await api('GET', path); } catch (e) {
      if (e.status === 404 || e.status === 405) { missing.add(path.split('?')[0]); return null; }
      throw e;
    }
  }
  function listOf(reply) {
    if (!reply) return [];
    if (Array.isArray(reply)) return reply;
    return reply.data || reply.records || reply.rows || reply.items || [];
  }
  function fail(e) { toast(`${e.code || 'error'}: ${e.message}`, true); }

  async function startJob(operation, params) {
    const reply = await api('POST', EP.jobs, { operation, params });
    toast(`${operation} job ${short(reply.job)} ${reply.status}`);
    return reply.job;
  }

  // Polls one job with ?since=N; keeps every result.
  class Watch {
    constructor(key, onChange) {
      this.key = key; this.onChange = onChange; this.state = null; this.results = []; this.since = 0;
      this.timer = null; this.stopped = false; this.error = null;
    }
    async tick() {
      if (this.stopped) return;
      if (this.root && !this.root.isConnected) { this.stop(); return; }  // its view was left
      try {
        const s = await api('GET', EP.job(this.key, this.since));
        const fresh = s.results || [];
        for (const r of fresh) this.results.push(r);
        if (fresh.length) this.since = fresh[fresh.length - 1].seq;
        delete s.results;
        this.state = s; this.error = null;
        this.onChange(this, fresh);
        if (FINISHED.includes(s.status)) { this.stopped = true; return; }
      } catch (e) {
        this.error = e;
        this.onChange(this, []);
        if (e.status === 404) { this.stopped = true; return; }
      }
      if (!this.stopped) this.timer = setTimeout(() => this.tick(), POLL_MS);
    }
    start() { this.tick(); return this; }
    stop() { this.stopped = true; clearTimeout(this.timer); }
    of(type) { return this.results.filter((r) => r.type === type); }
  }
  // Every poller of the current view; a route change stops them.
  let watches = [];
  function watchIn(root, key, onChange) {
    const w = new Watch(key, onChange);
    w.root = root;
    if (root.isConnected) watches.push(w); else w.stopped = true;
    return w.start();
  }
  async function loadAll(key) {
    const s = await api('GET', EP.job(key, 0));
    return { state: s, results: s.results || [], of(type) { return this.results.filter((r) => r.type === type); } };
  }

  // Forms -------------------------------------------------------------------
  // spec: [{key, label, type: text|number|int|check|select|textarea|json|list|intlist, value, options, wide, hint}]
  function form(spec, onInput) {
    const inputs = {};
    const box = h('div', { class: 'fields' });
    for (const f of spec) {
      let input;
      const type = f.type || 'text';
      if (type === 'check') {
        input = h('input', { type: 'checkbox', checked: f.value });
        box.append(h('label', { class: 'check' + (f.wide ? ' wide' : ''), title: f.hint }, input, f.label));
      } else {
        if (type === 'select') {
          input = h('select', {}, f.options.map((o) => {
            const [v, l] = Array.isArray(o) ? o : [o, o === '' ? '(default)' : o];
            return h('option', { value: v }, l);
          }));
          if (f.value !== undefined) input.value = f.value;
        } else if (type === 'textarea' || type === 'json') {
          input = h('textarea', { placeholder: f.placeholder || '', spellcheck: 'false' });
          input.value = f.value === undefined ? '' : (type === 'json' && typeof f.value !== 'string' ? json(f.value) : f.value);
        } else {
          input = h('input', { type: type === 'number' || type === 'int' ? 'number' : 'text',
            step: type === 'number' ? 'any' : type === 'int' ? '1' : undefined, placeholder: f.placeholder || '',
            list: f.list });
          if (f.value !== undefined) input.value = f.value;
        }
        box.append(h('label', { class: f.wide || type === 'textarea' || type === 'json' ? 'wide' : '', title: f.hint },
          f.label, input));
      }
      inputs[f.key] = { input, f };
      input.addEventListener('input', () => onInput && onInput());
      input.addEventListener('change', () => onInput && onInput());
    }
    function values() {
      const out = {};
      for (const [k, { input, f }] of Object.entries(inputs)) {
        const type = f.type || 'text';
        input.classList.remove('invalid');
        if (type === 'check') { if (input.checked !== !!f.omitWhen) out[k] = input.checked; continue; }
        const raw = input.value.trim();
        if (raw === '') continue;
        if (type === 'number' || type === 'int') {
          const n = Number(raw);
          if (!isFinite(n)) { input.classList.add('invalid'); throw new Error(`${f.label}: not a number`); }
          out[k] = type === 'int' ? Math.trunc(n) : n;
        } else if (type === 'json') {
          try { out[k] = JSON.parse(raw); } catch (e) { input.classList.add('invalid'); throw new Error(`${f.label}: ${e.message}`); }
        } else if (type === 'list') {
          out[k] = raw.split(',').map((s) => s.trim()).filter(Boolean);
        } else if (type === 'intlist') {
          out[k] = raw.split(/[\s,]+/).filter(Boolean).map(Number);
          if (out[k].some((n) => !Number.isInteger(n))) { input.classList.add('invalid'); throw new Error(`${f.label}: integers`); }
        } else out[k] = raw;
      }
      return out;
    }
    function set(k, v) {
      const it = inputs[k];
      if (!it) return;
      if ((it.f.type || 'text') === 'check') it.input.checked = !!v;
      else it.input.value = v === undefined || v === null ? '' : (typeof v === 'object' ? json(v) : String(v));
    }
    return { el: box, values, set, inputs };
  }

  // A params preview with an extra-JSON box merged over the form's values.
  function starter(operation, build, opts) {
    const extra = h('textarea', { placeholder: '{"extra": "params merged over the form"}', spellcheck: 'false' });
    const preview = h('pre', {});
    const box = h('div', {},
      h('details', {}, h('summary', {}, 'Params preview and extra JSON'),
        h('label', { class: 'field' }, 'Extra params (JSON object, merged over the form)', extra), preview));
    function params() {
      const p = build();
      const raw = extra.value.trim();
      extra.classList.remove('invalid');
      if (raw) {
        let e;
        try { e = JSON.parse(raw); } catch (err) { extra.classList.add('invalid'); throw new Error('extra params: ' + err.message); }
        Object.assign(p, e);
      }
      return p;
    }
    function refresh() {
      try { preview.textContent = json({ operation, params: params() }); } catch (e) { preview.textContent = e.message; }
    }
    extra.addEventListener('input', refresh);
    const button = h('button', { type: 'button', onclick: async () => {
      let p;
      try { p = params(); } catch (e) { toast(e.message, true); return; }
      if (opts && opts.confirm && !window.confirm(opts.confirm(p))) return;
      button.disabled = true;
      try {
        const key = await startJob(operation, p);
        if (opts && opts.started) opts.started(key, p); else go(`#job/${key}`);
      } catch (e) { fail(e); } finally { button.disabled = false; }
    } }, (opts && opts.label) || `Start ${operation}`);
    return { el: box, button, refresh, params, extra };
  }

  // Header ------------------------------------------------------------------
  let gufoInfo = null;
  let uploadRefused = false;
  function noteUploadRefused(e) {
    if (e && (e.code === 'unsupported_profile_update' || (e.status === 404 && e.code === 'not_found' && e.message !== 'unknown path'))) {
      uploadRefused = true; banner();
    }
  }
  function banner() {
    const b = $('#banner');
    const lines = [];
    const url = gufoInfo && gufoInfo.base_url;
    let port = null;
    try { if (url) port = new URL(url).port || (new URL(url).protocol === 'https:' ? '443' : '80'); } catch (e) { port = null; }
    if (port === '18740') {
      lines.push('The gufo URL looks like the production server (port 18740). Experiments belong on an experiment server ' +
        'started with --collect-dir and --allow-profile-updates; without the latter, profile uploads are refused.');
    }
    if (uploadRefused) lines.push('gufo refused a profile upload: the server runs without --allow-profile-updates.');
    if (gufoInfo === false) lines.push('The steer service runs without --gufo-url: fit, test, train and uploads need it.');
    b.hidden = !lines.length;
    b.textContent = lines.join(' ');
  }
  let loadedProfiles = { abliteration: [], 'logit-bias': [] };
  async function header() {
    const hp = $('#health'), gp = $('#gufo'), pp = $('#profiles');
    try {
      const s = await api('GET', EP.health);
      hp.textContent = `service: ${s.status}`; hp.className = 'pill ok';
      gufoInfo = s.gufo || false;
      gp.textContent = gufoInfo ? `gufo: ${gufoInfo.base_url}${gufoInfo.api_key ? ' (key)' : ''}` : 'gufo: none';
      gp.className = 'pill' + (gufoInfo ? '' : ' bad');
      if (s.gufo_health !== undefined || s.gufo_status !== undefined) {
        const g = s.gufo_health ?? s.gufo_status;
        gp.title = typeof g === 'string' ? g : json(g);
      }
    } catch (e) {
      hp.textContent = 'service: down'; hp.className = 'pill bad';
    }
    banner();
    if (!gufoInfo) { pp.textContent = 'profiles: -'; return; }
    try {
      const props = await optional(EP.props);
      if (props === null) { pp.textContent = 'profiles: (no /props passthrough)'; pp.title = 'the service has no GET /props yet'; return; }
      const p = props.props || props;
      const ab = ((p.abliteration && p.abliteration.profiles) || []).map((x) => x.profile || x.name).filter(Boolean);
      const lbSrc = p.logit_bias || p['logit-bias'] || p.logit_bias_profiles || {};
      const lb = (Array.isArray(lbSrc) ? lbSrc : lbSrc.profiles || []).map((x) => (typeof x === 'string' ? x : x.profile || x.name)).filter(Boolean);
      loadedProfiles = { abliteration: ab, 'logit-bias': lb, raw: p };
      pp.textContent = `profiles: ${ab.length ? ab.join(', ') : 'none'}${lb.length ? ' | bias: ' + lb.join(', ') : ''}`;
      pp.title = 'gufo /props (through the tool)';
      pp.className = 'pill';
      fillDatalists();
    } catch (e) { pp.textContent = `profiles: gufo /props failed (${e.code})`; pp.title = e.message; pp.className = 'pill bad'; }
  }
  function fillDatalists() {
    let dl = $('#dl-profiles');
    if (!dl) { dl = h('datalist', { id: 'dl-profiles' }); document.body.append(dl); }
    dl.replaceChildren(...loadedProfiles.abliteration.map((n) => h('option', { value: n })));
    let dlb = $('#dl-bias');
    if (!dlb) { dlb = h('datalist', { id: 'dl-bias' }); document.body.append(dlb); }
    dlb.replaceChildren(...loadedProfiles['logit-bias'].map((n) => h('option', { value: n })));
  }

  // Job panel (every view) ----------------------------------------------------
  // Rebuild `box` with fn(), keeping which <details> were open (by data-k, else summary label).
  function keepOpen(box, fn) {
    const k = (d) => d.dataset.k || ((d.querySelector('summary') || {}).textContent || '').split(':')[0];
    const open = new Set([...box.querySelectorAll('details')].filter((d) => d.open).map(k));
    fn();
    for (const d of box.querySelectorAll('details')) if (open.has(k(d))) d.open = true;
  }
  function statusSpan(s) { return h('span', { class: `status-${s}` }, s); }

  function jobPanel(w, opts) {
    const box = h('div', { class: 'card' });
    function render() {
      const s = w.state;
      if (!s) { box.replaceChildren(w.error ? h('div', { class: 'bad' }, `${w.error.code}: ${w.error.message}`) : C.empty('loading job ' + short(w.key))); return; }
      const p = s.progress || {};
      const pct = p.total ? Math.min(100, (100 * p.done) / p.total) : (s.status === 'done' ? 100 : 0);
      const bar = h('div', { class: 'progress' }, h('i', { style: { width: pct + '%' } }));
      const actions = h('div', { class: 'row' });
      if (!FINISHED.includes(s.status)) {
        actions.append(h('button', { class: 'danger small', type: 'button', onclick: async () => {
          await cancelOrDelete(w.key, s.operation);
        } }, 'Cancel'));
      }
      actions.append(h('button', { class: 'ghost small', type: 'button', onclick: () => rerun(s) }, 'Re-run with the same params'));
      actions.append(h('a', { href: `#history/${w.key}` }, 'in history'));
      keepOpen(box, () => box.replaceChildren(
        h('div', { class: 'row' }, h('strong', {}, `${s.operation} `), h('code', {}, w.key), statusSpan(s.status),
          s.cancel_requested && s.status === 'running' ? h('span', { class: 'muted' }, '(cancelling)') : null,
          h('span', { class: 'spacer' }), actions),
        bar,
        h('div', { class: 'row muted' },
          h('span', {}, `${p.done || 0}/${p.total || 0}`), p.stage ? h('span', {}, `stage ${p.stage}`) : null,
          p.step ? h('span', {}, p.step) : null, h('span', { class: 'spacer' }),
          h('span', {}, `created ${time(s.created_at)}`),
          h('span', {}, `ran ${duration(s.started_at, FINISHED.includes(s.status) ? s.updated_at : null)}`)),
        s.error ? h('pre', { class: 'bad' }, s.error) : null,
        w.error ? h('div', { class: 'bad' }, `poll: ${w.error.code}: ${w.error.message}`) : null,
        h('details', {}, h('summary', {}, 'Params'), h('pre', {}, json(s.params))),
        s.output ? h('details', {}, h('summary', {}, 'Output'), h('pre', {}, json(s.output))) : null,
        opts && opts.raw === false ? null : rawResults(w.results, opts && opts.known)));
    }
    render();
    return { el: box, render };
  }
  function rawResults(results, known) {
    const other = known ? results.filter((r) => !known.includes(r.type)) : results;
    const counts = {};
    for (const r of results) counts[r.type] = (counts[r.type] || 0) + 1;
    const summary = Object.entries(counts).map(([t, n]) => `${t} ${n}`).join(', ') || 'none yet';
    const d = h('details', {}, h('summary', {}, `Results: ${summary}${known && other.length ? ` (${other.length} shown raw)` : ''}`));
    d.dataset.k = 'results';
    d.addEventListener('toggle', () => {
      if (d.open && d.children.length === 1) d.append(h('pre', {}, (known ? other : results).slice(-200).map((r) => JSON.stringify(r)).join('\n')));
    });
    return d;
  }
  // DELETE cancels an unfinished job but deletes a finished one: decide on the
  // job's current state, never on a list that may be stale.
  async function cancelOrDelete(key, operation) {
    let state;
    try { state = await api('GET', EP.job(key, Number.MAX_SAFE_INTEGER)); } catch (e) { fail(e); return false; }
    const active = !FINISHED.includes(state.status);
    if (!window.confirm(active ? `Cancel ${operation} job ${short(key)} (it is ${state.status})?`
      : `${operation} job ${short(key)} is ${state.status}. Delete it and its folder?`)) return false;
    if (active) {  // it may have finished while the dialog was open: a DELETE would then remove it
      try { state = await api('GET', EP.job(key, Number.MAX_SAFE_INTEGER)); } catch (e) { fail(e); return false; }
      if (FINISHED.includes(state.status)) { toast(`the job is ${state.status} now; nothing cancelled`); return false; }
    }
    try { await api('DELETE', `/jobs/${enc(key)}`); toast(active ? 'cancel requested' : 'deleted'); return true; } catch (e) { fail(e); return false; }
  }
  async function rerun(s) {
    if (!window.confirm(`Start a new ${s.operation} job with the same params?`)) return;
    try { const key = await startJob(s.operation, s.params); go(`#job/${key}`); } catch (e) { fail(e); }
  }
  function jobLinks(results) {
    // Child job keys anywhere in a result (pipelines, profile jobs).
    const keys = new Set();
    for (const r of results) for (const [k, v] of Object.entries(r)) {
      if (/job$/.test(k) && k !== 'gufo_job' && typeof v === 'string' && /^[0-9a-f]{32}$/.test(v)) keys.add(v);
    }
    return [...keys];
  }

  // Promotion is a production change: show the command, never run it.
  function sq(v) { return `'${String(v).replace(/'/g, "'\\''")}'`; }
  function promoteBox(name, path) {
    const cmd = `# Production change: review, then run it yourself on the gufo host.\n` +
      `cp ${sq(path)} "$ABLITERATION_PROFILES_DIR"/${sq(name + '.json')}   # the production --abliteration-profiles DIR\n` +
      `# then restart the production server (or POST the file with --allow-profile-updates).`;
    return h('div', { class: 'cmd' }, h('pre', {}, cmd), copyButton(cmd));
  }
  async function upload(kind, name, source) {
    try {
      const reply = await api('POST', EP.profile(kind, name), source);
      toast(`uploaded ${kind} profile ${name}`);
      header();
      return reply;
    } catch (e) { fail(e); noteUploadRefused(e); return null; }
  }

  // Shared drafts between views (a fit or train result into a test config).
  const drafts = { test: [] };

  // Router ------------------------------------------------------------------
  let current = null;  // this route's container; a view started earlier keeps a detached one
  const main = () => current;
  function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }
  async function route() {
    for (const w of watches) w.stop();
    watches = [];
    const raw = location.hash.replace(/^#/, '') || 'datasets';
    const [path, query] = raw.split('?');
    const parts = path.split('/').map(decodeURIComponent);
    const q = new URLSearchParams(query || '');
    let tab = parts[0], key = parts[1];
    const box = h('div', {});
    current = box;
    $('#main').replaceChildren(box);
    if (tab === 'job' && key) {
      try {
        const s = await api('GET', EP.job(key, 0) );
        const to = { fit: 'run', train: 'train', test: 'test', pipeline: 'pipeline' }[s.operation] || 'history';
        location.replace(`#${to}/${key}`);
      } catch (e) { fail(e); box.replaceChildren(C.empty('no such job')); }
      return;
    }
    for (const a of document.querySelectorAll('#tabs a')) a.classList.toggle('active', a.dataset.tab === tab);
    const views = { datasets: viewDatasets, run: viewRun, train: viewTrain, test: viewTest, pipeline: viewPipeline, history: viewHistory };
    const view = views[tab] || viewDatasets;
    try { await view(key, q); } catch (e) { fail(e); box.append(h('pre', { class: 'bad' }, String(e && e.stack || e))); }
  }

  async function recentJobs(operation, onOpen) {
    const box = h('div', { class: 'card' }, h('h3', {}, `Recent ${operation} jobs`));
    try {
      const list = listOf(await api('GET', EP.jobs)).filter((j) => j.operation === operation).slice(0, 12);
      if (!list.length) box.append(C.empty('none yet'));
      else box.append(h('table', {}, h('tr', {}, h('th', {}, 'job'), h('th', {}, 'status'), h('th', {}, 'created'), h('th', {}, 'step')),
        list.map((j) => h('tr', { class: 'clickable', onclick: () => onOpen(j.key) },
          h('td', {}, h('code', {}, short(j.key))), h('td', {}, statusSpan(j.status)), h('td', {}, time(j.created_at)),
          h('td', { class: 'muted' }, (j.progress && j.progress.step) || '')))));
    } catch (e) { box.append(h('div', { class: 'bad' }, e.message)); }
    return box;
  }

  // 1. Datasets ---------------------------------------------------------------
  async function viewDatasets(name) {
    const root = main();
    const listBox = h('div', { class: 'card' });
    const detail = h('div', {});
    root.append(h('h2', {}, 'Datasets'), h('div', { class: 'grid2' }, listBox, createCard()), detail);
    async function refresh() {
      let list = [];
      try { list = listOf(await api('GET', EP.datasets)); } catch (e) { listBox.replaceChildren(h('div', { class: 'bad' }, e.message)); return; }
      listBox.replaceChildren(h('div', { class: 'row' }, h('h3', {}, 'All datasets'), h('span', { class: 'spacer' }),
        h('button', { class: 'ghost small', type: 'button', onclick: refresh }, 'Refresh')),
      list.length ? h('div', { class: 'scroll' }, h('table', {},
        h('tr', {}, ['name', 'kind', 'records', 'groups', 'control', 'builder', 'created'].map((t) => h('th', {}, t))),
        list.map((m) => h('tr', { class: 'clickable' + (m.name === name ? ' selected' : ''), onclick: () => go(`#datasets/${enc(m.name)}`) },
          h('td', {}, h('strong', {}, m.name)), h('td', {}, m.kind), h('td', { class: 'num' }, m.records),
          h('td', {}, Object.entries(m.groups || {}).map(([g, n]) => `${g} ${n}`).join(', ')),
          h('td', {}, m.control || (m.source && m.source.control_of ? `control of ${m.source.control_of}` : '')),
          h('td', {}, (m.source && m.source.builder) || ''), h('td', { class: 'muted' }, time(m.created_at)))))) : C.empty('no datasets yet'));
    }
    refresh();
    if (name) detail.append(await datasetDetail(name));
  }

  function sourcesEditor(secondLabel) {
    const box = h('div', { class: 'wide' });
    const rows = [];
    function add(v) {
      const r = { name: h('input', { placeholder: 'name', value: (v && v.name) || '' }),
        problems: h('input', { placeholder: 'problems.jsonl (id, problem)', value: (v && v.problems) || '' }),
        traces: h('input', { placeholder: secondLabel, value: (v && v.traces) || '' }) };
      const line = h('div', { class: 'row' }, r.name, r.problems, r.traces,
        h('button', { class: 'ghost small', type: 'button', onclick: () => { rows.splice(rows.indexOf(r), 1); line.remove(); } }, 'x'));
      for (const i of [r.name, r.problems, r.traces]) { i.style.flex = i === r.name ? '0 0 120px' : '1 1 220px'; i.style.width = 'auto'; }
      rows.push(r); list.append(line);
    }
    const list = h('div', {});
    box.append(h('h4', {}, 'Sources (name, problems JSONL, ' + secondLabel + ')'), list,
      h('button', { class: 'ghost small', type: 'button', onclick: () => add() }, 'Add source'));
    add();
    return { el: box, values: () => rows.map((r) => ({ name: r.name.value.trim(), problems: r.problems.value.trim(), traces: r.traces.value.trim() }))
      .filter((r) => r.name || r.problems || r.traces) };
  }

  function createCard() {
    const card = h('div', { class: 'card' }, h('h3', {}, 'Create a dataset'));
    const pick = h('select', {},
      h('option', { value: 'traces' }, 'Traces builder (thought boundaries, with a shuffled control)'),
      h('option', { value: 'pairs' }, 'Pairs builder (CAA: with and without an instruction)'),
      h('option', { value: 'generate' }, 'Model-generated pairs (pairs job: the model answers both ways)'),
      h('option', { value: 'import' }, 'Import a collector JSONL'));
    let body = h('div', {});
    card.append(pick, body);
    function show() {
      const fresh = h('div', {});
      body.replaceWith(fresh);
      body = fresh;
      const kind = pick.value;
      let f, src, op = 'dataset';
      const common = [{ key: 'name', label: 'Name' }, { key: 'replace', label: 'Replace an existing dataset', type: 'check' }];
      if (kind === 'traces') {
        src = sourcesEditor('traces JSONL (id, reasoning)');
        f = form([...common, { key: 'control_name', label: 'Control name', placeholder: 'NAME-shuf' },
          { key: 'suffix', label: 'Suffix after each problem', type: 'textarea', placeholder: '\n\nPut the final answer in \\boxed{}.' },
          { key: 'max_chars', label: 'Max chars', type: 'int', placeholder: '36000' },
          { key: 'min_each', label: 'Min of each label', type: 'int', placeholder: '3' },
          { key: 'seed', label: 'Control seed', type: 'int', placeholder: '0' },
          { key: 'reasoning_field', label: 'Reasoning field', placeholder: 'reasoning' },
          { key: 'opener', label: 'Re-check opener regex', wide: true, placeholder: 'default: Wait|Hmm|Let me verify ...' },
          { key: 'exclude_ids', label: 'Exclude ids (comma)', type: 'list', wide: true }], () => s.refresh());
      } else if (kind === 'pairs') {
        src = sourcesEditor('rows JSONL (id, text or reasoning)');
        f = form([...common, { key: 'instruction', label: 'Instruction', type: 'textarea', wide: true },
          { key: 'mode', label: 'Mode', type: 'select', options: ['answers', 'segments'] },
          { key: 'answer_field', label: 'Answer field', placeholder: 'text / reasoning' },
          { key: 'suffix', label: 'Suffix', type: 'textarea' },
          { key: 'max_chars', label: 'Max chars', type: 'int', placeholder: '36000' },
          { key: 'require_stop', label: 'Only rows ending in stop', type: 'check', value: true, omitWhen: true },
          { key: 'exclude_ids', label: 'Exclude ids (comma)', type: 'list', wide: true }], () => s.refresh());
      } else if (kind === 'generate') {
        op = 'pairs';
        f = form([...common, { key: 'instruction', label: 'Instruction', type: 'textarea', wide: true },
          { key: 'prompts', label: 'Prompts JSONL (id, prompt or problem)', wide: true },
          { key: 'mode', label: 'Mode', type: 'select', options: ['answers', 'segments'] },
          { key: 'reasoning_effort', label: 'Reasoning effort', type: 'select', options: ['', 'low', 'medium', 'high', 'xhigh'] },
          { key: 'concurrency', label: 'Concurrency', type: 'int' },
          { key: 'contrast', label: 'Also write NAME-answers (own-answer contrast)', type: 'check', value: true, omitWhen: true },
          { key: 'exclude_ids', label: 'Exclude ids (comma)', type: 'list', wide: true },
          { key: 'sampling', label: 'Sampling (JSON, e.g. {"max_tokens": 4096})', type: 'json' }], () => s.refresh());
        body.append(h('p', { class: 'muted' }, 'Runs the `pairs` operation (README: model-generated instruction pairs). ' +
          'The field names follow the README; adjust them in the extra JSON if the service differs.'));
      } else {
        f = form([...common, { key: 'path', label: 'Collector JSONL path', wide: true },
          { key: 'control', label: 'Control JSONL path (optional)', wide: true },
          { key: 'mode', label: 'Mode', type: 'select', options: ['', 'collect', 'answers', 'segments'] }], () => s.refresh());
      }
      const s = starter(op, () => {
        const p = f.values();
        if (op === 'dataset') p.builder = kind;
        if (src) p.sources = src.values();
        if (p.replace === false) delete p.replace;
        return p;
      }, { label: kind === 'generate' ? 'Start the pairs job' : 'Build', started: (key) => go(`#history/${key}`) });
      body.append(f.el, src ? src.el : null, s.el, h('div', { class: 'row' }, s.button));
      body.addEventListener('input', () => s.refresh());
      s.refresh();
    }
    pick.addEventListener('change', show);
    show();
    return card;
  }

  async function datasetDetail(name) {
    const card = h('div', { class: 'card' });
    let meta;
    try { meta = await api('GET', EP.dataset(name)); } catch (e) { card.append(h('div', { class: 'bad' }, e.message)); return card; }
    const records = h('div', {});
    const isTraces = !!(meta.source && meta.source.builder === 'traces');
    const limit = meta.kind === 'segments' ? 4 : 20;
    let offset = 0;
    card.append(h('div', { class: 'row' }, h('h3', {}, `${name} (${meta.kind}, ${meta.records} records)`), h('span', { class: 'spacer' }),
      meta.kind !== 'collect' ? h('button', { class: 'small', type: 'button', onclick: () => { drafts.fit = { dataset: name }; go('#run'); } }, 'Fit this dataset') : null),
    h('div', { class: 'row' }, Object.entries(meta.groups || {}).map(([g, n]) => h('span', { class: 'chip' }, `${groupName(meta.kind, g, isTraces)}: ${n}`)),
      meta.control ? h('a', { href: `#datasets/${enc(meta.control)}` }, `control: ${meta.control}`) : null),
    h('details', {}, h('summary', {}, 'dataset.json'), h('pre', {}, json(meta))), records);
    async function page() {
      const reply = await optional(EP.records(name, offset, limit));
      if (reply === null) {
        records.replaceChildren(h('div', { class: 'empty' }, 'This service has no paged record listing ' +
          `(GET /datasets/${name}/records). The records are in WORK/datasets/${name}/records.jsonl.`));
        return;
      }
      const rows = listOf(reply);
      const total = reply.total ?? meta.records;
      let control = null;
      if (meta.kind === 'segments' && meta.control) {
        try { control = listOf(await optional(EP.records(meta.control, offset, limit))); } catch (e) { control = null; }
      }
      const nav = () => h('div', { class: 'row' },
        h('button', { class: 'ghost small', type: 'button', disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - limit); page().catch(fail); } }, 'Previous'),
        rows.length ? `${offset + 1}-${offset + rows.length} of ${total}` : `none of ${total}`,
        h('button', { class: 'ghost small', type: 'button', disabled: offset + rows.length >= total, onclick: () => { offset += limit; page().catch(fail); } }, 'Next'));
      records.replaceChildren(h('h3', {}, 'Records'), nav(), meta.kind === 'segments' ? tracePairs(rows, control, isTraces) : recordTable(meta.kind, rows), nav());
    }
    page().catch(fail);
    return card;
  }
  function groupName(kind, g, traces) {
    if (kind === 'segments' && traces) return g === 'exemplar' ? 'before a re-check (exemplar)' : g === 'own' ? 'moving on (own)' : g;
    if (kind === 'answers') return g === 'exemplar' ? 'with the instruction (exemplar)' : g === 'own' ? 'bare prompt (own)' : g;
    return g;
  }
  function recordTable(kind, rows) {
    return h('div', { class: 'scroll' }, h('table', {},
      h('tr', {}, h('th', {}, 'group'), h('th', {}, 'pair'), h('th', {}, 'text'), kind === 'answers' ? h('th', {}, 'answer') : h('th', {}, 'thinking')),
      rows.map((r) => h('tr', {}, h('td', {}, h('span', { class: `tag ${r.group}` }, groupName(kind, r.group))), h('td', { class: 'mono' }, r.pair || ''),
        h('td', {}, clip(r.text, 600)), h('td', {}, kind === 'answers' ? clip(r.answer, 600) : String(r.thinking))))));
  }
  function clip(s, n) { s = String(s ?? ''); return s.length > n ? s.slice(0, n) + ` ... (${s.length} chars)` : s; }
  // A trace's thoughts with each boundary labelled by the real groups and the shuffled control.
  function tracePairs(rows, control, traces) {
    const word = (g) => (traces ? (g === 'exemplar' ? 'before a re-check' : 'moving on') : g);
    const byPair = new Map();
    for (const r of rows) { if (!byPair.has(r.pair)) byPair.set(r.pair, {}); byPair.get(r.pair)[r.group] = r; }
    const ctrl = new Map();
    for (const r of control || []) { if (!ctrl.has(r.pair)) ctrl.set(r.pair, {}); ctrl.get(r.pair)[r.group] = r; }
    const out = h('div', {});
    for (const [pair, g] of byPair) {
      const any = g.exemplar || g.own;
      const real = labels(g);
      const c = ctrl.has(pair) ? labels(ctrl.get(pair)) : null;
      const box = h('div', { class: 'thoughts' });
      any.segments.forEach((seg, k) => {
        box.append(h('div', { class: 'thought' }, clip(seg, 1500)));
        if (k < any.segments.length - 1) {
          box.append(h('div', { class: 'boundary' }, `end of thought ${k + 1}:`,
            real[k] ? h('span', { class: `tag ${real[k]}` }, word(real[k])) : h('span', { class: 'muted' }, 'not selected'),
            c ? h('span', { class: 'muted' }, 'control:') : null,
            c ? (c[k] ? h('span', { class: `tag ${c[k]}` }, c[k] === 'exemplar' ? 'exemplar' : 'own') : h('span', { class: 'muted' }, '-')) : null));
        }
      });
      const n = (x) => (x ? x.select.length : 0);
      out.append(h('h4', {}, `${pair}: ${n(g.exemplar)} ${word('exemplar')}, ${n(g.own)} ${word('own')}` +
        (c ? ` (control: same counts, labels permuted within the trace)` : '')), h('div', { class: 'muted' }, clip(any.text, 300)), box);
    }
    if (!byPair.size) out.append(C.empty('no records on this page'));
    return out;
  }
  function labels(g) {
    const out = {};
    for (const grp of ['exemplar', 'own']) for (const k of (g[grp] && g[grp].select) || []) out[k] = grp;
    return out;
  }

  // 2. Run (fit) ----------------------------------------------------------------
  async function viewRun(key) {
    const root = main();
    root.append(h('h2', {}, 'Run: collect and fit'));
    const formCard = h('details', { class: 'card', open: !key }, h('summary', {}, 'New fit job'));
    root.append(formCard);
    let datasets = [];
    try { datasets = listOf(await api('GET', EP.datasets)).filter((d) => d.kind !== 'collect'); } catch (e) { fail(e); }
    const draft = drafts.fit || {}; drafts.fit = null;
    const f = form([
      { key: 'dataset', label: 'Dataset', type: 'select', options: ['', ...datasets.map((d) => [d.name, `${d.name} (${d.kind}, ${d.records})`])], value: draft.dataset || '' },
      { key: 'collection', label: 'Or an existing collection path', placeholder: '/collect/NAME' },
      { key: 'control_collection', label: 'Its control collection path' },
      { key: 'control', label: 'Fit the shuffled control too', type: 'check', value: true, omitWhen: true },
      { key: 'collect_only', label: 'Collect only (fit elsewhere)', type: 'check' },
      { key: 'collection_name', label: 'gufo collection name', placeholder: 'DATASET-<sha12>' },
      { key: 'collections_dir', label: 'Collections dir (local)', placeholder: 'service default' },
      { key: 'top_n', label: 'Top N layers', type: 'int', placeholder: '4' },
      { key: 'profile_layers', label: 'Profile layers', type: 'intlist', placeholder: 'top 2' },
      { key: 'alpha', label: 'Alpha (additive)', type: 'number', placeholder: '1.0' },
      { key: 'additive_kind', label: 'Additive kind', placeholder: 'mean_diff' },
      { key: 'prefix', label: 'Profile prefix', placeholder: 'the dataset name' },
    ], () => s.refresh());
    const s = starter('fit', () => { const p = f.values(); if (p.collect_only === false) delete p.collect_only; return p; },
      { started: (k) => go(`#run/${k}`) });
    formCard.append(f.el, s.el, h('div', { class: 'row' }, s.button));
    s.refresh();
    if (!key) { root.append(await recentJobs('fit', (k) => go(`#run/${k}`))); return; }

    const known = ['collect', 'collection', 'layer', 'cosine', 'transfer', 'top', 'control_cosine', 'profile', 'collected'];
    const live = h('div', {});
    let panel;
    let heatPick = 'real';
    const w = watchIn(root, key, () => { panel.render(); render(); });
    panel = jobPanel(w, { known });
    root.append(panel.el, live);
    function render() {
      const layers = { real: [], control: [] };
      for (const r of w.of('layer')) (layers[r.dataset] || (layers[r.dataset] = []))[r.layer] = r;
      const n = Math.max(layers.real.length, layers.control.length);
      const cats = [...Array(n).keys()];
      const top = {};
      for (const r of w.of('top')) top[r.dataset] = r.layers;
      const cos = {};
      for (const r of w.of('cosine')) cos[r.dataset] = r.matrix;
      const transfer = {};
      for (const r of w.of('transfer')) transfer[r.dataset] = r;
      const ccos = w.of('control_cosine').pop();
      const collects = {};
      for (const r of w.of('collect')) collects[r.dataset] = r;
      const colls = w.of('collection');
      const profiles = w.of('profile');
      const hl = new Set(top.real || []);
      const series = (field) => [{ name: 'real labels', cls: 's1', values: cats.map((i) => layers.real[i] ? layers.real[i][field] : null) },
        ...(layers.control.length ? [{ name: 'shuffled control', cls: 's2', values: cats.map((i) => layers.control[i] ? layers.control[i][field] : null) }] : [])];

      const collectCard = h('div', { class: 'card' }, h('h3', {}, 'Collection'),
        Object.keys(collects).length || colls.length ? null : C.empty('waiting for the collection'),
        Object.values(collects).map((c) => h('div', {}, h('div', { class: 'row' }, h('strong', {}, c.dataset), `${c.collection}`, statusSpan(c.status),
          `${c.done}/${c.total} records`, h('code', { class: 'muted' }, `gufo job ${short(c.gufo_job || '')}`)),
        h('div', { class: 'progress' }, h('i', { style: { width: (c.total ? (100 * c.done) / c.total : 0) + '%' } })),
        (c.numerics_deviations || []).length ? h('div', { class: 'bad' }, `numerics deviations: ${c.numerics_deviations.join(', ')}`) : null)),
        colls.map((c) => h('div', { class: 'muted' }, `${c.dataset}: ${c.pairs} pairs${c.skipped ? `, ${c.skipped} skipped` : ''}` +
          (c.layers ? `, ${c.layers} layers, hidden ${c.hidden}, readout ${c.readout}` : '') + ` - ${c.path}`)),
        w.of('collected').map((c) => h('pre', {}, json(c))));

      const sep = h('div', {}); const snr = h('div', {}); const norm = h('div', {});
      if (n) {
        C.bars(sep, { categories: cats, series: series('acc'), yMin: 0, yMax: 1, ref: 0.5, highlight: hl, yLabel: 'held-out accuracy', xLabel: 'layer', label: 'held-out separation per layer' });
        C.bars(snr, { categories: cats, series: series('snr'), highlight: hl, yLabel: 'held-out SNR', xLabel: 'layer', label: 'SNR per layer' });
        C.lines(norm, { series: [{ name: 'real labels', cls: 's1', dots: true, points: cats.map((i) => [i, layers.real[i] && layers.real[i].norm]) },
          ...(layers.control.length ? [{ name: 'shuffled control', cls: 's2', dots: true, points: cats.map((i) => [i, layers.control[i] && layers.control[i].norm]) }] : [])],
        yLabel: '|mean D|', xLabel: 'layer', height: 160 });
      } else { sep.append(C.empty('per-layer rows appear as the fit computes them')); }

      const heat = h('div', {});
      const pickers = h('div', { class: 'row' }, Object.keys(cos).map((d) => h('button', { type: 'button', class: d === heatPick ? 'small' : 'ghost small',
        onclick: () => { heatPick = d; render(); } }, d)));
      if (cos[heatPick] || cos.real) C.heatmap(heat, { matrix: cos[heatPick] || cos.real, min: -1, max: 1, diverging: true, rowName: 'layer', colName: 'layer', label: 'cross-layer cosine' });
      else heat.append(C.empty('the cosine table arrives after the per-layer rows'));

      const ctrlBox = h('div', {});
      if (ccos) C.lines(ctrlBox, { series: [{ name: 'cos(real, control)', cls: 's2', dots: true, points: ccos.cosines.map((v, i) => [i, v]) }], ref: 0,
        yLabel: 'cosine of mean differences', xLabel: 'layer', height: 160 });
      else ctrlBox.append(C.empty(layers.control.length ? 'after the control fit' : 'no control (the dataset has none, or control is off)'));

      const trBox = h('div', {});
      const tr = transfer.real;
      if (tr) {
        C.heatmap(trBox, { matrix: tr.sources.map((s0) => (tr.separated[String(s0)] || []).map((v) => v / tr.pairs)), rows: tr.sources,
          min: 0.5, max: 1, rowName: 'direction from layer', colName: 'applied at layer', label: 'transfer' });
      } else trBox.append(C.empty('after the top layers'));

      const topCard = h('div', { class: 'card' }, h('h3', {}, 'Top layers (by held-out SNR, then accuracy)'),
        Object.keys(top).length ? Object.entries(top).map(([d, ls]) => h('div', { class: 'row' }, h('strong', {}, d),
          h('div', { class: 'chips' }, ls.map((l) => h('span', { class: `chip ${d === 'real' ? 's1' : 's2'}` },
            `L${l}` + (layers[d] && layers[d][l] ? `: ${layers[d][l].separated}/${layers[d][l].pairs}, SNR ${fmt(layers[d][l].snr)}` : '')))))) : C.empty('not yet'));

      const profCard = h('div', { class: 'card' }, h('h3', {}, 'Default profiles'),
        profiles.length ? h('table', {}, h('tr', {}, ['name', 'mode', 'direction', 'upload as', ''].map((t) => h('th', {}, t))),
          profiles.map((p) => {
            const as = h('input', { value: p.name });
            const out = h('div', { class: 'muted' });
            return [h('tr', {}, h('td', {}, h('strong', {}, p.name)), h('td', {}, p.mode),
              h('td', {}, p.direction_layer !== null && p.direction_layer !== undefined ? `L${p.direction_layer} ${p.direction_kind || ''}` : '-'),
              h('td', {}, as),
              h('td', {}, h('div', { class: 'row' },
                h('button', { class: 'small', type: 'button', onclick: async () => {
                  const r = await upload('abliteration', as.value.trim(), { job: key, profile: p.name });
                  if (r) out.textContent = 'gufo: ' + JSON.stringify(r).slice(0, 300);
                } }, 'Upload'),
                h('button', { class: 'ghost small', type: 'button', onclick: () => {
                  drafts.test.push({ name: as.value.trim(), abliteration: { enabled: true, profile: as.value.trim() } });
                  toast(`added a test configuration for ${as.value.trim()}; upload it before the test`);
                } }, 'Add to test'),
                h('details', {}, h('summary', {}, 'promote'), promoteBox(as.value.trim(), p.path))))),
            h('tr', {}, h('td', { colspan: 5 }, out, h('code', { class: 'muted' }, p.path)))];
          })) : C.empty('written at the end of the fit'));

      const table = h('details', {}, h('summary', {}, 'Per-layer table'), n ? h('div', { class: 'scroll' }, h('table', {},
        h('tr', {}, ['layer', 'separated', 'acc', 'SNR', '|d|', 'control acc', 'control SNR', 'cos(real, control)'].map((t) => h('th', { class: 'num' }, t))),
        cats.map((i) => { const r = layers.real[i] || {}, c = layers.control[i] || {};
          return h('tr', { class: hl.has(i) ? 'diff-row' : '' }, h('td', { class: 'num' }, i), h('td', { class: 'num' }, r.separated !== undefined ? `${r.separated}/${r.pairs}` : '-'),
            h('td', { class: 'num' }, fmt(r.acc)), h('td', { class: 'num' }, fmt(r.snr)), h('td', { class: 'num' }, fmt(r.norm)),
            h('td', { class: 'num' }, fmt(c.acc)), h('td', { class: 'num' }, fmt(c.snr)), h('td', { class: 'num' }, fmt(ccos ? ccos.cosines[i] : null))); }))) : null);

      keepOpen(live, () => {
      live.replaceChildren(collectCard,
        h('div', { class: 'card' }, h('h3', {}, 'Held-out separation per layer'), h('div', { class: 'muted' }, 'Leave-one-out accuracy; 0.5 is chance. Shaded: the top layers.'), sep),
        h('div', { class: 'card' }, h('h3', {}, 'Held-out SNR per layer'), snr, h('h4', {}, '|mean difference| per layer'), norm, table),
        h('div', { class: 'grid2' },
          h('div', { class: 'card' }, h('h3', {}, 'Cross-layer cosine of the mean differences'), pickers, heat),
          h('div', { class: 'card' }, h('h3', {}, 'Control next to the real labels'), h('div', { class: 'muted' },
            'The control permutes the labels within each trace; its separation and SNR are the orange series above. ' +
            'Here: the cosine between the real and control mean differences.'), ctrlBox,
          h('h4', {}, 'Transfer: held-out separation along a top layer\'s direction'), trBox)),
        topCard, profCard);
      });
    }
    render();
  }

  // 3. Train --------------------------------------------------------------------
  const KINDS = ['attention', 'mlp', 'residual'];
  function argValue(args, name, fallback) {
    for (let i = 0; i < args.length; i++) {
      if (args[i] === name && i + 1 < args.length) return args[i + 1];
      if (args[i].startsWith(name + '=')) return args[i].slice(name.length + 1);
    }
    return fallback;
  }
  // spsa.py thetas are block parameters: [kind][block]; expand to per-layer.
  function expandTheta(theta, layers, block) {
    const nb = Math.ceil(layers / block);
    const out = {};
    KINDS.forEach((k, ki) => { out[k] = [...Array(layers).keys()].map((l) => theta[ki * nb + Math.floor(l / block)]); });
    return out;
  }

  async function viewTrain(key) {
    const root = main();
    root.append(h('h2', {}, 'Train: SPSA over per-layer strengths'));
    const formCard = h('details', { class: 'card', open: !key }, h('summary', {}, 'New train job'));
    root.append(formCard);
    const learn = KINDS.map((k) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: true, value: k }), k));
    const f = form([
      { key: 'profile', label: 'Base profile (--profile)', list: 'dl-profiles' },
      { key: 'additive', label: 'Additive profile (--additive)', type: 'check' },
      { key: 'direction_layer', label: 'Direction layer', type: 'int' },
      { key: 'direction_kind', label: 'Direction kind', placeholder: 'projected' },
      { key: 'objective', label: 'Objective job key (its pairs, keep and guard rows)', wide: true },
      { key: 'pairs', label: 'Or pairs JSONL', wide: true }, { key: 'ordinary', label: 'Ordinary JSONL', wide: true },
      { key: 'guard', label: 'Guard JSONL', wide: true }, { key: 'facts', label: 'Facts JSONL', wide: true },
      { key: 'base_profile', label: 'Base profile for the trained profile (file path, or JSON {"job", "profile"})', wide: true },
      { key: 'profile_name', label: 'Trained profile name' },
      { key: 'basis', label: 'Direction basis file (also learns the direction; needs the base profile above; options as more args)', wide: true },
      { key: 'resume', label: 'Resume from train job key' },
      { key: 'fast_defaults', label: 'Fast training-server defaults', type: 'check', value: true, omitWhen: true },
      { key: 'layers', label: 'Layers', type: 'int', value: 48 },
      { key: 'block_size', label: 'Block size', type: 'int', placeholder: '1' },
      { key: 'init', label: 'Init (number or JSON)', value: '0' },
      { key: 'min_strength', label: 'Min strength', type: 'number', placeholder: '0' },
      { key: 'max_strength', label: 'Max strength', type: 'number', placeholder: '4' },
      { key: 'steps', label: 'Steps', type: 'int', value: 50 },
      { key: 'a', label: 'a (step size)', type: 'number', placeholder: '0.15' },
      { key: 'c', label: 'c (perturbation)', type: 'number', placeholder: '0.15' },
      { key: 'A', label: 'A', type: 'number', placeholder: '10' },
      { key: 'lam', label: 'lam (ordinary drop)', type: 'number', placeholder: '4' },
      { key: 'nu', label: 'nu (guard weight)', type: 'number', placeholder: '1' },
      { key: 'guard_margin', label: 'Guard margin (nats)', type: 'number', placeholder: '0' },
      { key: 'mu', label: 'mu (facts weight)', type: 'number' },
      { key: 'minibatch', label: 'Minibatch', type: 'int', placeholder: '0 = all' },
      { key: 'minibatch_chains', label: 'Minibatch chains', type: 'int' },
      { key: 'eval_every', label: 'Eval every', type: 'int', placeholder: '5' },
      { key: 'answer_tokens', label: 'Answer tokens', type: 'int' },
      { key: 'seed', label: 'Seed', type: 'int' }, { key: 'concurrency', label: 'Concurrency', type: 'int' },
      { key: 'extra_args', label: 'More spsa.py args (JSON list, or space separated)', wide: true },
      { key: 'python', label: 'Python interpreter (default: the service\'s)', wide: true },
    ], () => s.refresh());
    const learnBox = h('div', { class: 'row' }, h('span', { class: 'muted' }, 'Learn:'), learn);
    learnBox.addEventListener('change', () => s.refresh());
    const s = starter('train', () => {
      const v = f.values();
      const args = [];
      const opt = { profile: '--profile', direction_layer: '--direction-layer', direction_kind: '--direction-kind', facts: '--facts', layers: '--layers', block_size: '--block-size', init: '--init',
        min_strength: '--min-strength', max_strength: '--max-strength', steps: '--steps', a: '--a', c: '--c', A: '--A', lam: '--lam',
        nu: '--nu', guard_margin: '--guard-margin', mu: '--mu', minibatch: '--minibatch', minibatch_chains: '--minibatch-chains',
        eval_every: '--eval-every', answer_tokens: '--answer-tokens', seed: '--seed', concurrency: '--concurrency' };
      for (const [k, flag] of Object.entries(opt)) if (v[k] !== undefined) args.push(flag, String(v[k]));
      if (v.additive) args.push('--additive');
      const kinds = learn.map((l) => l.firstChild).filter((i) => i.checked).map((i) => i.value);
      if (!kinds.length) throw new Error('learn at least one kind');
      if (kinds.length < 3) args.push('--learn', kinds.join(','));
      if (v.extra_args) {
        const t = v.extra_args.trim();
        args.push(...(t.startsWith('[') ? JSON.parse(t) : t.split(/\s+/)));
      }
      const p = { args };
      for (const k of ['python', 'objective', 'pairs', 'ordinary', 'guard', 'profile_name', 'resume', 'basis']) if (v[k] !== undefined) p[k] = v[k];
      if (v.fast_defaults === false) p.fast_defaults = false;
      if (v.base_profile) p.base_profile = v.base_profile.startsWith('{') ? JSON.parse(v.base_profile) : v.base_profile;
      return p;
    }, { started: (k) => go(`#train/${k}`) });
    formCard.append(f.el, learnBox, s.el, h('div', { class: 'row' }, s.button));
    s.refresh();
    if (!key) { root.append(await recentJobs('train', (k) => go(`#train/${k}`))); return; }

    const known = ['train', 'step', 'best', 'warning', 'profile', 'upload'];
    const live = h('div', {});
    let panel; let pick = null; let follow = true;
    const keepInputs = {};
    const w = watchIn(root, key, () => { panel.render(); render(); });
    panel = jobPanel(w, { known });
    root.append(panel.el, live);
    function render() {
      const head = w.of('train').pop() || {};
      const args = head.args || (w.state && w.state.params && w.state.params.args) || [];
      const layers = Number(head.layers || argValue(args, '--layers', 48));
      const block = Number(head.block_size || argValue(args, '--block-size', 1));
      const steps = w.of('step');
      const jp = [], jm = [], je = [], margin = [], drop = [], angle = [], alpha = [];
      let best = null;
      for (const r of steps) {
        const x = r.step + 1;
        if (typeof r.angle === 'number') angle.push([x, r.angle]);
        if (Array.isArray(r.alpha)) r.alpha.forEach((a, i) => { (alpha[i] = alpha[i] || []).push([x, a]); });
        const plus = r.j_plus ?? (r.plus && r.plus.j), minus = r.j_minus ?? (r.minus && r.minus.j);
        if (typeof plus === 'number') jp.push([x, plus]);
        if (typeof minus === 'number') jm.push([x, minus]);
        if (r.eval) { je.push([x, r.eval.j]); margin.push([x, r.eval.margin]); drop.push([x, r.eval.drop]);
          if (!best || r.eval.j > best.eval.j) best = r; }
      }
      const final = w.of('best').pop();
      const jBox = h('div', {}), cBox = h('div', {});
      C.lines(jBox, { series: [{ name: 'J+ (theta + c delta)', cls: 's1', points: jp }, { name: 'J- (theta - c delta)', cls: 's2', points: jm },
        { name: 'eval J (theta)', cls: 's3', points: je, dots: true }], xLabel: 'step (0 = init)', yLabel: 'J', empty: 'J per step appears as spsa.py logs it' });
      C.lines(cBox, { series: [{ name: 'eval margin', cls: 's1', points: margin, dots: true }, { name: 'eval drop', cls: 's2', points: drop, dots: true }],
        xLabel: 'step', yLabel: 'nats per token', height: 160, empty: 'evals every --eval-every steps' });
      let dirCard = null;
      if (angle.length) {
        const aBox = h('div', {}), cfBox = h('div', {});
        C.lines(aBox, { series: [{ name: 'angle to d0 (degrees)', cls: 's1', points: angle }], xLabel: 'step', yLabel: 'degrees', height: 140 });
        C.lines(cfBox, { series: alpha.map((pts, i) => ({ name: `alpha ${i + 1}`, cls: `s${(i % 3) + 1}`, dashed: i >= 3, points: pts })),
          xLabel: 'step', yLabel: 'coefficient', height: 160, ref: 0 });
        dirCard = h('div', { class: 'card' }, h('h3', {}, 'Direction (basis coefficients)'), aBox, h('h4', {}, 'alpha per basis vector'), cfBox);
      }
      // Strengths at the picked step (default: the latest), next to the best eval.
      const withTheta = steps.filter((r) => Array.isArray(r.theta));
      if (follow || pick === null || pick >= withTheta.length) pick = withTheta.length - 1;
      const at = withTheta[pick];
      const scrub = h('input', { type: 'range', min: 0, max: Math.max(0, withTheta.length - 1), value: Math.max(0, pick) });
      scrub.addEventListener('input', () => { pick = Number(scrub.value); follow = pick === withTheta.length - 1; render(); });
      const strengthBoxes = KINDS.map((k) => h('div', {}));
      if (at) {
        const cur = at.strengths || expandTheta(at.theta, layers, block);
        const bt = best && Array.isArray(best.theta) ? (best.strengths || expandTheta(best.theta, layers, block)) : null;
        KINDS.forEach((k, i) => C.bars(strengthBoxes[i], { categories: [...Array(layers).keys()],
          series: [{ name: `step ${at.step + 1}`, cls: 's1', values: cur[k] }, ...(bt && best !== at ? [{ name: `best eval (step ${best.step + 1})`, cls: 's3', values: bt[k] }] : [])],
          yLabel: k, xLabel: 'layer', height: 150, label: `${k} strengths` }));
      } else strengthBoxes[0].append(C.empty('per-layer strengths appear with the first step'));
      const warnings = w.of('warning');
      const bestCard = h('div', { class: 'card' }, h('h3', {}, 'Best so far'),
        warnings.map((x) => h('div', { class: 'bad' }, `warning ${x.code}: ${x.message || JSON.stringify(x)}`)),
        head.recipe ? h('details', {}, h('summary', {}, 'Training recipe'), h('pre', {}, json(head.recipe))) : null,
        best ? h('div', { class: 'chips' }, h('span', { class: 'chip' }, `step ${best.step + 1}`),
          Object.entries(best.eval).map(([k, v]) => h('span', { class: 'chip' }, `${k} ${fmt(v)}`))) : C.empty('no eval yet'),
        final ? h('details', {}, h('summary', {}, 'best.json (spsa.py --output)'), h('pre', {}, json(final))) : null,
        w.of('profile').map((pr) => h('div', { class: 'row' }, h('strong', {}, pr.name), h('code', { class: 'muted' }, pr.path),
          h('button', { class: 'small', type: 'button', onclick: () => upload('abliteration', pr.name, { job: key, profile: pr.name }) }, 'Upload'),
          h('details', {}, h('summary', {}, 'promote'), promoteBox(pr.name, pr.path)))),
        profileActions(key, final, w.state, keepInputs));
      keepOpen(live, () => {
      live.replaceChildren(
        h('div', { class: 'card' }, h('h3', {}, `J per step (${steps.filter((r) => r.step >= 0).length}/${head.steps || '?'})`), jBox, h('h4', {}, 'Eval components'), cBox),
        h('div', { class: 'card' }, h('div', { class: 'row' }, h('h3', {}, 'Per-layer strengths'), h('span', { class: 'spacer' }),
          at ? `step ${at.step + 1}${follow ? ' (following)' : ''}` : '', scrub), h('div', { class: 'muted' }, `layers ${layers}, block size ${block}`), strengthBoxes),
        ...(dirCard ? [dirCard] : []), bestCard);
      });
    }
    render();
  }

  function profileActions(key, best, state, keep) {
    if (!keep.name) {
      keep.name = h('input', { placeholder: 'trained profile name', value: (state && state.params && state.params.profile_name) || `trained-${short(key)}` });
      keep.base = h('input', { placeholder: 'base profile (default: the train job\'s base_profile)' });
      keep.out = h('div', {});
    }
    const { name, base, out } = keep;
    const done = state && state.status === 'done';
    async function build(uploadIt) {
      const n = name.value.trim();
      try {
        const params = { train: key, name: n, upload: uploadIt };
        const b = base.value.trim();
        if (b) params.base_profile = b.startsWith('{') ? JSON.parse(b) : b;
        const k = await startJob('profile', params);
        out.replaceChildren(h('div', {}, 'profile job ', h('a', { href: `#history/${k}` }, short(k)), uploadIt ? ' (builds, then uploads)' : ''));
      } catch (e) {
        if (e.code === 'unsupported_operation' || e.code === 'invalid_params') {
          out.replaceChildren(h('div', { class: 'bad' }, `${e.code}: ${e.message}. Without a profile operation, use "Use the best settings in a test": ` +
            'a test configuration can send the trained strengths in the request.'));
        } else fail(e);
      }
    }
    return h('div', {}, h('h4', {}, 'Trained profile'), h('div', { class: 'row' }, name, base,
      h('button', { class: 'small', type: 'button', disabled: !done, onclick: () => build(false) }, 'Build profile'),
      h('button', { class: 'small', type: 'button', disabled: !done, onclick: () => build(true) }, 'Build and upload'),
      h('button', { class: 'ghost small', type: 'button', disabled: !best, onclick: () => {
        drafts.test.push({ name: name.value.trim(), abliteration: Object.assign({ enabled: true }, best.settings) });
        toast('added a test configuration with the best settings'); go('#test');
      } }, 'Use the best settings in a test')), out,
      h('div', { class: 'muted' }, 'Build runs the profile operation (base profile + best strengths); the upload goes to gufo through the tool.'));
  }

  // 4. Test and tune --------------------------------------------------------------
  function configEditor(initial, onChange, layersHint) {
    const cfg = JSON.parse(JSON.stringify(initial || {}));
    const ab = cfg.abliteration || { enabled: false };
    const card = h('div', { class: 'card config' });
    const nameI = h('input', { value: cfg.name || 'config' });
    const effort = h('select', {}, ['', 'none', 'low', 'medium', 'high', 'xhigh'].map((v) => h('option', { value: v }, v || '(server default)')));
    effort.value = cfg.reasoning_effort || '';
    const on = h('input', { type: 'checkbox', checked: !!ab.enabled });
    const prof = h('input', { list: 'dl-profiles', placeholder: 'profile name (or a JSON list)', value: Array.isArray(ab.profile) ? JSON.stringify(ab.profile) : (ab.profile || '') });
    const layersI = h('input', { type: 'number', value: layersHint || (KINDS.map((k) => Array.isArray(ab[k]) ? ab[k].length : 0).find((n) => n) || 48), min: 1 });
    const rest = {};
    for (const [k, v] of Object.entries(ab)) if (!['enabled', 'profile', ...KINDS].includes(k)) rest[k] = v;
    const abExtra = h('textarea', { spellcheck: 'false', placeholder: '{"direction_layer": 45, "direction_kind": "projected"}' });
    abExtra.value = Object.keys(rest).length ? json(rest) : '';
    const lbp = h('input', { list: 'dl-bias', placeholder: 'wait, ...', value: (cfg.logit_bias_profiles || []).join(', ') });
    const lbo = h('textarea', { spellcheck: 'false', placeholder: '{"wait": -2}' });
    lbo.value = cfg.logit_bias_overrides ? json(cfg.logit_bias_overrides) : '';
    const samp = h('textarea', { spellcheck: 'false', placeholder: '{"max_tokens": 32768, "temperature": 0.6, "seed": 1}' });
    samp.value = cfg.sampling ? json(cfg.sampling) : '';
    // Per kind: profile default, one value, or per layer.
    const kinds = {};
    const kindBoxes = KINDS.map((k) => {
      const v = ab[k];
      const st = { mode: v === undefined || v === null ? 'default' : Array.isArray(v) ? 'layers' : 'uniform',
        uniform: typeof v === 'number' ? v : 1, values: Array.isArray(v) ? v.slice() : null, min: -4, max: 4 };
      if (Array.isArray(v)) { st.min = Math.min(st.min, ...v); st.max = Math.max(st.max, ...v); }
      kinds[k] = st;
      const box = h('div', {});
      st.render = () => renderKind(k, st, box);
      st.render();
      return box;
    });
    function nLayers() { return Math.max(1, Math.trunc(Number(layersI.value) || 48)); }
    function ensure(st) {
      const n = nLayers();
      if (!st.values) st.values = Array(n).fill(st.mode === 'uniform' ? st.uniform : 0);
      if (st.values.length !== n) st.values = [...Array(n).keys()].map((i) => st.values[i] ?? 0);
    }
    function renderKind(k, st, box) {
      const mode = h('select', {}, [['default', 'profile default'], ['uniform', 'one value'], ['layers', 'per layer']].map(([v, l]) => h('option', { value: v }, l)));
      mode.value = st.mode;
      mode.addEventListener('change', () => { st.mode = mode.value; if (st.mode === 'layers') ensure(st); st.render(); onChange(); });
      const parts = [h('div', { class: 'row' }, h('strong', {}, k), mode)];
      if (st.mode === 'uniform') {
        const u = h('input', { type: 'number', step: 'any', value: st.uniform });
        u.style.width = '90px';
        u.addEventListener('input', () => { st.uniform = Number(u.value); onChange(); });
        parts[0].append(u);
      } else if (st.mode === 'layers') {
        ensure(st);
        const lo = h('input', { type: 'number', step: 'any', value: st.min, title: 'slider minimum' });
        const hi = h('input', { type: 'number', step: 'any', value: st.max, title: 'slider maximum' });
        for (const i of [lo, hi]) i.style.width = '70px';
        lo.addEventListener('change', () => { st.min = Number(lo.value); st.render(); });
        hi.addEventListener('change', () => { st.max = Number(hi.value); st.render(); });
        const val = h('input', { type: 'number', step: 'any', value: 1 }); val.style.width = '70px';
        const from = h('input', { type: 'number', value: 0, min: 0 }); from.style.width = '60px';
        const to = h('input', { type: 'number', value: nLayers() - 1, min: 0 }); to.style.width = '60px';
        const blockI = h('input', { type: 'number', value: 4, min: 1 }); blockI.style.width = '55px';
        parts[0].append(h('span', { class: 'muted' }, 'range'), lo, hi);
        parts.push(h('div', { class: 'row' },
          h('button', { class: 'ghost small', type: 'button', onclick: () => { st.values = st.values.map(() => Number(val.value)); st.render(); onChange(); } }, 'All equal'),
          val,
          h('button', { class: 'ghost small', type: 'button', onclick: () => {
            const a = Math.max(0, Number(from.value)), b = Math.min(nLayers() - 1, Number(to.value));
            st.values = st.values.map((x, i) => (i >= a && i <= b ? Number(val.value) : x)); st.render(); onChange();
          } }, 'Set layers'), from, 'to', to,
          h('button', { class: 'ghost small', type: 'button', title: 'Average each block of consecutive layers', onclick: () => {
            const b = Math.max(1, Number(blockI.value));
            st.values = st.values.map((x, i) => { const s0 = Math.floor(i / b) * b; const blk = st.values.slice(s0, s0 + b); return +(blk.reduce((p, q) => p + q, 0) / blk.length).toFixed(5); });
            st.render(); onChange();
          } }, 'Block'), blockI));
        const grid = h('div', { class: 'sliders' });
        st.values.forEach((v, i) => {
          const r = h('input', { type: 'range', min: Math.min(st.min, v), max: Math.max(st.max, v), step: 'any', value: v });
          const n = h('input', { type: 'number', step: 'any', value: v });
          r.addEventListener('input', () => { st.values[i] = Number(r.value); n.value = +Number(r.value).toFixed(4); onChange(); });
          n.addEventListener('input', () => { st.values[i] = Number(n.value); r.value = n.value; onChange(); });
          grid.append(h('div', { class: 'slider' }, h('span', { class: 'muted' }, `L${i}`), r, n));
        });
        parts.push(h('details', { open: true }, h('summary', {}, `${st.values.length} layer sliders`), grid));
      }
      box.replaceChildren(...parts);
    }
    const fromJob = h('input', { placeholder: 'train job key' });
    async function loadTrain() {
      try {
        const j = await loadAll(fromJob.value.trim());
        const best = j.of('best').pop();
        let settings = best && best.settings;
        if (!settings) {
          const last = j.of('step').filter((r) => Array.isArray(r.theta)).pop();
          if (!last) throw new Error('that job has no strengths yet');
          const args = (j.state.params && j.state.params.args) || [];
          const t = expandTheta(last.theta, Number(argValue(args, '--layers', 48)), Number(argValue(args, '--block-size', 1)));
          settings = { ...t };
          const p = argValue(args, '--profile', ''); if (p) settings.profile = p;
        }
        apply(settings);
        toast('loaded the train job\'s strengths');
      } catch (e) { fail(e); }
    }
    function apply(settings) {
      on.checked = true;
      if (settings.profile) prof.value = Array.isArray(settings.profile) ? JSON.stringify(settings.profile) : settings.profile;
      const extra = {};
      for (const [k, v] of Object.entries(settings)) {
        if (KINDS.includes(k)) {
          const st = kinds[k];
          if (Array.isArray(v)) { st.mode = 'layers'; st.values = v.slice(); layersI.value = v.length; st.min = Math.min(st.min, ...v); st.max = Math.max(st.max, ...v); }
          else { st.mode = 'uniform'; st.uniform = v; }
          st.render();
        } else if (!['enabled', 'profile'].includes(k)) extra[k] = v;
      }
      abExtra.value = Object.keys(extra).length ? json(extra) : '';
      onChange();
    }
    layersI.addEventListener('change', () => { for (const k of KINDS) if (kinds[k].mode === 'layers') { ensure(kinds[k]); kinds[k].render(); } onChange(); });
    for (const el of [nameI, effort, on, prof, abExtra, lbp, lbo, samp]) { el.addEventListener('input', onChange); el.addEventListener('change', onChange); }
    const tools = h('div', { class: 'row' });
    card.append(h('div', { class: 'row' }, h('label', { class: 'field' }, 'Name', nameI), h('label', { class: 'field' }, 'Reasoning effort', effort),
      h('span', { class: 'spacer' }), tools),
    h('h4', {}, 'Abliteration'),
    h('div', { class: 'row' }, h('label', { class: 'check row' }, on, 'enabled'), prof, h('label', { class: 'row muted' }, 'layers', layersI),
      fromJob, h('button', { class: 'ghost small', type: 'button', onclick: loadTrain }, 'From a train job')),
    kindBoxes,
    h('details', {}, h('summary', {}, 'More: abliteration fields, logit bias, sampling'),
      h('div', { class: 'fields' },
        h('label', { class: 'wide' }, 'Other abliteration fields (JSON)', abExtra),
        h('label', {}, 'Logit-bias profiles (comma)', lbp),
        h('label', {}, 'Logit-bias overrides (JSON)', lbo),
        h('label', { class: 'wide' }, 'Sampling (JSON, merged into the request)', samp))));
    function parseJ(el, label) {
      el.classList.remove('invalid');
      const t = el.value.trim();
      if (!t) return undefined;
      try { return JSON.parse(t); } catch (e) { el.classList.add('invalid'); throw new Error(`${nameI.value}: ${label}: ${e.message}`); }
    }
    function value() {
      const c = { name: nameI.value.trim() };
      if (effort.value) c.reasoning_effort = effort.value;
      if (on.checked) {
        const a = { enabled: true };
        const p = prof.value.trim();
        if (p) a.profile = p.startsWith('[') ? JSON.parse(p) : p;
        for (const k of KINDS) {
          const st = kinds[k];
          if (st.mode === 'uniform') a[k] = st.uniform;
          else if (st.mode === 'layers') { ensure(st); a[k] = st.values.every((x) => x === st.values[0]) ? st.values[0] : st.values.map((x) => +x.toFixed(5)); }
        }
        Object.assign(a, parseJ(abExtra, 'abliteration fields') || {});
        c.abliteration = a;
      }
      const l = lbp.value.split(',').map((x) => x.trim()).filter(Boolean);
      if (l.length) c.logit_bias_profiles = l;
      const o = parseJ(lbo, 'logit-bias overrides'); if (o) c.logit_bias_overrides = o;
      const s = parseJ(samp, 'sampling'); if (s) c.sampling = s;
      return c;
    }
    return { el: card, value, tools, setBaseline: (b) => card.classList.toggle('baseline', b), apply };
  }

  async function viewTest(key, q) {
    const root = main();
    root.append(h('h2', {}, 'Test and tune'));
    const formCard = h('details', { class: 'card', open: !key || drafts.test.length > 0 || q.has('try') }, h('summary', {}, key ? 'Try next setting / new test job' : 'New test job'));
    root.append(formCard);
    const prev = key ? await loadAll(key).catch(() => null) : null;
    const pp = (prev && prev.state.params) || {};
    const set = pp.set || {};
    const f = form([
      { key: 'kind', label: 'Set', type: 'select', options: [['math', 'math (think_eval)'], ['coding', 'coding (sandboxed)'], ['prompts', 'prompts (ungraded)']], value: set.kind || 'math' },
      { key: 'path', label: 'Set path (JSONL)', wide: true, value: set.path },
      { key: 'split', label: 'Split', value: set.split }, { key: 'splits', label: 'Splits (coding, comma)', type: 'list', value: (set.splits || []).join(', ') },
      { key: 'ids', label: 'Problem ids (comma)', type: 'list', wide: true, value: (set.ids || []).join(', ') },
      { key: 'limit', label: 'Limit', type: 'int', value: set.limit }, { key: 'max_tokens', label: 'Max tokens', type: 'int', value: set.max_tokens },
      { key: 'seed_base', label: 'Seed base', type: 'int', value: set.seed_base },
      { key: 'suffix', label: 'Suffix', value: set.suffix }, { key: 'image', label: 'Sandbox image (coding)', value: set.image },
      { key: 'concurrency', label: 'Concurrency', type: 'int', value: pp.concurrency ?? 2 },
      { key: 'model', label: 'Model', value: pp.model }, { key: 'request_timeout', label: 'Request timeout (s)', type: 'number', value: pp.request_timeout },
      { key: 'resume', label: 'Resume: earlier test job keys (comma); their finished rows are reused', type: 'list', wide: true },
    ], () => s.refresh());
    const cfgBox = h('div', {});
    const editors = [];
    const layersHint = loadedProfiles.raw && loadedProfiles.raw.abliteration && loadedProfiles.raw.abliteration.layers;
    function addConfig(c, at) {
      const e = configEditor(c, () => s.refresh(), layersHint);
      e.tools.append(
        h('button', { class: 'ghost small', type: 'button', title: 'Clone with these settings', onclick: () => {
          const v = e.value(); v.name = nextName(v.name); addConfig(v, editors.indexOf(e) + 1);
        } }, 'Clone'),
        h('button', { class: 'ghost small', type: 'button', title: 'Make this the baseline (first)', onclick: () => {
          editors.splice(editors.indexOf(e), 1); editors.unshift(e); layout();
        } }, 'Baseline'),
        h('button', { class: 'danger small', type: 'button', onclick: () => { editors.splice(editors.indexOf(e), 1); layout(); } }, 'Remove'));
      editors.splice(at === undefined ? editors.length : at, 0, e);
      layout();
    }
    function layout() {
      cfgBox.replaceChildren(...editors.map((e, i) => { e.setBaseline(i === 0); return e.el; }));
      s.refresh();
    }
    function nextName(n) {
      const names = new Set(editors.map((e) => { try { return e.value().name; } catch (x) { return ''; } }));
      const m = /^(.*?)(?:-(\d+))?$/.exec(n);
      let i = m[2] ? Number(m[2]) + 1 : 2;
      while (names.has(`${m[1]}-${i}`)) i++;
      return `${m[1]}-${i}`;
    }
    const vs = q.get('vs');
    const tryFrom = q.get('try');
    const s = starter('test', () => {
      const v = f.values();
      const setSpec = {};
      for (const k of ['kind', 'path', 'split', 'splits', 'ids', 'limit', 'max_tokens', 'seed_base', 'suffix', 'image']) if (v[k] !== undefined) setSpec[k] = v[k];
      const p = { set: setSpec, configs: editors.map((e) => e.value()) };
      for (const k of ['concurrency', 'model', 'request_timeout', 'resume']) if (v[k] !== undefined) p[k] = v[k];
      return p;
    }, { started: (k) => go(`#test/${k}${key ? `?vs=${key}` : ''}`) });
    formCard.append(f.el, h('h3', {}, 'Configurations (the first is the baseline)'), cfgBox,
      h('div', { class: 'row' },
        h('button', { class: 'ghost small', type: 'button', onclick: () => addConfig({ name: editors.length ? nextName('config') : 'base' }) }, 'Add configuration'),
        h('button', { class: 'ghost small', type: 'button', onclick: () => addConfig({ name: 'base', reasoning_effort: 'low' }, 0) }, 'Add a baseline (low effort, OFF)')),
      s.el, h('div', { class: 'row' }, s.button, key ? h('span', { class: 'muted' }, `The new job's view compares it with ${short(key)}.`) : null));
    const startConfigs = drafts.test.splice(0);
    if (tryFrom && prev) {
      const src = (pp.configs || []).find((c) => c.name === tryFrom);
      if (src) addConfig({ ...src, name: nextName(src.name) });
    } else if (prev && !startConfigs.length) (pp.configs || []).forEach((c) => addConfig(c));
    startConfigs.forEach((c) => addConfig(c));
    if (!editors.length) addConfig({ name: 'base' });
    s.refresh();
    if (!key) { root.append(await recentJobs('test', (k) => go(`#test/${k}`))); return; }

    // Live tables, with the compared jobs' (finished) rows next to this job's.
    const known = ['config', 'row', 'totals', 'resume'];
    const live = h('div', {});
    let panel;
    const others = [];
    const cmpInput = h('input', { placeholder: 'another test job key', value: '' });
    const view = { a: null, b: null, onlyDiff: true };
    const w = watchIn(root, key, () => { panel.render(); render(); });
    panel = jobPanel(w, { known });
    root.append(panel.el, h('div', { class: 'card row' }, 'Compare with', cmpInput, h('button', { class: 'ghost small', type: 'button', onclick: () => addOther(cmpInput.value.trim()) }, 'Add'),
      h('span', { class: 'muted' }, 'Rows of finished jobs join this table, labelled with the job key.')), live);
    async function addOther(k) {
      if (!k || k === key || others.some((o) => o.key === k)) return;
      try { const j = await loadAll(k); others.push({ key: k, j }); render(); } catch (e) { fail(e); }
    }
    if (vs) addOther(vs);
    function render() {
      // Columns: this job's configs, then each compared job's (prefixed).
      const cols = [];
      const add = (src, prefix, jobKey) => {
        const totals = {}; const rows = {}; const problems = {};
        for (const r of src.of('config')) problems[r.config] = r.problems;
        for (const r of src.of('totals')) totals[r.config] = r;
        for (const r of src.of('row')) (rows[r.config] || (rows[r.config] = {}))[r.id] = r;
        const order = src.of('config').map((r) => r.config);
        for (const c of Object.keys(totals)) if (!order.includes(c)) order.push(c);
        for (const c of order) cols.push({ id: prefix + c, name: c, jobKey, totals: totals[c], rows: rows[c] || {}, problems: problems[c] });
      };
      add(w, '', key);
      for (const o of others) add(o.j, short(o.key) + '/', o.key);
      if (!cols.length) { live.replaceChildren(C.empty('waiting for the first configuration')); return; }
      // The baseline: the compared job's first configuration when this job came from "try next setting".
      const baseline = (vs && cols.find((c) => c.jobKey === vs)) || cols[0];
      const dl = (v, b, lowerBetter) => {
        if (typeof v !== 'number' || typeof b !== 'number' || v === b) return null;
        const better = lowerBetter ? v < b : v > b;
        return h('span', { class: better ? 'better' : 'worse' }, ` (${v > b ? '+' : ''}${fmt(+(v - b).toFixed(3))})`);
      };
      const totalsTable = h('table', {}, h('tr', {}, ['configuration', 'done', 'correct', 'accuracy', 'mean tokens', 'median tokens', 'truncated', 'hesitations', 'seconds', 'errors', ''].map((t, i) => h('th', { class: i && i < 10 ? 'num' : '' }, t))),
        cols.map((c) => {
          const t = c.totals || {}, b = baseline.totals || {};
          const isBase = c === baseline;
          return h('tr', { class: isBase ? 'diff-row' : '' }, h('td', {}, h('strong', {}, c.id), isBase ? h('span', { class: 'muted' }, ' baseline') : null),
            h('td', { class: 'num' }, `${t.n ?? 0}/${c.problems ?? '?'}`),
            h('td', { class: 'num' }, t.correct ?? '-', isBase ? null : dl(t.correct, b.correct)),
            h('td', { class: 'num' }, fmt(t.accuracy), isBase ? null : dl(t.accuracy, b.accuracy)),
            h('td', { class: 'num' }, fmt(t.tokens_mean), isBase ? null : dl(t.tokens_mean, b.tokens_mean, true)),
            h('td', { class: 'num' }, fmt(t.tokens_median), isBase ? null : dl(t.tokens_median, b.tokens_median, true)),
            h('td', { class: 'num' }, t.truncated ?? '-', isBase ? null : dl(t.truncated, b.truncated, true)),
            h('td', { class: 'num' }, fmt(t.hesitations_mean), isBase ? null : dl(t.hesitations_mean, b.hesitations_mean, true)),
            h('td', { class: 'num' }, fmt(t.seconds_mean)), h('td', { class: 'num' + (t.errors ? ' bad' : '') }, t.errors ?? 0),
            h('td', {}, h('button', { class: 'ghost small', type: 'button', title: 'Clone this configuration into the editor above, adjust it and run it',
              onclick: () => { go(`#test/${c.jobKey}?try=${enc(c.name)}`); } }, 'Try next setting')));
        }));
      // Per-problem rows, with a diff between two configurations.
      if (!view.a || !cols.some((c) => c.id === view.a)) view.a = baseline.id;
      if (!view.b || !cols.some((c) => c.id === view.b)) view.b = (cols.find((c) => c.id !== view.a) || cols[0]).id;
      const selA = h('select', {}, cols.map((c) => h('option', { value: c.id }, c.id))); selA.value = view.a;
      const selB = h('select', {}, cols.map((c) => h('option', { value: c.id }, c.id))); selB.value = view.b;
      const only = h('input', { type: 'checkbox', checked: view.onlyDiff });
      selA.addEventListener('change', () => { view.a = selA.value; render(); });
      selB.addEventListener('change', () => { view.b = selB.value; render(); });
      only.addEventListener('change', () => { view.onlyDiff = only.checked; render(); });
      const A = cols.find((c) => c.id === view.a), B = cols.find((c) => c.id === view.b);
      const ids = [...new Set(cols.flatMap((c) => Object.keys(c.rows)))];
      const cell = (r) => {
        if (!r) return h('td', { class: 'muted' }, '-');
        if (r.error) return h('td', { class: 'bad', title: r.error }, 'error');
        return h('td', { class: 'num', title: [`tokens ${r.tokens}`, `reasoning ${r.reasoning_tokens}`, `hesitations ${r.hesitations}`, `finish ${r.finish_reason}`,
          r.got !== undefined ? `got ${r.got} / answer ${r.answer}` : null, r.passed !== undefined ? `passed ${r.passed}/${r.tests}` : null, `${fmt(r.seconds)} s`].filter(Boolean).join('\n') },
        h('span', { class: r.correct ? 'ok' : 'bad' }, r.correct ? 'yes' : 'no'), ` ${r.tokens ?? ''}${r.truncated ? ' T' : ''}`);
      };
      let shown = 0, changed = 0;
      const body = ids.map((id) => {
        const ra = A.rows[id], rb = B.rows[id];
        const differs = ra && rb && (!!ra.correct !== !!rb.correct);
        if (differs) changed++;
        if (view.onlyDiff && !differs) return null;
        shown++;
        const dt = ra && rb && typeof ra.tokens === 'number' && typeof rb.tokens === 'number' ? rb.tokens - ra.tokens : null;
        return h('tr', { class: differs ? 'diff-row' : '' }, h('td', { class: 'mono' }, id), cols.map((c) => cell(c.rows[id])),
          h('td', { class: 'num' + (dt === null ? '' : dt < 0 ? ' better' : dt > 0 ? ' worse' : '') }, dt === null ? '-' : (dt > 0 ? '+' : '') + dt),
          h('td', {}, ra && rb ? (differs ? (rb.correct ? `${B.id} fixes` : `${B.id} breaks`) : '') : ''));
      });
      const rowsTable = h('div', { class: 'scroll' }, h('table', {}, h('tr', {}, h('th', {}, 'problem'), cols.map((c) => h('th', { class: 'num' }, c.id)),
        h('th', { class: 'num' }, `tokens ${B.id} - ${A.id}`), h('th', {}, 'diff')), body.filter(Boolean)));
      const tokBox = h('div', {});
      C.bars(tokBox, { categories: cols.map((c) => c.id), series: [{ name: 'mean tokens', cls: 's1', values: cols.map((c) => c.totals && c.totals.tokens_mean) }],
        yLabel: 'mean tokens', height: 150 });
      const accBox = h('div', {});
      C.bars(accBox, { categories: cols.map((c) => c.id), series: [{ name: 'accuracy', cls: 's3', values: cols.map((c) => c.totals && c.totals.accuracy) }],
        yMin: 0, yMax: 1, yLabel: 'accuracy', height: 150 });
      keepOpen(live, () => {
      live.replaceChildren(
        h('div', { class: 'card' }, h('h3', {}, 'Per configuration'), h('div', { class: 'scroll' }, totalsTable),
          h('div', { class: 'grid2' }, accBox, tokBox)),
        h('div', { class: 'card' }, h('h3', {}, 'Per problem'),
          h('div', { class: 'row' }, 'diff', selA, 'against', selB, h('label', { class: 'check row' }, only, 'only problems where correctness differs'),
            h('span', { class: 'muted' }, `${changed} differ, ${shown} shown of ${ids.length}`)), rowsTable,
          h('div', { class: 'muted' }, 'Cell: correct, tokens, T = truncated. Hover for details. Full answers: the job\'s generations folder.')));
      });
    }
    render();
  }

  // 5. Pipeline -----------------------------------------------------------------
  async function viewPipeline(key) {
    const root = main();
    root.append(h('h2', {}, 'Pipeline'));
    const formCard = h('details', { class: 'card', open: !key }, h('summary', {}, key ? 'Edit and run a pipeline' : 'Load, edit and run a pipeline'));
    root.append(formCard);
    const text = h('textarea', { spellcheck: 'false', placeholder: '{"steps": [{"operation": "test", "params": {...}}, ...]}' });
    text.style.minHeight = '120px';
    const file = h('input', { type: 'file', accept: '.json,application/json' });
    const stepsBox = h('div', {});
    let doc = null;
    if (key) {
      try { const j = await api('GET', EP.job(key, 0)); doc = j.params; text.value = json(doc); } catch (e) { fail(e); }
    }
    file.addEventListener('change', () => {
      const fl = file.files[0]; if (!fl) return;
      const r = new FileReader();
      r.onload = () => { text.value = String(r.result); parse(); };
      r.readAsText(fl);
    });
    function parse() {
      text.classList.remove('invalid');
      try { doc = JSON.parse(text.value || '{}'); } catch (e) { text.classList.add('invalid'); stepsBox.replaceChildren(h('div', { class: 'bad' }, e.message)); return; }
      // A file may hold {"operation": "pipeline", "params": {...}} or the params themselves.
      const params = doc && doc.operation === 'pipeline' && doc.params ? doc.params : doc;
      const steps = (params && params.steps) || [];
      stepsBox.replaceChildren(h('h4', {}, `${steps.length} steps`), steps.map((st, i) => {
        const ta = h('textarea', { spellcheck: 'false' });
        ta.value = json(st.params || {});
        ta.addEventListener('change', () => {
          try { st.params = JSON.parse(ta.value); ta.classList.remove('invalid'); text.value = json(doc); } catch (e) { ta.classList.add('invalid'); }
        });
        return h('details', { class: 'card' }, h('summary', {}, `${i}. ${st.operation || st.op || '?'}${st.name ? ' - ' + st.name : ''}`), ta);
      }));
    }
    text.addEventListener('change', parse);
    if (doc) parse();
    const run = h('button', { type: 'button', onclick: async () => {
      parse(); if (!doc) return;
      const params = doc.operation === 'pipeline' && doc.params ? doc.params : doc;
      if (!window.confirm(`Start a pipeline of ${(params.steps || []).length} steps on the gufo server?`)) return;
      try { const k = await startJob('pipeline', params); go(`#pipeline/${k}`); } catch (e) { fail(e); }
    } }, 'Run pipeline');
    formCard.append(h('div', { class: 'row' }, 'Load a file (for example tools/steer/examples/r10.json):', file),
      h('label', { class: 'field' }, 'Pipeline JSON (edit freely; step params below update it)', text), stepsBox, h('div', { class: 'row' }, run));
    if (!key) { root.append(await recentJobs('pipeline', (k) => go(`#pipeline/${k}`))); return; }

    const live = h('div', {});
    let panel;
    const childState = {};
    const w = watchIn(root, key, () => { panel.render(); render(); });
    panel = jobPanel(w, {});
    root.append(live, panel.el);
    async function refreshChildren(keys) {
      for (const k of keys) {
        if (childState[k] && FINISHED.includes(childState[k].status)) continue;
        try { childState[k] = await api('GET', EP.job(k, 1e15)); } catch (e) { childState[k] = { status: 'unknown', error: e.message }; }
      }
    }
    let childTimer = null;
    function render() {
      const s = w.state || {};
      const steps = ((s.params && s.params.steps) || []);
      // Step results: any result naming a step index and a child job.
      const byStep = {};
      for (const r of w.of('pipeline_step')) {
        const i = r.index;
        if (typeof i !== 'number') continue;
        byStep[i] = Object.assign(byStep[i] || {}, r);
      }
      const children = jobLinks(w.results);
      clearTimeout(childTimer);
      childTimer = setTimeout(() => refreshChildren(children).then(() => renderTable()), 0);
      function renderTable() {
        live.replaceChildren(h('div', { class: 'card' }, h('h3', {}, 'Steps'), h('table', {},
          h('tr', {}, ['#', 'operation', 'status', 'job', 'progress', 'output'].map((t) => h('th', {}, t))),
          steps.map((st, i) => {
            const r = byStep[i] || {};
            const child = r.job || r.child_job || r.child;
            const cs = child ? childState[child] : null;
            const status = (cs && cs.status) || r.status || 'pending';
            return h('tr', {}, h('td', {}, i), h('td', {}, `${st.name || r.name || ''} (${st.operation || r.operation || '?'})`, r.reused ? h('span', { class: 'muted' }, ' reused') : null,
              r.error ? h('div', { class: 'bad' }, r.error) : null), h('td', {}, statusSpan(status)),
              h('td', {}, child ? h('a', { href: `#job/${child}` }, short(child)) : '-'),
              h('td', { class: 'muted' }, cs && cs.progress ? `${cs.progress.done}/${cs.progress.total} ${cs.progress.step || ''}` : ''),
              h('td', {}, r.output || (cs && cs.output) ? h('details', {}, h('summary', {}, 'output'), h('pre', {}, json(r.output || cs.output))) : ''));
          })),
        children.length && !Object.keys(byStep).length ? h('div', {}, 'Child jobs: ', children.map((k) => h('a', { href: `#job/${k}` }, short(k) + ' '))) : null,
        h('div', { class: 'muted' }, 'Open a step\'s job to see its own live view.')));
      }
      renderTable();
    }
    render();
    watches.push({ stop: () => clearTimeout(childTimer) });
  }

  // 6. History --------------------------------------------------------------------
  async function viewHistory(key) {
    const root = main();
    root.append(h('h2', {}, 'History'));
    const filter = h('select', {}, h('option', { value: '' }, 'all operations'));
    const statusF = h('select', {}, h('option', { value: '' }, 'any status'), ['queued', 'running', 'done', 'failed', 'cancelled'].map((s) => h('option', { value: s }, s)));
    const tableBox = h('div', { class: 'card' });
    const detail = h('div', {});
    root.append(h('div', { class: 'row card' }, filter, statusF, h('button', { class: 'ghost small', type: 'button', onclick: refresh }, 'Refresh')), detail, tableBox);
    let list = [];
    filter.addEventListener('change', draw); statusF.addEventListener('change', draw);
    async function refresh() {
      try { list = listOf(await api('GET', EP.jobs)); } catch (e) { fail(e); return; }
      const ops = [...new Set(list.map((j) => j.operation))].sort();
      const cur = filter.value;
      filter.replaceChildren(h('option', { value: '' }, 'all operations'), ops.map((o) => h('option', { value: o }, o)));
      filter.value = cur;
      draw();
    }
    function draw() {
      const rows = list.filter((j) => (!filter.value || j.operation === filter.value) && (!statusF.value || j.status === statusF.value));
      tableBox.replaceChildren(rows.length ? h('div', { class: 'scroll' }, h('table', {},
        h('tr', {}, ['job', 'operation', 'status', 'params', 'created', 'started', 'ran', 'step', ''].map((t) => h('th', {}, t))),
        rows.map((j) => h('tr', { class: j.key === key ? 'selected' : '' },
          h('td', {}, h('a', { href: `#history/${j.key}` }, h('code', {}, short(j.key)))), h('td', {}, j.operation), h('td', {}, statusSpan(j.status)),
          h('td', { class: 'mono', title: json(j.params) }, clip(JSON.stringify(j.params), 90)),
          h('td', { class: 'muted' }, time(j.created_at)), h('td', { class: 'muted' }, time(j.started_at)),
          h('td', { class: 'num' }, j.started_at ? duration(j.started_at, FINISHED.includes(j.status) ? j.updated_at : null) : '-'),
          h('td', { class: 'muted' }, (j.progress && j.progress.step) || ''),
          h('td', {}, h('div', { class: 'row' },
            h('a', { href: `#job/${j.key}` }, 'open'),
            h('button', { class: 'ghost small', type: 'button', onclick: () => rerun(j) }, 'Re-run'),
            h('button', { class: 'danger small', type: 'button', onclick: async () => {
              await cancelOrDelete(j.key, j.operation);
              refresh();
            } }, FINISHED.includes(j.status) ? 'Delete' : 'Cancel'))))))) : C.empty('no jobs'));
    }
    await refresh();
    if (key) {
      const w = watchIn(root, key, () => { panel.render(); extras(); });
      const panel = jobPanel(w, {});
      const more = h('div', {});
      detail.append(panel.el, more);
      function extras() {
        const s = w.state; if (!s) return;
        const parts = [];
        const profs = w.of('profile');
        if (profs.length) {
          parts.push(h('h4', {}, 'Profiles'), profs.map((p) => h('div', {}, h('div', { class: 'row' }, h('strong', {}, p.name), h('code', { class: 'muted' }, p.path || ''),
            h('button', { class: 'small', type: 'button', onclick: () => upload(p.kind || 'abliteration', p.name, { job: key, profile: p.name }) }, 'Upload to gufo')),
          p.path ? h('details', {}, h('summary', {}, 'Promote into the runtime profile directory (a production change)'), promoteBox(p.name, p.path)) : null)));
        }
        const sets = w.of('objective_set');
        if (sets.length) {
          const keys = [...new Set(sets.flatMap((x) => Object.keys(x.counts || {})))];
          parts.push(h('h4', {}, 'Objective rows per set'), h('div', { class: 'scroll' }, h('table', {},
            h('tr', {}, h('th', {}, 'set'), h('th', {}, 'kind'), keys.map((k) => h('th', { class: 'num' }, k))),
            sets.map((x) => h('tr', {}, h('td', {}, x.set), h('td', {}, `${x.kind}${x.effort ? ' ' + x.effort : ''}`),
              keys.map((k) => h('td', { class: 'num' }, fmt((x.counts || {})[k]))))))),
          sets.map((x) => h('details', {}, h('summary', {}, `${x.set}: examples and cost`), h('pre', {}, json({ examples: x.examples, cost: x.cost, neutral_reasons: x.neutral_reasons })))));
        }
        for (const d of w.of('dataset')) parts.push(h('div', {}, 'dataset ', h('a', { href: `#datasets/${enc(d.name)}` }, d.name), ` (${d.kind}, ${d.records} records)`));
        const kids = jobLinks(w.results).filter((k) => k !== key);
        if (kids.length) parts.push(h('div', {}, 'Related jobs: ', kids.map((k) => h('a', { href: `#job/${k}` }, short(k) + ' '))));
        const special = { fit: 'run', train: 'train', test: 'test', pipeline: 'pipeline' }[s.operation];
        if (special) parts.push(h('div', {}, h('a', { href: `#${special}/${key}` }, `Open the ${special} view with charts`)));
        more.replaceChildren(parts.length ? h('div', { class: 'card' }, parts) : '');
      }
    }
  }

  // Boot --------------------------------------------------------------------------
  function theme() {
    const btn = $('#theme');
    let saved = null;
    try { saved = localStorage.getItem('steer-theme'); } catch (e) { saved = null; }
    if (saved) document.documentElement.dataset.theme = saved;
    btn.addEventListener('click', () => {
      const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark'
        : window.matchMedia('(prefers-color-scheme: dark)').matches;
      const next = dark ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('steer-theme', next); } catch (e) { /* private window */ }
    });
  }
  theme();
  window.addEventListener('hashchange', route);
  header();
  setInterval(header, 15000);
  route();
})();

// Deterministic browser-controller tests against the actual live.html script.
// Node builtins only: no server, browser, real timer, or npm install is needed.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.resolve(__dirname, '../tetris_rl/live.html'), 'utf8');
const source = html.split('<script>')[1].split('</script>')[0];

function state(overrides = {}) {
  return {board: Array.from({length: 24}, () => Array(10).fill(0)),
    current: 'T', hold: null, hold_available: true, next: ['I', 'J', 'L', 'O', 'S'],
    lines: 0, pieces: 0, input_ticks: 0, game_time_seconds: 0, done: false, reason: null,
    active: {piece: 'T', rotation: 0, x: 3, y: 2, cells: [[4, 2], [3, 3], [4, 3], [5, 3]]},
    ghost_cells: [[4, 22], [3, 23], [4, 23], [5, 23]], ...overrides};
}

function placement({done = false} = {}) {
  // Empty hold changes current and public next immediately. Lock consumes no input tick.
  const held = {current: 'I', hold: 'T', hold_available: false, next: ['J', 'L', 'O', 'S', 'Z']};
  const frames = ['hold', 'left', 'hard_drop'].map((input, i) => ({input,
    state: state({...held, input_ticks: i + 1, game_time_seconds: (i + 1) / 60,
      active: {piece: 'I', cells: [[2, 20], [3, 20], [4, 20], [5, 20]]}})}));
  frames.push({input: 'lock', state: state({current: done ? null : 'J', hold: 'T', pieces: 1,
    next: ['L', 'O', 'S', 'Z', 'T'], input_ticks: 3, game_time_seconds: 3 / 60,
    done, reason: done ? 'success' : null, lines: done ? 40 : 0,
    active: done ? null : {piece: 'J', cells: [[3, 2], [3, 3], [4, 3], [5, 3]]}})});
  return {session_id: 'game-1', revision: 1, frames, state: frames.at(-1).state,
    decision: {piece: 'I', hold: true, inputs: ['hold', 'left', 'hard_drop'], inference_ms: 1.25},
    stats: {total_inference_ms: 1.25, placements: 1}};
}

const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
async function player({defaultSeed = null, statusFailure = false, holdStatus = false} = {}) {
  let now = 0;
  const elements = new Map();
  const makeElement = () => ({textContent: '', value: '', children: [], listeners: {}, hidden: false,
    disabled: false, tagName: 'DIV', style: {setProperty(k, v) { this[k] = v; }},
    setAttribute(k, v) { this[k] = v; }, replaceChildren(...children) { this.children = children; },
    append(...children) { this.children.push(...children); },
    addEventListener(name, fn) { this.listeners[name] = fn; }});
  for (const m of html.matchAll(/id="([^"]+)"/g)) elements.set(m[1], makeElement());
  elements.get('board').getContext = () => new Proxy({}, {get: () => () => {}});
  elements.get('speed').value = '1'; elements.get('timing-mode').value = 'view';
  const document = {hidden: false, listeners: {}, getElementById(id) {
    assert(elements.has(id), `Missing HTML element ${id}`); return elements.get(id);
  }, createElement: makeElement, addEventListener(name, fn) { this.listeners[name] = fn; }};
  const requests = [];
  function fetch(url, options) {
    const request = {url, options, body: options.body ? JSON.parse(options.body) : undefined};
    requests.push(request);
    return new Promise((resolve, reject) => {
      request.respond = (data, status = 200) => resolve({ok: status >= 200 && status < 300, status,
        json: async () => data});
      request.reject = reject;
      if (url === '/api/status' && !holdStatus) {
        if (statusFailure) request.respond({error: 'Model temporarily unavailable'}, 503);
        else request.respond({ready: true, default_seed: defaultSeed,
          model: {name: 'test-model'}, input_hz: 60, target_lines: 40});
      }
    });
  }
  const context = vm.createContext({document, performance: {now: () => now}, fetch,
    AbortSignal: {timeout: () => ({})}, requestAnimationFrame() {}});
  const evaluate = code => vm.runInContext(code, context);
  evaluate(source); await settle();
  const count = url => requests.filter(r => r.url === url).length;
  const latest = url => requests.filter(r => r.url === url).at(-1);
  assert.equal(count('/api/status'), 1, 'Bootstrap must inspect server defaults');
  if (!statusFailure && !holdStatus) {
    assert.equal(count('/api/new'), 1, 'Bootstrap must create exactly one game');
    latest('/api/new').respond({session_id: 'game-1', revision: 0, seed: defaultSeed ?? 3000000,
      state: state(), model: {name: 'test-model'}});
    await settle();
  }
  return {evaluate, elements, document, requests, count, latest,
    tick(ms) { now += ms; evaluate('tick(performance.now())'); },
    click(id) { const e = elements.get(id); if (!e.disabled) e.onclick?.(); },
    select(id, value) { const e = elements.get(id); e.value = String(value); e.onchange(); },
    cursor: () => evaluate('cursor'), playing: () => evaluate('playing'),
    async respond(data = placement(), status = 200) { latest('/api/step').respond(data, status); await settle(); }};
}

(async () => {
  {
    const p = await player({defaultSeed: 3456789});
    assert.equal(p.elements.get('seed').value, '3456789');
    assert.deepEqual(p.latest('/api/new').body, {seed: 3456789}, 'CLI seed must become initial game seed');
    assert.equal(p.count('/api/step'), 0, 'Bootstrap must not perform a policy placement');
  }
  {
    const p = await player({holdStatus: true});
    for (const id of ['play', 'step', 'original', 'new']) {
      assert.equal(p.elements.get(id).disabled, true, `${id} must be disabled while status is pending`);
      p.click(id);
    }
    assert.equal(p.count('/api/new'), 0, 'Cannot create competing games during initialization');
    p.latest('/api/status').respond({ready: true, default_seed: null, model: {name: 'ready-model'}});
    await settle(); assert.equal(p.count('/api/new'), 1);
    for (const id of ['play', 'step', 'original', 'new']) assert.equal(p.elements.get(id).disabled, true);
    p.latest('/api/new').respond({session_id: 'game-ready', revision: 0, seed: 1, state: state()});
    await settle();
    for (const id of ['play', 'step', 'original', 'new']) assert.equal(p.elements.get(id).disabled, false);
  }
  {
    const p = await player({statusFailure: true});
    assert.equal(p.count('/api/new'), 0); assert.equal(p.evaluate('fault'), true);
    assert.equal(p.elements.get('error').hidden, false); assert.equal(p.elements.get('new').disabled, false);
    assert.match(p.elements.get('error').textContent, /Model temporarily unavailable/);
    p.click('new'); assert.equal(p.count('/api/new'), 1);
    p.latest('/api/new').respond({session_id: 'recovered', revision: 0, seed: 2, state: state()});
    await settle();
    assert.equal(p.evaluate('fault'), false); assert.equal(p.elements.get('error').hidden, true);
    assert.equal(p.elements.get('play').disabled, false); assert.equal(p.evaluate('session'), 'recovered');
  }
  {
    const p = await player(); p.tick(60000);
    assert.equal(p.playing(), false); assert.equal(p.count('/api/step'), 0, 'No inference before play');
    p.click('play'); p.tick(60000); p.tick(60000);
    assert.equal(p.count('/api/step'), 1, 'Only one placement request may be in flight');
    assert.deepEqual(p.latest('/api/step').body, {session_id: 'game-1', revision: 0});
  }
  for (const rate of [.25, .5, 1, 2, 4]) {
    const p = await player(); p.select('speed', rate); p.click('play'); await p.respond();
    p.tick(120 / rate - .001); assert.equal(p.cursor(), 0, `view ${rate}x early frame`);
    p.tick(.002); assert.equal(p.cursor(), 1, `view ${rate}x frame timing`);
    assert.equal(p.count('/api/step'), 1, 'Must finish current path before asking for next placement');
  }
  for (const rate of [.25, .5, 1]) {
    const p = await player(); p.click('original'); p.select('speed', rate); await p.respond();
    p.tick(50 / rate - .001); assert.equal(p.cursor(), 2); assert.equal(p.evaluate('state.pieces'), 0);
    p.tick(.002); assert.equal(p.evaluate('state.pieces'), 1, 'Lock shares hard-drop tick in original mode');
    assert.equal(p.cursor(), 4); assert.equal(p.count('/api/step'), 1);
    assert.equal(p.elements.get('game-time').textContent, '0.05');
  }
  {
    const p = await player(); p.click('play'); await p.respond(); p.tick(40); p.click('play');
    p.tick(10000); assert.equal(p.cursor(), 0); p.click('play');
    p.tick(79.999); assert.equal(p.cursor(), 0); p.tick(.002); assert.equal(p.cursor(), 1);
    assert.equal(p.count('/api/step'), 1, 'Resume preserves the existing placement');
  }
  {
    const p = await player(); p.click('play'); await p.respond(); p.tick(60);
    p.select('timing-mode', 'original');
    assert.ok(Math.abs(p.evaluate('elapsed') - 1000 / 120) < 1e-9, 'Mode switch preserves half-frame progress');
    p.tick(1000 / 120 - .001); assert.equal(p.cursor(), 0); p.tick(.002); assert.equal(p.cursor(), 1);
  }
  {
    const p = await player(); p.click('play');
    p.tick(5000); p.select('timing-mode', 'original');
    assert.equal(p.evaluate('elapsed'), 0); assert.equal(p.count('/api/step'), 1);
    await p.respond(); p.tick(1000 / 60 - .001); assert.equal(p.cursor(), 0);
    p.tick(.002); assert.equal(p.cursor(), 1);
    p.tick(1000); assert.equal(p.evaluate('queue.length'), 0);
    p.select('timing-mode', 'view'); assert.equal(p.count('/api/step'), 2);
    assert.equal(p.evaluate('elapsed'), 0, 'Empty queue mode switch starts a fresh frame');
    await p.respond(); p.tick(119.999); assert.equal(p.cursor(), 0); p.tick(.002); assert.equal(p.cursor(), 1);
  }
  {
    const p = await player(); p.click('play'); p.tick(50); p.click('play'); await p.respond();
    p.tick(10000); assert.equal(p.cursor(), 0); assert.equal(p.evaluate('state.current'), 'T');
    assert.equal(p.playing(), false, 'A delayed response must not resume a paused game');
    p.click('play'); p.tick(119); assert.equal(p.cursor(), 0); p.tick(1); assert.equal(p.cursor(), 1);
  }
  {
    const p = await player(); p.click('step'); await p.respond(); p.tick(10000);
    assert.equal(p.evaluate('state.pieces'), 1); assert.equal(p.playing(), false);
    p.tick(10000); assert.equal(p.count('/api/step'), 1, 'Single step must not precompute the next piece');
    p.click('step'); assert.equal(p.count('/api/step'), 2);
    assert.deepEqual(p.latest('/api/step').body, {session_id: 'game-1', revision: 1});
  }
  {
    const p = await player(); p.click('play'); await p.respond(); p.tick(120);
    assert.equal(p.elements.get('current-name').textContent, 'I 미노');
    assert.equal(p.elements.get('current-shape')['aria-label'], 'I 미노');
    assert.equal(p.elements.get('hold-name').textContent, 'T · 사용됨');
    assert.equal(p.elements.get('next').children[0].children[1].textContent, 'J');
    p.tick(360); assert.equal(p.elements.get('current-name').textContent, 'J 미노');
    assert.equal(p.elements.get('hold-name').textContent, 'T');
    assert.equal(p.elements.get('next').children[0].children[1].textContent, 'L');
  }
  for (const reject of [false, true]) {
    const p = await player(); p.click('play'); const old = p.latest('/api/step');
    p.elements.get('seed').value = '4000000'; p.click('new');
    assert.deepEqual(p.latest('/api/new').body, {seed: 4000000});
    p.latest('/api/new').respond({session_id: 'game-2', revision: 0, seed: 4000000, state: state({current: 'Z'})});
    await settle();
    if (reject) old.reject(Error('obsolete request failure')); else old.respond(placement());
    await settle(); p.tick(10000);
    assert.equal(p.evaluate('session'), 'game-2'); assert.equal(p.evaluate('revision'), 0);
    assert.equal(p.evaluate('state.current'), 'Z'); assert.equal(p.evaluate('queue.length'), 0);
    assert.equal(p.evaluate('fault'), false, 'Obsolete requests must not overwrite or fault the new game');
  }
  for (const kind of ['server', 'network', 'empty']) {
    const p = await player(); p.click('play');
    if (kind === 'server') await p.respond({error: 'Game expired'}, 404);
    if (kind === 'network') { p.latest('/api/step').reject(Error('Network failed')); await settle(); }
    if (kind === 'empty') await p.respond({...placement(), frames: []});
    assert.equal(p.playing(), false); assert.equal(p.evaluate('fault'), true);
    assert.equal(p.elements.get('error').hidden, false); assert.match(p.elements.get('error').textContent, /요청 실패:/);
    assert.equal(p.elements.get('play').disabled, true); assert.equal(p.elements.get('new').disabled, false);
    p.tick(60000); assert.equal(p.count('/api/step'), 1, 'Errors must stop automatic requests');
  }
  {
    const p = await player(); p.click('play'); await p.respond(); p.tick(60);
    p.document.hidden = true; p.document.listeners.visibilitychange(); p.tick(60000);
    assert.equal(p.playing(), false); assert.equal(p.cursor(), 0);
    assert.equal(p.count('/api/step'), 1); p.document.hidden = false; p.document.listeners.visibilitychange();
    assert.equal(p.playing(), false, 'Visibility restoration must not silently resume');
  }
  {
    const p = await player(); p.click('play'); await p.respond(placement({done: true})); p.tick(480);
    assert.equal(p.playing(), false); assert.equal(p.elements.get('current-name').textContent, '—');
    assert.equal(p.elements.get('current-label').textContent, '플레이 종료');
    assert.equal(p.elements.get('current-shape').children.length, 0);
    assert.equal(p.elements.get('overlay').hidden, false); assert.equal(p.elements.get('play').disabled, true);
    p.tick(10000); assert.equal(p.count('/api/step'), 1, 'Terminal states must not request another decision');
  }
  console.log('PASS: live bootstrap/default seed/recovery, initial controls, rates/mode changes, zero-tick lock, pause/resume, delayed HTTP, one-placement stepping, bounded inference, stale responses, hold/next sync, errors, hidden-tab pause, and terminal state.');
})().catch(error => { console.error(error); process.exitCode = 1; });

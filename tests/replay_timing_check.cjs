// Deterministic clock checks against the actual generated replay script.
// Run with Node.js; no browser or additional npm packages are required.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'artifacts/replays/model_2000000.html'), 'utf8');
const replay = JSON.parse(fs.readFileSync(path.join(root, 'artifacts/replays/model_2000000.json'), 'utf8'));
const source = html.split('<script>')[1].split('</script>')[0];

function player(data = replay) {
  let now = 0;
  const elements = new Map();
  const makeElement = () => ({textContent: '', value: '', children: [], listeners: {},
    style: {setProperty(k, v) { this[k] = v; }},
    setAttribute(k, v) { this[k] = v; },
    replaceChildren(...children) { this.children = children; },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    click() { this.onclick?.(); }});
  for (const m of html.matchAll(/id="([^"]+)"/g)) elements.set(m[1], makeElement());
  const ctx = new Proxy({}, {get: () => () => {}});
  elements.get('board').getContext = () => ctx;
  elements.get('speed').value = '1';
  elements.get('timing-mode').value = 'view';
  const document = {hidden: false, listeners: {}, getElementById: id => {
    assert(elements.has(id), `Missing HTML element ${id}`); return elements.get(id);
  }, createElement: makeElement, addEventListener(name, fn) { this.listeners[name] = fn; }};
  const context = vm.createContext({document, performance: {now: () => now}, requestAnimationFrame() {}});
  const evaluate = code => vm.runInContext(code, context);
  evaluate(source.replace(/^const data=.*;$/m, `const data=${JSON.stringify(data)};`));
  return {evaluate, elements, document,
    tick(ms) { now += ms; evaluate('animate(performance.now())'); },
    click(id) { elements.get(id).click(); },
    select(id, value) { const e = elements.get(id); e.value = String(value); e.listeners.change(); },
    index() { return evaluate('index'); }, playing() { return evaluate('playing'); }};
}

for (const rate of [.25, .5, 1, 2, 4, 8]) {
  const p = player(); p.select('speed', rate); p.click('play');
  p.tick(600 / rate - .001); assert.equal(p.index(), 0);
  p.tick(.002); assert.equal(p.index(), 1);
}
const totalMs = (replay.frames.at(-1).input_ticks - replay.frames[0].input_ticks) * 1000 / 60;
for (const rate of [.25, .5, 1, 2, 4, 8]) {
  const p = player(); p.click('original-speed'); p.select('speed', rate);
  p.tick(totalMs / rate - .001); assert.equal(p.index(), replay.actions.length - 1);
  p.tick(.002); assert.equal(p.index(), replay.actions.length);
  assert.equal(p.playing(), false); assert.equal(p.elements.get('play').textContent, '재생');
  p.click('play'); assert.equal(p.index(), 0); assert.equal(p.playing(), true);
}
{
  const p = player(); p.click('play'); p.tick(200); p.click('play');
  p.tick(10000); assert.equal(p.index(), 0); p.click('play');
  p.tick(399); assert.equal(p.index(), 0); p.tick(1); assert.equal(p.index(), 1);
}
{
  const p = player(); p.click('play'); p.tick(300); p.select('speed', .5);
  p.tick(599); assert.equal(p.index(), 0); p.tick(1); assert.equal(p.index(), 1);
}
{
  const p = player(); p.click('play'); p.tick(300); p.select('timing-mode', 'recorded');
  const halfFirst = (replay.frames[1].input_ticks - replay.frames[0].input_ticks) * 1000 / 120;
  p.tick(halfFirst - .001); assert.equal(p.index(), 0); p.tick(.002); assert.equal(p.index(), 1);
}
{
  const p = player(); p.click('play'); p.tick(500); p.evaluate('seek(50)');
  p.tick(599); assert.equal(p.index(), 50); p.tick(1); assert.equal(p.index(), 51);
  p.select('speed', 8); p.click('original-speed');
  assert.equal(p.evaluate('timingMode'), 'recorded'); assert.equal(p.evaluate('playbackRate'), 1);
  assert.equal(p.index(), 51); assert.equal(p.evaluate('elapsed'), 0);
}
{
  const p = player(); p.click('original-speed'); p.tick(1000);
  assert.equal(p.index(), replay.frames.findLastIndex(f => f.input_ticks <= 60));
  p.document.hidden = true; p.document.listeners.visibilitychange();
  assert.equal(p.playing(), false);
}
{
  const data = structuredClone(replay);
  for (const f of data.frames) { delete f.input_ticks; delete f.game_time_seconds; }
  const p = player(data); assert.equal(p.evaluate('hasRecordedTiming'), true);
  p.click('original-speed'); p.tick(totalMs + 1); assert.equal(p.playing(), false);
  for (const a of data.actions) a.inputs = [];
  const missing = player(data); assert.equal(missing.evaluate('hasRecordedTiming'), false);
  assert.equal(missing.elements.get('original-speed').disabled, true);
  missing.click('original-speed'); assert.equal(missing.playing(), false);
}
console.log('PASS: six playback rates, recorded duration, pause/resume, seek, mode/rate changes, catch-up, replay restart, and missing timing.');
console.log(`Recorded playback: 1x ${(totalMs/1000).toFixed(4)}s; 0.5x ${(totalMs/500).toFixed(4)}s; 0.25x ${(totalMs/250).toFixed(4)}s.`);

'use strict';
/* Opening scene. Pandora flies through a neural network: the neurons she touches fire, their pulses
   run forward layer by layer, and each one that reaches the output layer leaves as a token. She
   speeds up while the counter climbs to the measured speed. Decorative only: no requests, no inference. */
(() => {
  const intro = document.querySelector('.intro');
  if (!intro) return;
  const SVG = 'http://www.w3.org/2000/svg';
  const net = intro.querySelector('#intro-net');
  const trail = intro.querySelector('#intro-trail');
  const tokens = intro.querySelector('#intro-tokens');
  const flyer = intro.querySelector('#intro-flyer');
  const counter = intro.querySelector('[data-count]');
  const target = Number(counter.dataset.count);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const WORDS = ['Local', 'AI', 'on', 'the', 'laptop', 'you', 'already', 'have', '.', 'No', 'big', 'GPU',
                 'needed', ':', 'just', 'Pithos', '.'];
  const ENTER = 500, START = 1700, RAMP = 3000, CALM = 7500; // ms from load
  const lerp = (a, b, k) => a + (b - a) * Math.min(Math.max(k, 0), 1);
  const el = (name, attrs, parent) => {
    const e = document.createElementNS(SVG, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.append(e);
    return e;
  };

  // ---- the network: five fully connected layers, left (input) to right (output)
  const CY = 220;
  const layers = [[50, 4], [165, 6], [280, 7], [395, 6], [510, 3]].map(([x, n], li) =>
    Array.from({length: n}, (_, i) => ({x, y: CY + (i - (n - 1) / 2) * 54, layer: li, lit: -1e9})));
  const edges = el('g', {class: 'net-edges'}, net);
  layers.slice(0, -1).forEach((row, i) => {
    for (const a of row) for (const b of layers[i + 1]) el('line', {x1: a.x, y1: a.y, x2: b.x, y2: b.y}, edges);
  });
  const pulses = el('g', {}, net);
  for (const row of layers) for (const p of row) {
    p.glow = el('circle', {class: 'net-glow', cx: p.x, cy: p.y, r: 13}, net);
    el('circle', {class: 'net-node', cx: p.x, cy: p.y, r: 5.5}, net);
  }
  const nodes = layers.flat();

  // Pandora's flight: left to right across the network on a wave, a new wave for every crossing
  const X0 = -70, X1 = 640;
  let wave = {amp: 95, len: 520, shift: 0};
  const newWave = () => { wave = {amp: lerp(70, 130, Math.random()), len: lerp(420, 640, Math.random()),
                                   shift: Math.random() * Math.PI * 2}; };
  const waveY = x => CY + wave.amp * Math.sin(2 * Math.PI * (x - X0) / wave.len + wave.shift);
  const place = (x, y, rot, s) =>
    flyer.setAttribute('transform', `translate(${x} ${y}) rotate(${rot}) scale(${s})`);

  const showFinal = () => { counter.textContent = String(target); };
  if (reduced.matches) {
    // a still frame: Pandora in mid-flight, a chain of neurons lit up behind her
    place(222, 160, -8, 1);
    for (const p of [layers[0][1], layers[1][2], layers[2][2], layers[3][3], layers[4][1]]) p.glow.style.opacity = '.75';
    showFinal();
    return;
  }

  // ---- pulses: a fired neuron passes the signal to one or two neurons of the next layer
  const flash = (p, delay = 0) => p.glow.animate([{opacity: 0}, {opacity: .95}, {opacity: 0}],
    {duration: 520, delay, easing: 'ease-out'});
  let hop = 320;
  function fire(p, fan) {
    flash(p);
    if (p.layer === layers.length - 1) { emit(p); return; }
    const next = layers[p.layer + 1];
    for (let i = 0; i < fan; i++) {
      if (pulses.childElementCount > 120) return;
      const b = next[Math.floor(Math.random() * next.length)];
      const dot = el('circle', {class: 'net-pulse', r: 3, cx: 0, cy: 0}, pulses);
      dot.animate([{transform: `translate(${p.x}px,${p.y}px)`}, {transform: `translate(${b.x}px,${b.y}px)`}],
                  {duration: hop, easing: 'ease-in'}).onfinish = () => { dot.remove(); fire(b, 1); };
    }
  }

  let word = 0, lastToken = 0;
  function emit(p) {
    const now = performance.now();
    if (tokens.childElementCount > 14 || now - lastToken < 150) return;
    lastToken = now;
    const tint = Math.random();
    const g = el('g', {class: 'intro-token' + (tint < .2 ? ' hot' : tint < .35 ? ' warm' : '')}, tokens);
    const rect = el('rect', {}, g);
    const text = el('text', {'text-anchor': 'middle', y: 4}, g);
    text.textContent = WORDS[word++ % WORDS.length];
    const w = Math.max(22, text.getComputedTextLength() + 16);
    rect.setAttribute('x', String(-w / 2)); rect.setAttribute('y', '-11');
    rect.setAttribute('width', String(w)); rect.setAttribute('height', '22'); rect.setAttribute('rx', '7');
    const angle = lerp(-0.35, -1.45, Math.random()), reach = lerp(80, 150, Math.random());
    const dx = Math.cos(angle) * reach, dy = Math.sin(angle) * reach;
    g.animate([
      {transform: `translate(${p.x}px,${p.y}px) scale(.3)`, opacity: 0},
      {transform: `translate(${p.x + dx * .2}px,${p.y + dy * .2}px) scale(1)`, opacity: 1, offset: .2},
      {transform: `translate(${p.x + dx * .8}px,${p.y + dy * .8}px) scale(1)`, opacity: .9, offset: .75},
      {transform: `translate(${p.x + dx}px,${p.y + dy}px) scale(.95)`, opacity: 0}
    ], {duration: 1500, easing: 'cubic-bezier(.25,.6,.35,1)'}).onfinish = () => g.remove();
  }

  // ---- the flight
  const TRAIL = 16;
  const ghosts = Array.from({length: TRAIL}, (_, i) =>
    el('circle', {class: 'intro-ghost', r: lerp(14, 2, i / TRAIL), opacity: 0}, trail));
  const history = [];
  const t0 = performance.now();
  let x = X0, last = t0, visible = true, frame = 0;
  const paused = () => document.documentElement.classList.contains('inference-motion-paused');

  function tick(now) {
    frame = 0;
    if (!visible || document.hidden || paused()) return;
    const age = now - t0, dt = Math.min(now - last, 50) / 1000;
    last = now;
    if (age >= ENTER) {
      const ramp = (age - START) / RAMP, calm = age >= CALM;
      const speed = calm ? 210 : age < START ? 170 : lerp(170, 430, ramp); // px per second
      hop = calm ? 300 : lerp(320, 150, ramp);
      x += speed * dt;
      if (x > X1) { x = X0; newWave(); history.length = 0; } // out on the right, back in on the left
      const y = waveY(x);
      const slope = (waveY(x + 4) - y) / 4;
      // she grows to full size on her first entrance
      const s = lerp(.5, 1, (age - ENTER) / 900);
      place(x, y, Math.max(-28, Math.min(28, Math.atan(slope) * 57.3)), s);
      flyer.setAttribute('opacity', String(Math.min(lerp(0, 1, (x - X0) / 80), lerp(1, 0, (x - 540) / 90))));
      history.unshift({x, y});
      history.length = Math.min(history.length, TRAIL);
      ghosts.forEach((g, i) => {
        const h = history[i];
        g.setAttribute('opacity', h ? String(lerp(.35, 0, i / TRAIL) * lerp(.5, 1, speed / 430)) : '0');
        if (h) { g.setAttribute('cx', h.x); g.setAttribute('cy', h.y); }
      });
      // the neurons she brushes past fire
      for (const p of nodes) {
        if (now - p.lit > 450 && (p.x - x) ** 2 + (p.y - y) ** 2 < 46 * 46) {
          p.lit = now;
          fire(p, Math.random() < .35 ? 2 : 1);
        }
      }
    }
    if (age >= START) {
      const k = (age - START) / 2000;
      counter.textContent = String(Math.round(target * (1 - Math.pow(1 - Math.min(k, 1), 3))));
    }
    frame = requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame); frame = 0;
    last = performance.now();
    if (visible && !document.hidden && !paused()) frame = requestAnimationFrame(tick);
  }
  place(X0, waveY(X0), 0, .5);
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); }).observe(intro);
  document.addEventListener('visibilitychange', sync);
  // the page's "Pause animation" button toggles this class on <html>
  new MutationObserver(sync).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  reduced.addEventListener('change', () => {
    if (!reduced.matches) return;
    visible = false; sync(); showFinal(); tokens.replaceChildren(); pulses.replaceChildren();
  });
  sync();
})();

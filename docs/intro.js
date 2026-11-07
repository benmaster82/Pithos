'use strict';
/* Opening scene. The jar holds a small neural network: each forward pass sends pulses up its layers
   to the mouth, and out comes a token. Passes speed up while the counter climbs to the measured
   speed. Decorative only: no network requests, no inference. */
(() => {
  const intro = document.querySelector('.intro');
  if (!intro) return;
  const SVG = 'http://www.w3.org/2000/svg';
  const body = intro.querySelector('#intro-body-path');
  const net = intro.querySelector('#intro-net');
  const layer = intro.querySelector('#intro-tokens');
  const counter = intro.querySelector('[data-count]');
  const target = Number(counter.dataset.count);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const MOUTH = {x: 210, y: 146};
  const OUT = {x: 210, y: 174}; // where the last layer meets the neck
  const WORDS = ['Local', 'AI', 'on', 'the', 'laptop', 'you', 'already', 'have', '.', 'No', 'big', 'GPU',
                 'needed', ':', 'just', 'Pithos', '.'];
  const BOOT = 650, START = 1900, RAMP = 2600, CALM = 6800; // ms from load
  const lerp = (a, b, k) => a + (b - a) * Math.min(Math.max(k, 0), 1);
  const el = (name, attrs, parent) => {
    const e = document.createElementNS(SVG, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.append(e);
    return e;
  };

  // ---- the network: layers of neurons across the jar's belly, bottom to top
  const layers = [];
  for (const y of [374, 338, 302, 266, 230, 198]) {
    let x0 = 0, x1 = 0;
    for (let x = 80; x <= 340; x += 2) {
      if (body.isPointInFill(new DOMPoint(x, y))) { if (!x0) x0 = x; x1 = x; }
    }
    const span = x1 - x0 - 36, n = Math.max(2, Math.min(7, Math.round(span / 38) + 1));
    layers.push(Array.from({length: n}, (_, i) => ({x: x0 + 18 + span * (n === 1 ? .5 : i / (n - 1)), y})));
  }
  // each neuron feeds its two nearest neurons of the next layer; every neuron gets an input
  const links = layers.slice(0, -1).map((lower, i) => {
    const upper = layers[i + 1], set = new Set();
    lower.forEach((a, ai) => upper.map((b, bi) => [Math.abs(a.x - b.x), bi]).sort((p, q) => p[0] - q[0])
      .slice(0, 2).forEach(([, bi]) => set.add(ai + ':' + bi)));
    upper.forEach((b, bi) => {
      if (![...set].some(s => s.endsWith(':' + bi))) {
        const ai = lower.map((a, k) => [Math.abs(a.x - b.x), k]).sort((p, q) => p[0] - q[0])[0][1];
        set.add(ai + ':' + bi);
      }
    });
    return [...set].map(s => { const [ai, bi] = s.split(':').map(Number); return [lower[ai], upper[bi]]; });
  });
  links.push(layers[layers.length - 1].map(a => [a, OUT])); // the last layer converges on the mouth
  const edges = el('g', {class: 'net-edges'}, net);
  for (const group of links) for (const [a, b] of group) el('line', {x1: a.x, y1: a.y, x2: b.x, y2: b.y}, edges);
  const pulses = el('g', {}, net);
  for (const row of layers) for (const p of row) {
    p.glow = el('circle', {class: 'net-glow', cx: p.x, cy: p.y, r: 8}, net);
    el('circle', {class: 'net-node', cx: p.x, cy: p.y, r: 3.2}, net);
  }

  const showFinal = () => { counter.textContent = String(target); };
  if (reduced.matches) { showFinal(); return; }

  // ---- one forward pass: pulses climb layer by layer, neurons flash, a token leaves the mouth
  const flash = (p, delay) => p.glow.animate([{opacity: 0}, {opacity: .9}, {opacity: 0}],
    {duration: 420, delay, easing: 'ease-out'});
  function pass(step, emitToken, age) {
    layers[0].forEach(p => flash(p, 0));
    links.forEach((group, i) => {
      for (const [a, b] of group) {
        if (Math.random() > .6 || pulses.childElementCount > 140) continue;
        const dot = el('circle', {class: 'net-pulse', r: 2.2, cx: 0, cy: 0}, pulses);
        dot.animate([{transform: `translate(${a.x}px,${a.y}px)`, opacity: 0},
                     {transform: `translate(${a.x}px,${a.y}px)`, opacity: 1, offset: .05},
                     {transform: `translate(${b.x}px,${b.y}px)`, opacity: .9}],
                    {duration: step, delay: i * step, easing: 'ease-in', fill: 'backwards'}).onfinish = () => dot.remove();
        if (b !== OUT) flash(b, (i + 1) * step);
      }
    });
    if (emitToken) setTimeout(() => emit(age), links.length * step);
  }

  let word = 0;
  function emit(age) {
    if (layer.childElementCount > 40) return;
    const k = (age - START) / RAMP;
    const tint = Math.random();
    const g = el('g', {class: 'intro-token' + (tint < .2 ? ' hot' : tint < .35 ? ' warm' : '')}, layer);
    const rect = el('rect', {}, g);
    const text = el('text', {'text-anchor': 'middle', y: 4}, g);
    text.textContent = WORDS[word++ % WORDS.length];
    const w = Math.max(22, text.getComputedTextLength() + 16);
    rect.setAttribute('x', String(-w / 2)); rect.setAttribute('y', '-11');
    rect.setAttribute('width', String(w)); rect.setAttribute('height', '22'); rect.setAttribute('rx', '7');
    // out of the right side of the mouth, then off to the right in a spray (clear of Pandora)
    const angle = lerp(0.1, -0.85, Math.random());
    const reach = lerp(200, 300, Math.random());
    const x0 = MOUTH.x + 60, y0 = MOUTH.y + 8;
    const dx = Math.cos(angle) * reach, dy = Math.sin(angle) * reach;
    const duration = age < CALM ? lerp(1700, 750, k) : 1700;
    g.animate([
      {transform: `translate(${x0}px,${y0}px) scale(.3)`, opacity: 0},
      {transform: `translate(${x0 + dx * .12}px,${y0 + dy * .12 - 10}px) scale(1)`, opacity: 1, offset: .15},
      {transform: `translate(${x0 + dx * .75}px,${y0 + dy * .75}px) scale(1)`, opacity: .9, offset: .75},
      {transform: `translate(${x0 + dx}px,${y0 + dy}px) scale(.95)`, opacity: 0}
    ], {duration, easing: 'cubic-bezier(.25,.6,.35,1)'}).onfinish = () => g.remove();
  }

  const t0 = performance.now();
  let next = 0, booted = false, visible = true, frame = 0;
  const paused = () => document.documentElement.classList.contains('inference-motion-paused');
  function tick(now) {
    frame = 0;
    if (!visible || document.hidden || paused()) return;
    const age = now - t0;
    if (!booted && age >= BOOT) { booted = true; pass(80, false, age); } // the network wakes up as the lid opens
    if (age >= START) {
      const k = (age - START) / 1800;
      counter.textContent = String(Math.round(target * (1 - Math.pow(1 - Math.min(k, 1), 3))));
      if (now >= next) {
        const ramp = (age - START) / RAMP, calm = age >= CALM;
        pass(calm ? 120 : lerp(115, 55, ramp), true, age);
        next = now + (calm ? 850 : lerp(520, 150, ramp));
      }
    }
    frame = requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame); frame = 0;
    if (visible && !document.hidden && !paused()) frame = requestAnimationFrame(tick);
  }
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); }).observe(intro);
  document.addEventListener('visibilitychange', sync);
  // the page's "Pause animation" button toggles this class on <html>
  new MutationObserver(sync).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  reduced.addEventListener('change', () => {
    if (!reduced.matches) return;
    visible = false; sync(); showFinal(); layer.replaceChildren(); pulses.replaceChildren();
  });
  sync();
})();

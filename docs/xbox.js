'use strict';
// xbox.js - the story page: mobile menu, chapters and charts revealed on scroll, counting numbers.
document.documentElement.classList.add('js');

const menuButton = document.querySelector('.menu-button'), navigation = document.querySelector('#navigation');
function closeMenu() { menuButton.setAttribute('aria-expanded', 'false'); navigation.classList.remove('is-open'); }
menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(open));
  navigation.classList.toggle('is-open', open);
});
navigation.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));
document.addEventListener('keydown', e => { if (e.key === 'Escape' && navigation.classList.contains('is-open')) { closeMenu(); menuButton.focus(); } });

const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// numbers count up once, from zero, keeping their decimals and suffix
function count(el) {
  const target = parseFloat(el.dataset.count), digits = (el.dataset.count.split('.')[1] || '').length;
  const suffix = el.dataset.suffix || '', t0 = performance.now(), ms = 1200;
  const step = now => {
    const k = Math.min(1, (now - t0) / ms), v = target * (1 - Math.pow(1 - k, 3));
    el.textContent = v.toFixed(digits) + suffix;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const targets = [...document.querySelectorAll('.x-chapter, .x-chart, .x-stats, .x-try, .x-shot')];
targets.forEach(el => { if (!el.classList.contains('x-chart')) el.classList.add('x-reveal'); });
if (!('IntersectionObserver' in window) || still) {
  targets.forEach(el => el.classList.add('is-visible'));
} else {
  const seen = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add('is-visible');
    if (entry.target.classList.contains('x-stats')) entry.target.querySelectorAll('[data-count]').forEach(count);
    seen.unobserve(entry.target);
  }), { rootMargin: '0px 0px -12% 0px' });
  targets.forEach(el => seen.observe(el));
}

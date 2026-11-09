/* Pandora presents the hero figures: it walks along the line above them, leaving a gradient trail,
 * stops on each figure, nods, and the number counts up. Once per visit; a click on Pandora replays it.
 * With reduced motion, or without JavaScript, the figures are simply shown. */
(()=>{
 const strip=document.querySelector('.hero-figures');
 if(!strip||!strip.animate)return;
 const cards=[...strip.children];
 const nums=cards.map(c=>c.querySelector('[data-to]'));
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const trail=document.createElement('span');trail.className='hero-trail';trail.setAttribute('aria-hidden','true');
 const guide=document.createElement('button');guide.type='button';guide.className='hero-guide';
 guide.setAttribute('aria-label','Replay the figures');
 guide.innerHTML='<img src="assets/pandora-mark.svg" width="30" height="30" alt="">';
 const stage=document.createElement('div');stage.className='hero-stage';
 strip.before(stage);stage.append(trail,guide,strip);
 const SIZE=30;
 let run=0;
 const stops=()=>cards.map((c,i)=>c.offsetLeft+(i?14:0));
 const endX=()=>strip.clientWidth-SIZE;
 const place=x=>{guide.style.transform=`translateX(${x}px)`;trail.style.width=`${Math.max(0,x+SIZE/2)}px`;};
 const showAll=()=>{nums.forEach(n=>n.textContent=n.dataset.to);cards.forEach(c=>c.classList.add('on'));place(endX());guide.classList.add('here');};
 const wait=ms=>new Promise(r=>setTimeout(r,ms));
 function count(n,ms){
  const to=+n.dataset.to,t0=performance.now();
  return new Promise(done=>{const f=t=>{const k=Math.max(0,Math.min(1,(t-t0)/ms));n.textContent=Math.round(to*(1-Math.pow(1-k,3)));k<1?requestAnimationFrame(f):done();};requestAnimationFrame(f);});
 }
 async function hop(from,to,id){
  const d=Math.max(380,Math.min(900,Math.abs(to-from)*2.4));
  const a=guide.animate([{transform:`translateX(${from}px)`},{transform:`translateX(${(from+to)/2}px) translateY(-12px)`,offset:.5},{transform:`translateX(${to}px)`}],{duration:d,easing:'ease-in-out'});
  trail.animate([{width:`${Math.max(0,from+SIZE/2)}px`},{width:`${Math.max(0,to+SIZE/2)}px`}],{duration:d,easing:'ease-in-out'});
  await a.finished.catch(()=>{});
  if(id!==run)return false;
  place(to);
  return true;
 }
 async function present(){
  const id=++run;
  if(reduced.matches){showAll();return;}
  nums.forEach(n=>n.textContent='0');cards.forEach(c=>c.classList.remove('on'));
  guide.classList.remove('here');guide.classList.add('in');
  let x=-SIZE;place(x);
  const xs=stops();
  for(let i=0;i<cards.length;i++){
   if(!await hop(x,xs[i],id))return;
   x=xs[i];
   guide.animate([{transform:`translateX(${x}px) rotate(0)`},{transform:`translateX(${x}px) rotate(14deg)`,offset:.4},{transform:`translateX(${x}px) rotate(0)`}],{duration:520,easing:'ease-out'});
   cards[i].classList.add('on');
   await count(nums[i],700);
   if(id!==run)return;
   await wait(260);
  }
  if(!await hop(x,endX(),id))return;
  guide.classList.add('here');
 }
 guide.addEventListener('click',present);
 new ResizeObserver(()=>{if(guide.classList.contains('here'))place(endX());}).observe(strip);
 if(reduced.matches){showAll();return;}
 strip.classList.add('play');nums.forEach(n=>n.textContent='0');place(-SIZE);
 const io=new IntersectionObserver(es=>{if(es[0].isIntersecting){io.disconnect();setTimeout(present,350);}},{threshold:.6});
 io.observe(strip);
})();

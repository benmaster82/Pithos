/* Decorative ASCII projection of the existing vector mark; no network or inference. */
(()=>{
 const hero=document.querySelector('.hero');
 if(!hero)return;
 const canvas=document.createElement('canvas');
 canvas.className='inference-backdrop';canvas.setAttribute('aria-hidden','true');hero.prepend(canvas);
 const control=document.createElement('button');control.className='motion-toggle';control.type='button';control.textContent='Pause animation';control.setAttribute('aria-pressed','false');hero.append(control);
 const ctx=canvas.getContext('2d');if(!ctx){canvas.remove();control.remove();return;}
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 let paused=reduced.matches,visible=false,frame=0,last=0,phase=0,points=[],width=0,height=0;
 function resize(){const r=hero.getBoundingClientRect();width=r.width;height=r.height;const d=Math.min(devicePixelRatio||1,1.5);canvas.width=Math.round(width*d);canvas.height=Math.round(height*d);ctx.setTransform(d,0,0,d,0,0);draw();}
 function draw(){
  ctx.clearRect(0,0,width,height);if(!points.length)return;
  const compact=width<760;
  const size=compact?Math.min(width*.94,540):Math.min(width*.48,590),cx=compact?width*.5:width*.25,cy=compact?280:Math.min(height*.48,340);
  const angle=phase*.22,co=Math.cos(angle),si=Math.sin(angle);
  ctx.font=`${Math.max(9,size/49)}px monospace`;ctx.textAlign='center';ctx.textBaseline='middle';
  const chars='.:+*#@';
  for(const p of points){
   const x=p.x*co+p.z*si,z=-p.x*si+p.z*co,perspective=2.7/(2.7-z);
   const pulse=(Math.sin(p.y*8-phase*1.8)+1)/2;
   ctx.fillStyle=`rgba(218,238,255,${.24+(z+.6)*.3+pulse*.16})`;
   ctx.fillText(chars[Math.min(5,Math.floor(pulse*3+(z+.5)*2))],cx+x*size*perspective,cy+p.y*size*perspective);
  }
 }
 function tick(t){frame=0;if(paused||!visible||document.hidden)return;if(t-last>=50){phase+=Math.min((t-last)/1000,.1);last=t;draw();}frame=requestAnimationFrame(tick);}
 function sync(){cancelAnimationFrame(frame);frame=0;control.textContent=paused?'Play animation':'Pause animation';control.setAttribute('aria-pressed',String(paused));hero.classList.toggle('motion-paused',paused);document.documentElement.classList.toggle('inference-motion-paused',paused);if(!paused&&visible&&!document.hidden){last=performance.now();frame=requestAnimationFrame(tick);}else draw();}
 control.addEventListener('click',()=>{paused=!paused;sync();});reduced.addEventListener('change',()=>{paused=reduced.matches;sync();});document.addEventListener('visibilitychange',sync);
 new IntersectionObserver(es=>{visible=es[0].isIntersecting;sync();}).observe(hero);
 new ResizeObserver(resize).observe(hero);
 const mark=new Image();mark.onload=()=>{
  const sample=document.createElement('canvas');sample.width=64;sample.height=40;const s=sample.getContext('2d');s.drawImage(mark,0,0,64,40);const pixels=s.getImageData(0,0,64,40).data;
  for(let y=0;y<40;y++)for(let x=0;x<64;x++)if(pixels[(y*64+x)*4+3]>150){const px=(x-32)/64,py=(y-20)/40;points.push({x:px,y:py,z:.055*Math.cos(px*7)*Math.cos(py*5)});}
  resize();sync();
 };mark.onerror=()=>{canvas.remove();control.remove();};mark.src='assets/pandora-mark.svg';sync();
})();

'use strict';

const REPO = 'https://github.com/benmaster82/Pithos';
// The validated models, as in models.tsv. big is the engine's default (no -m needed).
const MODELS = {
  big:{id:'lfm2.5-24b',name:'LFM2.5-24B-A2B',file:'LFM2.5-24B-A2B-Q4_K_M.gguf',
    url:'https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/resolve/0912d33141f77e10705345a6732bfd83f0252fdc/LFM2.5-24B-A2B-Q4_K_M.gguf',
    hash:'d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d',
    download:'14.5 GB',ram:'16',gpu:'12–22 tok/s',cpu:'~3 tok/s',
    disk:'At least 15 GB free for the model',diskDetail:'About 14.5 GB to download. Allow extra room for the compiler and, if used, Ubuntu.',
    blurb:'24 billion parameters, about 2 billion active per token. The deepest answers; runs from disk when RAM is short.'},
  small:{id:'lfm3-1.7b',name:'LFM3-1.7B',file:'LFM3-1.7B-Q4_K_M.gguf',
    url:'https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/bad5729bba61d076fff20d2548e62526665af215/LFM3-1.7B-Q4_K_M.gguf',
    hash:'3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4',
    download:'1.1 GB',ram:'4',gpu:'~55 tok/s',cpu:'~20 tok/s',
    disk:'At least 2 GB free for the model',diskDetail:'About 1.1 GB to download. Allow extra room for the compiler and, if used, Ubuntu.',
    blurb:'1.7 billion parameters with sliding-window attention. Quick answers on modest hardware; the whole model fits in a 4 GB GPU.'}
};
// " -m models/FILE" for a model that is not the engine's default
const modelArg = (m, windows) => m === MODELS.big ? '' : windows ? ` -m models\\${m.file}` : ` -m models/${m.file}`;
const downloadFlag = m => m === MODELS.big ? '--download-model' : `--download-model=${m.id}`;
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escapeHTML = value => value.replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const external = (url, label) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label} ${icon('external')}</a>`;
const commandBlock = (command, context, id = 'guide-code') => `<div class="code-block"><div class="code-block-header"><span>${context}</span><button class="copy-button" data-copy-target="${id}" aria-label="Copy commands: ${context}">${icon('copy')} Copy</button></div><pre><code id="${id}">${escapeHTML(command)}</code></pre></div>`;
const note = content => `<div class="step-note">${icon('check')}<div>${content}</div></div>`;

const buildGuide = m => ({
  windows: {
    caption: 'Windows x64 · native build, no WSL required',
    steps: [
      {title:'Prepare Windows. Just once.', label:'01 / THE COMPILER',
        description:`<p>Pithos runs directly on Windows. To build it, install ${external('https://www.msys2.org/', 'MSYS2')} in the default folder <code>C:\\msys64</code>.</p><p>Open <strong>MSYS2 UCRT64</strong> from the Start menu and paste this command. Press Enter if asked to confirm.</p>`,
        command:'pacman -S mingw-w64-ucrt-x86_64-gcc', context:'MSYS2 UCRT64 · not PowerShell',
        note:'Already have MSYS2 GCC or Visual Studio/Build Tools MSVC? Skip to step 2. WSL2 is not needed.'},
      {title:'Set up Pithos and its model.', label:'02 / PROGRAM + MODEL',
        description:`<p><strong>1.</strong> Download and extract the project ZIP. Open <strong>Pithos-main</strong>, type <code>powershell</code> in the address bar, and press Enter.</p><p><strong>2.</strong> Run the commands below. Then download the model and move <strong>${m.file}</strong> into the <strong>models</strong> folder you just created.</p>`,
        command:'.\\build.cmd\nNew-Item -ItemType Directory -Force models', context:'PowerShell · inside Pithos-main',
        actions:[{url:`${REPO}/archive/refs/heads/main.zip`,label:'Project · ZIP'},{url:m.url,label:`Model · about ${m.download}`}],
        note:'Wait for “ok: build\\pithos.exe”. Use the exact GGUF model listed here; other files may not be compatible. The download can take a while.', checkHash:true},
      {title:'Open your first chat.', label:'03 / ALMOST THERE',
        description:'<p>Stay in the project folder in PowerShell. Start Pithos with this command and <strong>keep the terminal open</strong>.</p><p>When the server is ready, open the local address below in your browser.</p>',
        command:`.\\build\\pithos.exe${modelArg(m, true)} --server`, context:'PowerShell · inside Pithos-main', localLink:true,
        note:'A compatible GPU is used automatically; add --no-gpu to stay on the CPU. Press Ctrl+C to stop the program. To open it tomorrow, repeat just this step.'}
    ]
  },
  linux: {
    caption:'Ubuntu / Debian · guided build and download',
    steps:[
      {title:'Open a terminal. Set up Git.',label:'01 / THE STARTING POINT',
        description:`<p>On <strong>Ubuntu or Debian</strong>, open a terminal and paste the commands below. Git downloads the project.</p><p>On Fedora, Arch, or openSUSE, install Git using your package manager, then continue to step 2. ${external(`${REPO}#building`, 'See the README')}.</p>`,
        command:'sudo apt update\nsudo apt install -y git',context:'Linux terminal · Ubuntu / Debian',
        note:'Your sudo password does not show any characters as you type. That is normal. Type it and press Enter.'},
      {title:'One command sets up the engine.',label:'02 / PROGRAM + MODEL',
        description:'<p>Download the project, enter its folder, and run the script. It checks prerequisites and asks before installing missing packages.</p><p>The model is downloaded and verified automatically. <strong>Keep the terminal open</strong> until the process finishes.</p>',
        command:`git clone ${REPO}.git\ncd Pithos\nbash build.sh ${downloadFlag(m)}`,context:'Linux terminal',
        note:`The model is about ${m.download}. If the download stops, run “bash build.sh ${downloadFlag(m)}” again from the Pithos folder. Do not run the script with sudo.`},
      {title:'Your chat is one step away.',label:'03 / OPEN PANDORA',
        description:'<p>Run this command from the Pithos folder. It starts the local server. <strong>Keep the terminal open</strong> and open the address in your browser.</p>',
        command:`./build/pithos${modelArg(m, false)} --server`,context:'Linux terminal · inside Pithos',localLink:true,
        note:'The chat runs on your computer; a compatible GPU is used automatically (--no-gpu stays on the CPU). Press Ctrl+C to stop it. Next time, repeat only this step.'}
    ]
  },
  wsl:{
    caption:'Windows alternative · Ubuntu through WSL2',
    steps:[
      {title:'Set up Ubuntu inside Windows.',label:'01 / ONLY IF YOU CHOOSE WSL2',
        description:`<p>This route is <strong>optional</strong>. Native Windows does not require WSL. To use Ubuntu, open <strong>PowerShell as administrator</strong> and run this command.</p><p>Restart if prompted, open <strong>Ubuntu</strong> from the Start menu, and create a Linux username and password. ${external('https://learn.microsoft.com/en-us/windows/wsl/install', 'Microsoft guide')}.</p>`,
        command:'wsl --install -d Ubuntu',context:'PowerShell · as administrator',
        note:'For the next steps, use the Ubuntu terminal, not PowerShell. Under WSL, Ubuntu receives only part of your PC memory.'},
      {title:'Switch to the Ubuntu terminal.',label:'02 / PROGRAM + MODEL',
        description:'<p>Open <strong>Ubuntu</strong> from the Start menu and paste these commands. The script builds Pithos, offers to install missing prerequisites, and downloads the verified model.</p>',
        command:`sudo apt update\nsudo apt install -y git\n\ngit clone ${REPO}.git\ncd Pithos\nbash build.sh ${downloadFlag(m)}`,context:'Ubuntu terminal · not PowerShell',
        note:`The download is about ${m.download}. If interrupted, run “bash build.sh ${downloadFlag(m)}” again from the Pithos folder. Your Linux password is not displayed as you type.`},
      {title:'Open your Windows browser.',label:'03 / OPEN PANDORA',
        description:'<p>Start the server from the Pithos folder in the Ubuntu terminal. Open the local address in your usual <strong>Windows browser</strong>. Keep Ubuntu open.</p>',
        command:`./build/pithos${modelArg(m, false)} --server --no-gpu`,context:'Ubuntu terminal · inside Pithos',localLink:true,
        note:'This guide uses the CPU. OpenCL GPU acceleration in WSL was not verified. Press Ctrl+C in the Ubuntu terminal to stop the program.'}
    ]
  }
});

const detectedOS = /win/i.test(navigator.userAgentData?.platform || navigator.platform || '') ? 'windows' : 'linux';
let state = {os:detectedOS,step:0,model:'big',done:{windows:[],linux:[],wsl:[]}};
try {
  const saved = JSON.parse(localStorage.getItem('pithos-guide-v1'));
  if (saved && ['windows','linux','wsl'].includes(saved.os)) {
    state.os=saved.os;
    if (Object.hasOwn(MODELS,saved.model)) state.model=saved.model;
    state.step=Number.isInteger(saved.step) && saved.step>=0 && saved.step<=2 ? saved.step : 0;
    for(const os of ['windows','linux','wsl']) state.done[os]=Array.isArray(saved.done?.[os]) ? [...new Set(saved.done[os].filter(n=>Number.isInteger(n)&&n>=0&&n<=2))] : [];
  }
} catch { /* Storage is optional; the guide works without it. */ }
let activeMode='browser';
const saveState=()=>{try{localStorage.setItem('pithos-guide-v1',JSON.stringify(state));}catch{}};

function renderGuide() {
  const model=MODELS[state.model], guide=buildGuide(model);
  const step=guide[state.os].steps[state.step];
  document.querySelectorAll('[data-os]').forEach(button=>{
    const selected=button.dataset.os===state.os;
    button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;
  });
  document.querySelector('#os-caption').textContent=guide[state.os].caption;
  document.querySelector('#guide-progress').textContent=state.done[state.os].length===3 ? 'All 3 steps complete' : `Step ${state.step+1} of 3`;
  document.querySelectorAll('[data-step]').forEach(button=>{
    const index=Number(button.dataset.step);
    if(index===state.step)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');
    const done=state.done[state.os].includes(index);button.classList.toggle('is-complete',done);
    button.querySelector('span').textContent=done?'✓':String(index+1);
  });
  const panel=document.querySelector('#guide-panel');
  panel.setAttribute('aria-labelledby',`os-${state.os}`);
  panel.innerHTML=`<span class="step-label">${step.label}</span><h3 class="step-title" tabindex="-1">${step.title}</h3><div class="step-description">${step.description}</div>${step.actions?`<div class="step-actions">${step.actions.map(action=>external(action.url,action.label)).join('')}</div>`:''}${commandBlock(step.command,step.context)}${step.localLink?`<div class="step-actions"><a href="http://127.0.0.1:8080" target="_blank" rel="noopener noreferrer">Open the chat on your PC ${icon('external')}</a></div>`:''}${note(step.note)}${step.checkHash?`<details class="model-check"><summary>How to verify the downloaded model</summary><p>Run this command from the project folder. The result must match the expected value below.</p>${commandBlock(`(Get-FileHash .\\models\\${model.file} -Algorithm SHA256).Hash`,'PowerShell','hash-code')}<p>Expected SHA-256:</p><code>${model.hash.toUpperCase()}</code></details>`:''}${state.step===2&&state.done[state.os].includes(2)?'<div class="guide-finished"><strong>Chat open? Enjoy your workspace!</strong>Next time, just start Pithos from its folder. No new download is needed.</div>':''}`;
  document.querySelector('#guide-back').disabled=state.step===0;
  document.querySelector('#guide-next').innerHTML=(state.step===2?'I have opened the chat':'I have completed this step')+icon(state.step===2?'check':'arrow');
  saveState();
  renderModel();
  renderMode();
}

// The model picker at the top: figures, requirements and every command on the page follow it.
function renderModel() {
  const m=MODELS[state.model];
  document.querySelectorAll('[data-model]').forEach(button=>{const selected=button.dataset.model===state.model;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});
  document.querySelector('#model-card').setAttribute('aria-labelledby',`model-${state.model}`);
  document.querySelector('#model-size').textContent=m.download;
  document.querySelector('#model-ram').textContent=`${m.ram} GB`;
  document.querySelector('#model-gpu').textContent=m.gpu;
  document.querySelector('#model-cpu').textContent=m.cpu;
  document.querySelector('#model-blurb').textContent=m.blurb;
  document.querySelector('#req-ram').firstChild.nodeValue=m.ram;
  document.querySelector('#req-disk').textContent=m.disk;
  document.querySelector('#req-disk-detail').textContent=m.diskDetail;
  document.querySelector('#guide-model-name').textContent=m.name;
}

const SHOTS={
  pandora:{src:'assets/pandora-dark.png',width:1280,height:1075,label:'PANDORA / ACTUAL INTERFACE',
    alt:'Actual Pandora interface: a conversation with tools for running commands and viewing their results.',
    title:'An engine.<br><em>And a place to work.</em>',
    text:'Pandora is the web interface included with Pithos. Open your conversations in the browser, and see the tools and their results when using agent mode.'},
  terminal:{src:'assets/pithos-banner.png',width:892,height:424,label:'TERMINAL / ACTUAL STARTUP BANNER',
    alt:'The Pithos terminal agent starting up: the Pandora mark and the PITHOS wordmark in a violet, rose and amber gradient, with the detected system and the agent mode.',
    title:'An engine.<br><em>And a terminal that works.</em>',
    text:'The terminal agent opens with this banner, then runs commands, edits files, and checks its own work. In safe mode it asks before every action, and it detects Windows or Linux to use the right shell syntax.'}
};

function renderMode() {
  const windows=state.os==='windows';
  const binary=(windows?'.\\build\\pithos.exe':'./build/pithos')+modelArg(MODELS[state.model],windows);
  const cpu=state.os==='wsl'?' --no-gpu':''; // GPU acceleration was not verified under WSL
  document.querySelector('#hero-code').textContent=`${binary} --server${cpu}`;
  document.querySelector('#hero-context').textContent=windows?'WINDOWS / NATIVE':state.os==='wsl'?'UBUNTU / WSL2':'LINUX / LOCAL';
  document.querySelector('#hero-prompt').textContent=windows?'PS>':'$';
  const config={
    browser:{title:'Your browser becomes your chat.',description:'After setup, start the server and open the local address. Keep the terminal open while you chat.',command:`${binary} --server${cpu}`,followup:'Then open http://127.0.0.1:8080 in your browser.',shot:'pandora'},
    terminal:{title:'Ask and answer, right in your terminal.',description:'Start a text conversation in the terminal. For a single answer, use -p followed by your question.',command:`${binary}${cpu}`,followup:'Press Ctrl+C to end the session.',shot:'terminal'},
    agent:{title:'Practical help for your projects.',description:'The agent can work with files and commands. In safe mode, you approve actions. For another project, set the model path with -m. The README has examples.',command:`${binary} --agent${cpu}`,followup:'The agent works in the folder where you start it.',shot:'terminal'},
    api:{title:'Connect an application to the engine.',description:'Use /v1/chat/completions or /v1/completions through the local HTTP server. These are OpenAI-compatible inference endpoints; the engine processes one inference request at a time.',command:`${binary} --server${cpu}`,followup:'API base URL: http://127.0.0.1:8080/v1',shot:'pandora'}
  }[activeMode];
  document.querySelectorAll('[data-mode]').forEach(button=>{const selected=button.dataset.mode===activeMode;button.classList.toggle('is-selected',selected);button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});
  document.querySelector('#mode-panel').setAttribute('aria-labelledby',`mode-${activeMode}`);
  document.querySelector('#mode-title').textContent=config.title;
  document.querySelector('#mode-description').textContent=config.description;
  document.querySelector('#mode-code').textContent=config.command;
  document.querySelector('#mode-context').textContent=`From the Pithos folder · ${windows?'PowerShell Windows':state.os==='wsl'?'Ubuntu terminal / WSL2':'Linux terminal'}`;
  document.querySelector('#mode-followup').textContent=config.followup;
  const shot=SHOTS[config.shot], img=document.querySelector('#product-shot');
  if(img.getAttribute('src')!==shot.src){
    img.src=shot.src;img.width=shot.width;img.height=shot.height;img.alt=shot.alt;
    document.querySelector('#product-link').href=shot.src;
    document.querySelector('#product-label').textContent=shot.label;
    document.querySelector('#product-title').innerHTML=shot.title;
    document.querySelector('#product-text').textContent=shot.text;
    document.querySelector('#product-card').classList.toggle('is-terminal',config.shot==='terminal');
  }
}

document.querySelectorAll('[data-os]').forEach(button=>button.addEventListener('click',()=>{state.os=button.dataset.os;state.step=0;renderGuide();}));
document.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',()=>{state.step=Number(button.dataset.step);renderGuide();}));
document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{activeMode=button.dataset.mode;renderMode();}));
document.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>{state.model=button.dataset.model;renderGuide();}));
document.querySelector('#guide-back').addEventListener('click',()=>{state.step=Math.max(0,state.step-1);renderGuide();});
document.querySelector('#guide-next').addEventListener('click',()=>{if(!state.done[state.os].includes(state.step))state.done[state.os].push(state.step);if(state.step<2)state.step++;renderGuide();document.querySelector('.step-title').focus({preventScroll:true});});

function bindTabKeyboard(container,selector) {
  container.addEventListener('keydown',event=>{
    if(!['ArrowRight','ArrowLeft','Home','End'].includes(event.key))return;
    const tabs=[...container.querySelectorAll(selector)];const index=tabs.indexOf(document.activeElement);
    if(index<0)return;event.preventDefault();
    const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    tabs[next].click();tabs[next].focus();
  });
}
bindTabKeyboard(document.querySelector('.os-tabs'),'[data-os]');
bindTabKeyboard(document.querySelector('.mode-tabs'),'[data-mode]');
bindTabKeyboard(document.querySelector('.model-tabs'),'[data-model]');

let toastTimer;
function toast(message) {const node=document.querySelector('#toast');node.textContent=message;node.classList.add('is-visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>node.classList.remove('is-visible'),2400);}
async function copyText(value) {
  try {await navigator.clipboard.writeText(value);return true;} catch {
    const input=document.createElement('textarea');input.value=value;input.style.position='fixed';input.style.left='-10000px';input.setAttribute('aria-label','Text to copy');document.body.append(input);input.select();
    let success=false;try{success=document.execCommand('copy');}catch{}input.remove();return success;
  }
}
document.addEventListener('click',async event=>{
  const button=event.target.closest('[data-copy-target]');if(!button)return;
  const target=document.getElementById(button.dataset.copyTarget);if(!target)return;
  const copied=await copyText(target.textContent);toast(copied?'Commands copied. Paste them into the terminal shown.':'Copy unavailable. Select the command and press Ctrl+C.');
  if(copied){const old=button.innerHTML;button.innerHTML=icon('check')+(button.classList.contains('icon-only')?'':' Copied');setTimeout(()=>{button.innerHTML=old;},1800);}
});

const menuButton=document.querySelector('.menu-button');const navigation=document.querySelector('#navigation');
function closeMenu(){menuButton.setAttribute('aria-expanded','false');navigation.classList.remove('is-open');}
menuButton.addEventListener('click',()=>{const open=menuButton.getAttribute('aria-expanded')!=='true';menuButton.setAttribute('aria-expanded',String(open));navigation.classList.toggle('is-open',open);});
navigation.querySelectorAll('a').forEach(link=>link.addEventListener('click',closeMenu));
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&navigation.classList.contains('is-open')){closeMenu();menuButton.focus();}});
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){navigation.querySelectorAll('a').forEach(link=>{if(link.getAttribute('href')===`#${entry.target.id}`)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});}});},{rootMargin:'-20% 0px -55% 0px'});document.querySelectorAll('#project,#architecture,#specifications,#get-started,#mcp,#faq').forEach(section=>observer.observe(section));}
renderGuide();


// Explanatory architecture controls; they do not configure a local engine.
const executionPaths={
  cpu:{title:'Model execution',cpu:'Tokenization, routing, attention/convolution, and selected experts run through the CPU path.',summary:'CPU-only execution.',detail:'All model compute stays on the CPU. OS-backed paging, read-ahead, and expert prediction help with cold weights.'},
  hybrid:{title:'Schedule and combine',cpu:'CPU batch processing and routing. During hybrid decode, CPU workers compute selected experts that are not resident in VRAM.',summary:'Hybrid execution for the reference MoE model.',detail:'The prompt uses the CPU batch path, with optional cached-expert GPU offload. In single-token generation, the GPU can run dense decode and resident experts while CPU workers process non-resident experts. Contributions are combined before sampling. A dense model such as LFM3 decodes entirely on the GPU.'}
};
function renderExecution(mode){
  const path=executionPaths[mode];
  document.querySelectorAll('[data-execution]').forEach(button=>{const selected=button.dataset.execution===mode;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});
  document.querySelector('#execution-panel').setAttribute('aria-labelledby',`execution-${mode}`);
  document.querySelector('#runtime-cpu-title').textContent=path.title;
  document.querySelector('#runtime-cpu-text').textContent=path.cpu;
  document.querySelector('#runtime-gpu').hidden=mode!=='hybrid';
  document.querySelector('.engine-core').classList.toggle('is-hybrid',mode==='hybrid');
  document.querySelector('#execution-summary').textContent=path.summary;
  document.querySelector('#execution-detail').textContent=path.detail;
}
document.querySelectorAll('[data-execution]').forEach(button=>button.addEventListener('click',()=>renderExecution(button.dataset.execution)));
bindTabKeyboard(document.querySelector('.execution-tabs'),'[data-execution]');
renderExecution('cpu');

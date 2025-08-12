// ===== Config =====
const API_BASE = document.querySelector('meta[name="api-base-url"]')?.content?.trim() || "http://localhost:8000";
const TIMEOUT_MS = 60_000;

// ===== Helpers =====
const $ = (sel) => document.querySelector(sel);
const create = (t, cls) => { const el = document.createElement(t); if (cls) el.className = cls; return el; };
const nowTime = () => new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
const escapeHTML = (s) => s.replace(/[&<>"']/g, (m) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
function renderMarkdownSafe(text){
  let html = escapeHTML(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/^🚨(.+)$/m, '<div class="callout callout--danger">🚨$1</div>');
  return html.replace(/\n/g, "<br/>");
}
function toast(msg, kind=""){ const host=$("#toastHost"); const el=create("div",`toast ${kind?`toast--${kind}`:""}`); el.textContent=msg; host.appendChild(el); setTimeout(()=>{el.style.opacity="0"; setTimeout(()=>el.remove(),250);},3500); }
function fetchJSON(url, init={}, timeout=TIMEOUT_MS){
  const controller=new AbortController(); const t=setTimeout(()=>controller.abort(), timeout);
  return fetch(url,{...init, signal:controller.signal}).finally(()=>clearTimeout(t));
}

// ===== Theme =====
const root=document.documentElement;
function prefersDark(){ return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches; }
function effective(){ return root.getAttribute("data-theme") || (prefersDark() ? "dark" : "light"); }
function applyTheme(t){ if(t==="dark") root.setAttribute("data-theme","dark"); else if(t==="light") root.setAttribute("data-theme","light"); else root.removeAttribute("data-theme"); }
applyTheme(localStorage.getItem("theme") || "");

// ===== State =====
const chatList=$("#chatList"), scroller=$("#chatScroller"), jumpBottom=$("#jumpBottom");
const input=$("#input"), sendBtn=$("#send");
const btnTopPdf=$("#btnTopPdf");
const rxOverlay=$("#rxOverlay"), rxModal=$("#rxModal"), rxClose=$("#rxClose"), rxGenerate=$("#rxGenerate"), rxDownload=$("#rxDownload");
const rxName=$("#rxName"), rxAge=$("#rxAge"), rxGender=$("#rxGender"), rxDoctor=$("#rxDoctor"), rxClinic=$("#rxClinic");

let messages=[];
appendAssistant("Hi! I’m your medical assistant. Ask me about your reports or symptoms.\nI’ll use the knowledge base and include cautions. For emergencies, consult a doctor immediately.");

// ===== Events =====
sendBtn.addEventListener("click", onSend);
input.addEventListener("keydown",(e)=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); onSend(); }});
input.addEventListener("input",()=>{ input.style.height="0px"; input.style.height=Math.min(160,input.scrollHeight)+"px"; });
btnTopPdf.addEventListener("click", openRxModal);
rxClose.addEventListener("click", closeRxModal);
rxOverlay.addEventListener("click", closeRxModal);
rxGenerate.addEventListener("click", generatePdf);
jumpBottom.addEventListener("click", ()=> scroller.scrollTo({top:scroller.scrollHeight, behavior:"smooth"}));
scroller.addEventListener("scroll", ()=>{
  const nearBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 120;
  jumpBottom.classList.toggle("hidden", nearBottom);
});

// ===== Chat logic =====
async function onSend(){
  const text = input.value.trim(); if(!text) return;
  input.value=""; input.style.height="0px";
  appendUser(text);

  // typing skeleton bubble
  const skelId = showTypingSkeleton();

  try{
    const res=await fetchJSON(`${API_BASE}/chat`,{
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({query:text})
    });
    if(!res.ok){ const errText=await res.text(); throw new Error(`Chat failed (${res.status}): ${errText}`); }
    const data=await res.json();
    replaceSkeletonWithAssistant(skelId, data.answer || "", data.references || []);
  }catch(e){
    replaceSkeletonWithAssistant(skelId, `Sorry, I couldn’t reach the server right now.\n\nError: ${e.message || e}`, []);
    toast("Chat request failed","error");
  }
}

function appendUser(content){
  messages.push({role:"user", content});
  const row=create("li","msg msg--user");
  const bubble=create("div","bubble"); bubble.textContent=content;
  const avatar=create("div","avatar"); avatar.textContent="You";
  const time=create("div","time"); time.textContent=nowTime();
  row.appendChild(bubble); row.appendChild(avatar);
  bubble.appendChild(time);
  chatList.appendChild(row);
  scrollToBottom();
}

function appendAssistant(content, references=[]){
  messages.push({role:"assistant", content, references});
  const row=create("li","msg msg--assistant");
  const avatar=create("div","avatar"); avatar.textContent="AI";
  const bubble=create("div","bubble"); bubble.innerHTML=renderMarkdownSafe(content);
  const time=create("div","time"); time.textContent=nowTime();
  row.appendChild(avatar); row.appendChild(bubble);

  if(references.length){
    const details=create("details","sources");
    const summary=create("summary"); summary.textContent="Show sources";
    const wrap=create("div","sources__wrap");
    references.forEach(ref=>{ const pre=create("pre"); pre.textContent=ref; wrap.appendChild(pre); });
    details.appendChild(summary); details.appendChild(wrap);
    bubble.appendChild(details);
  }
  bubble.appendChild(time);
  chatList.appendChild(row);
  scrollToBottom();
}

function showTypingSkeleton(){
  const id = "skel-"+crypto.randomUUID();
  const row=create("li","msg msg--assistant"); row.id=id;
  const avatar=create("div","avatar"); avatar.textContent="AI";
  const shell=create("div","skel");
  shell.innerHTML=`<div class="line" style="width:72%"></div><div class="line" style="width:86%"></div><div class="line short"></div>`;
  row.appendChild(avatar); row.appendChild(shell);
  chatList.appendChild(row); scrollToBottom();
  return id;
}
function replaceSkeletonWithAssistant(id, content, references){
  const row=$("#"+CSS.escape(id));
  if(!row){ appendAssistant(content, references); return; }
  row.innerHTML="";
  const avatar=create("div","avatar"); avatar.textContent="AI";
  const bubble=create("div","bubble"); bubble.innerHTML=renderMarkdownSafe(content);
  const time=create("div","time"); time.textContent=nowTime();
  row.className="msg msg--assistant"; row.appendChild(avatar); row.appendChild(bubble);

  if(references.length){
    const details=create("details","sources");
    const summary=create("summary"); summary.textContent="Show sources";
    const wrap=create("div","sources__wrap");
    references.forEach(ref=>{ const pre=create("pre"); pre.textContent=ref; wrap.appendChild(pre); });
    details.appendChild(summary); details.appendChild(wrap);
    bubble.appendChild(details);
  }
  bubble.appendChild(time);
  scrollToBottom();
}

function scrollToBottom(){ scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }); }

// ===== PDF =====
function openRxModal(){
  const lastUser=[...messages].reverse().find(m=>m.role==="user");
  const lastAssistant=[...messages].reverse().find(m=>m.role==="assistant");
  if(!lastUser || !lastAssistant){ toast("Ask a question first so I can base the PDF on it.","error"); return; }
  rxDownload.classList.add("hidden");
  rxOverlay.classList.remove("hidden"); rxModal.classList.remove("hidden");
}
function closeRxModal(){ rxOverlay.classList.add("hidden"); rxModal.classList.add("hidden"); }

async function generatePdf(){
  const lastUser=[...messages].reverse().find(m=>m.role==="user");
  const lastAssistant=[...messages].reverse().find(m=>m.role==="assistant");
  if(!lastUser || !lastAssistant){ toast("Ask a question first so I can base the PDF on it.","error"); return; }

  rxGenerate.disabled=true; rxGenerate.textContent="Generating…"; rxDownload.classList.add("hidden");
  try{
    const payload={
      latest_user_query:lastUser.content,
      assistant_answer:lastAssistant.content,
      patient_name: rxName.value.trim() || undefined,
      patient_age:  rxAge.value.trim()? Number(rxAge.value.trim()) : undefined,
      patient_gender: rxGender.value.trim() || undefined,
      doctor_name: rxDoctor.value.trim() || undefined,
      clinic_name: rxClinic.value.trim() || undefined
    };
    const res=await fetchJSON(`${API_BASE}/generate_prescription_pdf_from_chat`,{
      method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(payload)
    }, 90_000);
    if(!res.ok){ const txt=await res.text(); throw new Error(`PDF failed (${res.status}): ${txt}`); }
    const blob=await res.blob(); const url=URL.createObjectURL(blob);
    rxDownload.href=url; rxDownload.classList.remove("hidden"); toast("Prescription generated.");
  }catch(e){ console.error(e); toast(e.message || "PDF generation failed","error"); }
  finally{ rxGenerate.disabled=false; rxGenerate.textContent="Generate PDF"; }
}

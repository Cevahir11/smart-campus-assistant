const CAT_TR = { exam: "Sınav", event: "Etkinlik", scholarship: "Burs", cancellation: "Ders İptali",
                 room_change: "Derslik Değişikliği", registration: "Kayıt / Takvim" };
const SRC_TR = { university_website: "Üniversite sitesi", department_website: "Bölüm sitesi", email: "E-posta",
                 student_club: "Öğrenci kulübü", user: "Kullanıcı", hand_written: "Elle", unknown: "?" };
const TR_MONTHS = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"];
const EXAMPLES = ["Bu hafta sınav var mı?", "Yarın etkinlik var mı?", "CENG 301 sınavı ne zaman?",
                  "Gelecek hafta hangi etkinlikler var?", "Erasmus+ Staj Hareketliliği son başvuru tarihi ne zaman?",
                  "Are there any exams next week?"];

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtDate = (iso) => { if (!iso) return ""; const [y, m, d] = iso.split("-").map(Number); return `${d} ${TR_MONTHS[m - 1]} ${y}`; };
let state = { category: "", q: "", upcoming: false };

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error((await r.json()).error || r.statusText);
  return r.json();
}

function highlight(text, ents) {
  let out = "", pos = 0;
  for (const e of [...ents].sort((a, b) => a.start - b.start)) {
    if (e.start < pos) continue;
    out += esc(text.slice(pos, e.start)) + `<mark class="ent-${e.label}" title="${e.label}: ${esc(e.norm)}">${esc(text.slice(e.start, e.end))}</mark>`;
    pos = e.end;
  }
  return out + esc(text.slice(pos));
}

function card(r) {
  const f = r.fields || {};
  const facts = [];
  if (f.course) facts.push(`📘 ${esc(f.course)}`);
  if (f.date) facts.push(`📅 ${fmtDate(f.date)}${f.date2 ? " → " + fmtDate(f.date2) : ""}`);
  if (f.time) facts.push(`⏰ ${esc(f.time)}${f.end_time ? "–" + esc(f.end_time) : ""}`);
  if (f.location) facts.push(`📍 ${esc(f.location)}${f.old_location ? ` (önceki: ${esc(f.old_location)})` : ""}`);
  return `<article class="card" id="card-${esc(r.id)}" style="--c: var(--${r.category})">
    <div class="metaline"><span class="badge" style="--c: var(--${r.category})">${CAT_TR[r.category] || r.category}</span>
      ${esc(SRC_TR[r.source] || r.source)} · yayın ${fmtDate(r.published)} · güven ${(r.confidence * 100).toFixed(0)}% · ${esc(r.id)}</div>
    <h4>${esc(r.title)}${r.url ? ` <a href="${esc(r.url)}" target="_blank" rel="noopener" title="Orijinal duyuru">↗</a>` : ""}</h4>
    <div class="summary">${esc(r.summary)}</div>
    <div class="facts">${facts.map((x) => `<span class="fact">${x}</span>`).join("")}</div>
    <details><summary>Tam metin (çıkarılan bilgiler işaretli)</summary><p>${highlight(r.text, r.entities || [])}</p></details>
  </article>`;
}

async function loadList() {
  const p = new URLSearchParams({ category: state.category, q: state.q, upcoming: state.upcoming ? "1" : "" });
  const rows = await api(`/api/announcements?${p}`);
  $("#list").innerHTML = rows.length ? rows.map(card).join("") : `<p class="legend">Duyuru bulunamadı.</p>`;
}

function addMsg(html, who = "bot") {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.innerHTML = html;
  $("#messages").appendChild(div);
  $("#messages").scrollTop = 1e9;
  return div;
}

async function focusCard(id) {
  state = { category: "", q: "", upcoming: false };
  $("#search").value = ""; $("#upcoming").checked = false; renderFilters();
  await loadList();
  const el = document.getElementById(`card-${id}`);
  if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 2000); }
}

async function ask(question) {
  addMsg(esc(question), "user");
  const pending = addMsg("…");
  try {
    const res = await api("/api/ask", { method: "POST", body: JSON.stringify({ question, mode: $("#llmMode").checked ? "llm" : "rules" }) });
    const seen = new Set();
    const srcs = res.sources.filter((s) => !seen.has(s.id) && seen.add(s.id)).slice(0, 8);
    pending.innerHTML = esc(res.answer) +
      (srcs.length ? `<div class="src">Kaynaklar: ${srcs.map((s) => `<a data-id="${esc(s.id)}">${esc(s.title)}</a>`).join(" · ")}</div>` : "") +
      `<div class="tag">${res.mode === "llm" ? "LLM (Claude) cevabı" : "Kural tabanlı cevap"}</div>`;
    pending.querySelectorAll("a[data-id]").forEach((a) => a.addEventListener("click", () => focusCard(a.dataset.id)));
  } catch (e) { pending.textContent = "Hata: " + e.message; }
}

function renderFilters() {
  const cats = [["", "Tümü"], ...Object.entries(CAT_TR)];
  $("#filters").innerHTML = cats.map(([k, v]) => `<button class="chip ${state.category === k ? "active" : ""}" data-k="${k}">${v}</button>`).join("");
  $("#filters").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { state.category = b.dataset.k; renderFilters(); loadList(); }));
}

function renderAnalysis(r) {
  const f = r.fields || {};
  $("#analysis").innerHTML = `<p><b>Kategori:</b> ${CAT_TR[r.category]} (${(r.confidence * 100).toFixed(0)}%)<br>
    <b>Özet:</b> ${esc(r.summary)}<br><b>Alanlar:</b> <code>${esc(JSON.stringify(f, null, 0))}</code></p>
    <p>${highlight(r.text, r.entities)}</p>
    <p class="legend"><mark class="ent-DATE">tarih</mark> <mark class="ent-TIME">saat</mark> <mark class="ent-LOCATION">yer</mark> <mark class="ent-COURSE">ders kodu</mark></p>`;
}

async function init() {
  const s = await api("/api/stats");
  $("#meta").innerHTML = `${s.total} duyuru · Referans tarih: <b>${fmtDate(s.today)}</b><br>Sınıflandırıcı: ${esc(s.classifier)} · Çıkarım: ${esc(s.extractor)}`;
  if (s.llm_available) { $("#llmMode").disabled = false; }
  $("#examples").innerHTML = EXAMPLES.map((e) => `<button class="chip">${esc(e)}</button>`).join("");
  $("#examples").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
  renderFilters();
  loadList();
}

$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#question").value.trim(); if (q) { $("#question").value = ""; ask(q); } });
let t;
$("#search").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { state.q = e.target.value; loadList(); }, 250); });
$("#upcoming").addEventListener("change", (e) => { state.upcoming = e.target.checked; loadList(); });
$("#openAdd").addEventListener("click", () => { $("#analysis").innerHTML = ""; $("#addDialog").showModal(); });
$("#analyzeBtn").addEventListener("click", async () => {
  try { renderAnalysis(await api("/api/analyze", { method: "POST", body: JSON.stringify({ title: $("#addTitle").value, text: $("#addText").value }) })); }
  catch (e) { $("#analysis").textContent = "Hata: " + e.message; }
});
$("#saveBtn").addEventListener("click", async () => {
  try {
    const r = await api("/api/announcements", { method: "POST", body: JSON.stringify({ title: $("#addTitle").value, text: $("#addText").value }) });
    renderAnalysis(r);
    $("#addTitle").value = ""; $("#addText").value = "";
    await loadList(); init();
  } catch (e) { $("#analysis").textContent = "Hata: " + e.message; }
});
init();

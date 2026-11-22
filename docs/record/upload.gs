// Google Apps Script that hands out speaker numbers and saves clips from the stride recording page
// to Google Drive.
//
// Setup (once): script.google.com -> New project -> paste this file -> Project Settings ->
// Script properties:
//   FOLDER_ID  the id of a Drive folder you own (the part of its URL after /folders/)
//   KEY        the secret word at the end of the shared link: .../record/?k=<KEY>
//   CODES      (optional) extra codes you use with ?code=, comma-separated, e.g. S01
// Then Deploy -> New deployment -> Web app -> Execute as: Me, Who has access: Anyone.
// Paste the web-app URL into UPLOAD_URL in index.html.
//
// Friends type their first and last name and get
// the next free number (S02, S03, ...). The same name always gets the same number back. Names are
// stored only in the ROSTER script property and mirrored to _roster.txt in the folder; clips are
// saved under the number. A re-recorded clip replaces the old one, and "restart" moves a speaker's
// clips to the Drive trash (recoverable for 30 days).
//
// The page source (and so this script's URL) is public, so every request must carry KEY. Uploads are
// also limited to numbers this script handed out, small WAV files, and MAX_SPEAKERS people. To stop
// collecting, archive the deployment (Deploy -> Manage deployments).

const MAX_WAV_BYTES = 3 * 1024 * 1024;   // ~15 s at 48 kHz, 16-bit mono is ~1.4 MB
const MAX_CLIPS_PER_CODE = 400;
const MAX_SPEAKERS = 100;
const FIRST_AUTO_NUMBER = 2;              // S01 is the project owner
const NAME_RE = /^S\d{2,3}_[A-Za-z0-9-]{1,40}_[a-z]\d{2}$/;
// First and last name, lowercase, single spaces: "maya chen", "jo-ann o'neil", "maya r chen".
const PERSON_RE = /^(?=.{3,60}$)[\p{L}][\p{L}.'-]*( [\p{L}][\p{L}.'-]*)+$/u;

function doPost(e) {
  try {
    const props = PropertiesService.getScriptProperties();
    const req = JSON.parse(e.postData.contents);
    if (!props.getProperty("KEY") || req.key !== props.getProperty("KEY")) return reply({ ok: false, error: "bad key" });
    if (req.action === "claim") return claim(props, req);
    if (!allowedCodes(props).has(req.code)) return reply({ ok: false, error: "unknown speaker code" });
    if (req.action === "reset") return reset(props, req);
    return save(props, req);
  } catch (err) {
    return reply({ ok: false, error: "server error" });
  }
}

function roster(props) {
  return JSON.parse(props.getProperty("ROSTER") || "{}");
}

function allowedCodes(props) {
  const manual = (props.getProperty("CODES") || "").split(",").map((s) => s.trim()).filter(String);
  const r = roster(props);
  return new Set(manual.concat(Object.keys(r).map((n) => r[n])));
}

// Give this name its number: the existing one if the name was seen before, else the next free.
// For a known name, also say how many clips it already has, so the page can ask "is that you?".
function claim(props, req) {
  const name = String(req.name || "").trim().toLowerCase().replace(/\s+/g, " ");
  if (!PERSON_RE.test(name)) return reply({ ok: false, error: "bad name" });
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const r = roster(props);
    if (r[name]) {
      const clips = Number(props.getProperty("N_" + r[name]) || 0);
      return reply({ ok: true, code: r[name], existing: true, clips: clips });
    }
    const taken = allowedCodes(props);
    if (Object.keys(r).length >= MAX_SPEAKERS) return reply({ ok: false, error: "full" });
    let n = FIRST_AUTO_NUMBER;
    while (taken.has(code(n))) n += 1;
    r[name] = code(n);
    props.setProperty("ROSTER", JSON.stringify(r));
    writeRoster(props, r);
    return reply({ ok: true, code: r[name] });
  } finally {
    lock.releaseLock();
  }
}

function code(n) {
  return "S" + (n < 10 ? "0" + n : String(n));
}

// A readable copy of the roster in the Drive folder (number = name), for the owner only.
function writeRoster(props, r) {
  const root = DriveApp.getFolderById(props.getProperty("FOLDER_ID"));
  const lines = Object.keys(r).sort((a, b) => r[a].localeCompare(r[b])).map((n) => r[n] + " = " + n);
  const text = "Private. Never commit or share.\n\n" + lines.join("\n") + "\n";
  const it = root.getFilesByName("_roster.txt");
  if (it.hasNext()) it.next().setContent(text);
  else root.createFile("_roster.txt", text, "text/plain");
}

function save(props, req) {
  if (!NAME_RE.test(req.name) || req.name.indexOf(req.code + "_") !== 0) {
    return reply({ ok: false, error: "bad clip name" });
  }
  const bytes = Utilities.base64Decode(req.wav);
  if (bytes.length > MAX_WAV_BYTES || bytes.length < 44) return reply({ ok: false, error: "bad size" });
  if (Utilities.newBlob(bytes.slice(0, 4)).getDataAsString() !== "RIFF") {
    return reply({ ok: false, error: "not a wav file" });
  }
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const root = DriveApp.getFolderById(props.getProperty("FOLDER_ID"));
    const folder = subfolder(root, req.code);
    const replacing = trash(folder, req.name + ".wav");
    const countKey = "N_" + req.code;
    const count = Number(props.getProperty(countKey) || 0);
    if (!replacing && count >= MAX_CLIPS_PER_CODE) return reply({ ok: false, error: "clip limit reached" });
    trash(folder, req.name + ".json");
    folder.createFile(Utilities.newBlob(bytes, "audio/wav", req.name + ".wav"));
    folder.createFile(req.name + ".json", JSON.stringify(req.meta, null, 1), "application/json");
    if (!replacing) props.setProperty(countKey, String(count + 1));
  } finally {
    lock.releaseLock();
  }
  return reply({ ok: true });
}

// "Restart" on the page: move this speaker's clips from one session to the Drive trash
// (recoverable for 30 days) and give back their clip quota. The number stays theirs.
function reset(props, req) {
  if (!/^[A-Za-z0-9-]{1,40}$/.test(String(req.session))) return reply({ ok: false, error: "bad session" });
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const root = DriveApp.getFolderById(props.getProperty("FOLDER_ID"));
    const it = root.getFoldersByName(req.code);
    if (!it.hasNext()) return reply({ ok: true, removed: 0 });
    const prefix = req.code + "_" + req.session + "_";
    const files = it.next().getFiles();
    let removed = 0;
    while (files.hasNext()) {
      const f = files.next();
      if (f.getName().indexOf(prefix) !== 0) continue;
      if (/\.wav$/.test(f.getName())) removed += 1;
      f.setTrashed(true);
    }
    const countKey = "N_" + req.code;
    props.setProperty(countKey, String(Math.max(0, Number(props.getProperty(countKey) || 0) - removed)));
    return reply({ ok: true, removed: removed });
  } finally {
    lock.releaseLock();
  }
}

function subfolder(root, name) {
  const it = root.getFoldersByName(name);
  return it.hasNext() ? it.next() : root.createFolder(name);
}

function trash(folder, name) {
  const it = folder.getFilesByName(name);
  let found = false;
  while (it.hasNext()) { it.next().setTrashed(true); found = true; }
  return found;
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

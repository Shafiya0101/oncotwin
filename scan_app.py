"""OncoTwin — local clinician-facing scan app.

A local web app where you upload a patient's scan (NIfTI .nii/.nii.gz, or a .zip
of DICOM slices), it segments the tumour and extracts radiomics, stores the
patient and every scan in a small database, and recalibrates the twin across all
of that patient's scans over time.

Run:   python scan_app.py      then open http://localhost:8000
Needs: pip install fastapi uvicorn python-multipart SimpleITK

Honest notes:
- Runs locally (has a Python backend + database); it is NOT the GitHub Pages demo.
- Automatic segmentation is a simple method: accurate on phantoms and when you
  also upload a tumour mask, only approximate on raw clinical CT. Upload a mask
  (second file) for correct radiomics on real scans.
- For local testing only; not secured for real patient data.
"""
import os, io, json, zipfile, tempfile, sqlite3, datetime
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from oncotwin import (OncoTwinEngine, PatientFeatures, TreatmentPlan,
                      TreatmentCourse, TreatmentKind, TumorMeasurement)
from oncotwin.imaging import segment_tumor, extract_radiomics

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oncotwin.db")
os.makedirs(os.path.dirname(DB), exist_ok=True)
engine = OncoTwinEngine()
app = FastAPI(title="OncoTwin Scan App")


# ----------------------------- database ----------------------------- #
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER,
            stage INTEGER, histology TEXT, created TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS scans(
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER,
            scan_date TEXT, volume_cm3 REAL, heterogeneity REAL,
            radiomics TEXT, filename TEXT, uploaded TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id))""")


# ----------------------------- imaging ------------------------------ #
def read_volume(path):
    """Read a NIfTI file or a directory of DICOM slices -> (array z,y,x), spacing."""
    if os.path.isdir(path):
        r = sitk.ImageSeriesReader()
        r.SetFileNames(r.GetGDCMSeriesFileNames(path))
        img = r.Execute()
    else:
        img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(float), img.GetSpacing()


def analyze(scan_bytes, scan_name, mask_bytes=None):
    """Save upload to a temp location, load, segment (or use mask), extract radiomics."""
    with tempfile.TemporaryDirectory() as tmp:
        vol, spacing, used_mask = None, None, None
        if scan_name.lower().endswith(".zip"):
            zp = os.path.join(tmp, "s.zip"); open(zp, "wb").write(scan_bytes)
            dcm = os.path.join(tmp, "dcm"); os.makedirs(dcm)
            with zipfile.ZipFile(zp) as z: z.extractall(dcm)
            # find the folder that actually holds the DICOMs
            folder = next((root for root, _, files in os.walk(dcm)
                           if any(f.lower().endswith(".dcm") for f in files)), dcm)
            vol, spacing = read_volume(folder)
        else:
            ext = ".nii.gz" if scan_name.lower().endswith(".nii.gz") else ".nii"
            sp = os.path.join(tmp, "scan" + ext); open(sp, "wb").write(scan_bytes)
            vol, spacing = read_volume(sp)

        if mask_bytes:
            mext = ".nii.gz" if scan_name.lower().endswith(".gz") else ".nii"
            mp = os.path.join(tmp, "mask" + mext); open(mp, "wb").write(mask_bytes)
            m, _ = read_volume(mp); mask = m > 0
        else:
            mask = segment_tumor(vol)

        if mask.shape != vol.shape or int(mask.sum()) == 0:
            raise ValueError("Segmentation produced an empty or mismatched mask. "
                             "Try uploading a tumour mask file too.")
        feats = extract_radiomics(vol, mask, spacing)
        return feats.as_dict()


# ----------------------------- twin --------------------------------- #
def patient_twin(pid):
    with db() as con:
        prow = con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        scans = con.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY scan_date",
                            (pid,)).fetchall()
    if not prow:
        raise HTTPException(404, "patient not found")
    scans = [dict(s) for s in scans]
    result = {"patient": dict(prow), "scans": scans, "twin": None}
    if not scans:
        return result

    base_date = datetime.date.fromisoformat(scans[0]["scan_date"])
    for s in scans:
        s["day"] = (datetime.date.fromisoformat(s["scan_date"]) - base_date).days
    baseline_vol = scans[0]["volume_cm3"]
    het = scans[0]["heterogeneity"]
    features = PatientFeatures(age=prow["age"] or 60, stage=prow["stage"] or 3,
                               histology=prow["histology"] or "nsclc",
                               baseline_volume_cm3=baseline_vol, ki67=0.3,
                               radiomic_heterogeneity=het if het is not None else 0.5)
    twin = engine.create_twin(str(pid), features)
    measurements = [TumorMeasurement(s["day"], s["volume_cm3"]) for s in scans if s["day"] > 0]
    if measurements:
        engine.assimilate(twin, measurements)
    belief = engine.explain(twin)
    plan = TreatmentPlan("Chemo + radiotherapy", [
        TreatmentCourse(TreatmentKind.CHEMO, 30, 200, 1.0),
        TreatmentCourse(TreatmentKind.RADIO, 30, 75, 1.0)])
    F = engine.forecast(twin, plan, horizon_days=365)
    fc = F.summary(365)
    step = max(1, len(F.t) // 90)
    result["twin"] = {
        "doubling_time_days": round(belief["implied_doubling_time_days"], 1),
        "version": belief["version"], "n_scans": len(scans),
        "chemo_sensitivity": round(belief["chemo_sensitivity"], 4),
        "baseline_volume": round(baseline_vol, 1),
        "forecast_median_1yr": round(fc["volume_median"], 1),
        "forecast_ci90": [round(fc["volume_ci90"][0], 1), round(fc["volume_ci90"][1], 1)],
        "t": [float(x) for x in F.t[::step]],
        "median": [round(float(x), 1) for x in F.median[::step]],
        "lo": [round(float(x), 1) for x in F.lower[::step]],
        "hi": [round(float(x), 1) for x in F.upper[::step]],
        "points": [{"day": s["day"], "vol": round(s["volume_cm3"], 1)} for s in scans],
    }
    return result


# ----------------------------- API ---------------------------------- #
@app.post("/api/patients")
async def create_patient(name: str = Form(...), age: int = Form(...),
                         stage: int = Form(...), histology: str = Form("nsclc")):
    with db() as con:
        cur = con.execute("INSERT INTO patients(name,age,stage,histology,created) VALUES(?,?,?,?,?)",
                          (name, age, stage, histology, datetime.datetime.now().isoformat()))
        return {"id": cur.lastrowid, "name": name}


@app.get("/api/patients")
async def list_patients():
    with db() as con:
        rows = con.execute("""SELECT p.*, COUNT(s.id) AS n_scans FROM patients p
            LEFT JOIN scans s ON s.patient_id=p.id GROUP BY p.id ORDER BY p.id DESC""").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/patients/{pid}")
async def get_patient(pid: int):
    return patient_twin(pid)


@app.post("/api/patients/{pid}/scans")
async def upload_scan(pid: int, scan_date: str = Form(...),
                      file: UploadFile = File(...), mask: UploadFile = File(None)):
    try:
        scan_bytes = await file.read()
        mask_bytes = await mask.read() if mask is not None else None
        feats = analyze(scan_bytes, file.filename, mask_bytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    with db() as con:
        con.execute("""INSERT INTO scans(patient_id,scan_date,volume_cm3,heterogeneity,
            radiomics,filename,uploaded) VALUES(?,?,?,?,?,?,?)""",
            (pid, scan_date, feats["volume_cm3"], feats.get("heterogeneity"),
             json.dumps(feats), file.filename, datetime.datetime.now().isoformat()))
    return {"ok": True, "radiomics": feats}


# ----------------------------- UI ----------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def home():
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OncoTwin — Clinical</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--ink:#0E1416;--panel:#161E21;--panel-2:#111A1D;--line:#26343A;--text:#E4EAEC;--muted:#7C8B90;--faint:#4A5B61;--cyan:#35D0BA;--cyan-dim:#1c6d63;--coral:#FF7A59;--coral-dim:#7d3826;--amber:#F2B441;}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--ink);color:var(--text);font-family:"Inter",system-ui,sans-serif;font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:20px 20px 48px}
.status{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:14px 18px;border:1px solid var(--line);border-radius:12px;background:linear-gradient(180deg,var(--panel),var(--panel-2))}
.brand{font-family:"Space Grotesk",sans-serif;font-weight:600;font-size:18px}.brand span{color:var(--cyan)}
.pid{font-family:"IBM Plex Mono",monospace;color:var(--muted);font-size:13px}
.pill{font-family:"IBM Plex Mono",monospace;font-size:11px;padding:4px 9px;border-radius:20px;border:1px solid var(--cyan-dim);color:var(--cyan)}.pill::before{content:"● "}
.chips{margin-left:auto;display:flex;gap:10px;flex-wrap:wrap}
.chip{display:flex;flex-direction:column;gap:2px;padding:7px 12px;border:1px solid var(--line);border-radius:9px;background:#0d1517;min-width:92px}
.chip .k{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}.chip .v{font-family:"IBM Plex Mono",monospace;font-size:15px}
.chip.v-ver .v{color:var(--cyan)}
.grid{display:grid;grid-template-columns:1.35fr 1fr;gap:16px;margin-top:16px}
.card{border:1px solid var(--line);border-radius:12px;background:var(--panel);overflow:hidden;margin-bottom:16px}
.card h2{margin:0;padding:14px 18px 0;font-family:"Space Grotesk",sans-serif;font-size:13px;font-weight:500;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
.stage-card{position:relative;min-height:380px;display:flex;flex-direction:column}
#stage{flex:1;min-height:320px;position:relative}#stage canvas{display:block}
.readout{position:absolute;left:18px;bottom:16px;pointer-events:none}
.readout .k{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
.readout .vol{font-family:"IBM Plex Mono",monospace;font-size:32px;color:var(--coral);line-height:1.05;margin-top:2px}.readout .vol small{font-size:14px;color:var(--muted)}
.leg{position:absolute;right:18px;top:16px;text-align:right;font-size:11px;color:var(--faint);line-height:1.7}.leg i{color:var(--coral);font-style:normal}
.fc{display:flex;gap:18px;padding:10px 18px 16px}
.fc .k{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}.fc .v{font-family:"IBM Plex Mono",monospace;font-size:22px;margin-top:2px}.fc .v.med{color:var(--coral)}.fc .v.ci{font-size:15px}
#chartWrap{padding:6px 10px 14px}#chart{width:100%;display:block}
label{display:block;font-size:11px;color:var(--muted);margin:10px 0 5px;font-family:"Space Grotesk",sans-serif;text-transform:uppercase;letter-spacing:.05em}
input,button{font:inherit;padding:9px 10px;border-radius:8px;border:1px solid var(--line);background:#0f181b;color:var(--text);width:100%}
.row2{display:flex;gap:10px}.row2 input{width:100%}
button{background:var(--cyan);color:var(--ink);font-family:"Space Grotesk",sans-serif;font-weight:600;border:none;cursor:pointer;width:auto;padding:9px 16px;margin-top:10px}button:hover{background:#4fe0cb}
.body{padding:6px 18px 18px}
.pat{padding:9px 12px;border:1px solid var(--line);border-radius:8px;margin-top:8px;cursor:pointer;background:#0f181b;font-size:14px}
.pat:hover{border-color:var(--cyan-dim)}.pat.sel{border-color:var(--cyan);background:#0c211f}
.pat .s{color:var(--muted);font-size:12px;font-family:"IBM Plex Mono",monospace}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:7px 4px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:500}td{font-family:"IBM Plex Mono",monospace}
.msg{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--cyan);margin-top:8px}.msg.warn{color:var(--amber)}
.hint{color:var(--faint);font-size:12px;margin-top:6px}
footer{margin-top:20px;text-align:center;color:var(--faint);font-size:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="wrap">
  <div class="status">
    <div class="brand">Onco<span>Twin</span></div>
    <div class="pid" id="who">— no patient selected —</div>
    <span class="pill">local clinical</span>
    <div class="chips">
      <div class="chip"><span class="k">Doubling time</span><span class="v" id="chDbl">— d</span></div>
      <div class="chip"><span class="k">Chemo response</span><span class="v" id="chResp">—</span></div>
      <div class="chip v-ver"><span class="k">Twin version</span><span class="v" id="chVer">v0</span></div>
    </div>
  </div>

  <div class="grid">
    <div>
      <div class="card stage-card">
        <h2>Tumour volume · from uploaded scan</h2>
        <div id="stage">
          <div class="leg">scan field<br><i>tumour burden</i></div>
          <div class="readout"><div class="k mono" id="roK">baseline</div><div class="vol" id="roVol">—<small> cm³</small></div></div>
        </div>
      </div>
      <div class="card">
        <h2>Forecast trajectory (recalibrated)</h2>
        <div id="chartWrap"><canvas id="chart" height="280"></canvas></div>
        <div class="fc">
          <div><div class="k">Median @ 1 year</div><div class="v med" id="fcMed">—</div></div>
          <div><div class="k">90% interval</div><div class="v ci" id="fcCI">—</div></div>
        </div>
      </div>
    </div>

    <div>
      <div class="card"><h2>Patients</h2><div class="body">
        <div class="row2"><input id="pname" placeholder="Name"><input id="page" type="number" value="63" title="age"></div>
        <div class="row2"><input id="pstage" type="number" value="3" min="1" max="4" title="stage (1-4)"><button onclick="addPatient()">Add patient</button></div>
        <div id="plist"></div>
      </div></div>

      <div class="card" id="upcard" style="display:none"><h2>Upload scan</h2><div class="body">
        <label>Scan date</label><input id="sdate" type="date">
        <label>Scan file (.nii, .nii.gz, or .zip of DICOM)</label><input id="sfile" type="file">
        <label>Tumour mask — optional, .nii/.nii.gz (recommended for real CT)</label><input id="smask" type="file">
        <button onclick="uploadScan()">Upload &amp; analyse</button>
        <div class="msg" id="upmsg"></div>
        <div class="hint">Demo scans are in data/demo_scans/. Real CT without a mask is segmented approximately.</div>
      </div></div>

      <div class="card" id="scancard" style="display:none"><h2>Scans over time</h2><div class="body">
        <table id="scantbl"><thead><tr><th>Date</th><th>Day</th><th>Volume</th><th>Heterog.</th></tr></thead><tbody></tbody></table>
      </div></div>
    </div>
  </div>
  <footer>OncoTwin — local clinical tool. Illustrative model, not a medical device.</footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
let current=null, fc=null;
const stageEl=document.getElementById('stage');
let renderer,scene,camera,tumor,field,tScale=0.5,tTarget=0.5;
function volToScale(v){return Math.min(2.1,Math.max(0.22,0.9*Math.cbrt((v||40)/200)));}
function initThree(){
  const w=stageEl.clientWidth,h=stageEl.clientHeight||320;
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(w,h);stageEl.appendChild(renderer.domElement);
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(45,w/h,0.1,100);camera.position.set(0,0,5.6);
  field=new THREE.Mesh(new THREE.IcosahedronGeometry(2.5,1),new THREE.MeshBasicMaterial({color:0x35D0BA,wireframe:true,transparent:true,opacity:0.10}));scene.add(field);
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(2.52,32,24),new THREE.MeshBasicMaterial({color:0x35D0BA,transparent:true,opacity:0.03})));
  const g=new THREE.SphereGeometry(1,64,48),p=g.attributes.position;
  const nz=(x,y,z)=>0.5*Math.sin(2.5*x)*Math.sin(2.9*y)*Math.sin(2.2*z);
  for(let i=0;i<p.count;i++){const x=p.getX(i),y=p.getY(i),z=p.getZ(i),d=1+0.12*nz(x,y,z);p.setXYZ(i,x*d,y*d,z*d);}g.computeVertexNormals();
  tumor=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xFF7A59,emissive:0xFF4A2A,emissiveIntensity:0.35,roughness:0.55,metalness:0.05}));scene.add(tumor);
  scene.add(new THREE.AmbientLight(0x9fb4ba,0.6));
  const k=new THREE.DirectionalLight(0xffffff,0.9);k.position.set(3,4,5);scene.add(k);
  const wl=new THREE.PointLight(0xFF7A59,0.7,20);wl.position.set(-4,-2,3);scene.add(wl);
  animate();
}
function animate(){requestAnimationFrame(animate);tScale+=(tTarget-tScale)*0.12;
  if(tumor){tumor.scale.setScalar(tScale);tumor.rotation.y+=0.0035;tumor.rotation.x+=0.0012;field.rotation.y-=0.0016;tumor.material.emissiveIntensity=0.30+0.08*Math.sin(performance.now()*0.0016);}
  renderer.render(scene,camera);}
function resizeThree(){if(!renderer)return;const w=stageEl.clientWidth,h=stageEl.clientHeight||320;if(w>0){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}}

function chartCtx(){const cv=document.getElementById('chart'),wrap=document.getElementById('chartWrap');
  const CW=wrap.clientWidth-20,CH=cv.height,dpr=Math.min(devicePixelRatio,2);
  cv.style.width=CW+'px';cv.style.height=CH+'px';cv.width=CW*dpr;cv.height=CH*dpr;
  const ctx=cv.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,CW,CH};}
function drawChart(){const {ctx,CW,CH}=chartCtx();if(!CW)return;
  ctx.clearRect(0,0,CW,CH);
  if(!fc||!fc.t||!fc.t.length){ctx.fillStyle="#4A5B61";ctx.font='13px Inter';ctx.textAlign="center";ctx.fillText("Upload a scan to see the forecast",CW/2,CH/2);return;}
  let ymax=Math.max.apply(null,fc.hi);ymax=Math.max(200,Math.ceil(ymax/200)*200);
  const padL=48,padR=14,padT=12,padB=26,x0=padL,x1=CW-padR,y0=CH-padB,y1=padT;
  const X=d=>x0+(Math.min(d,365)/365)*(x1-x0),Y=v=>y0+(v/ymax)*(y1-y0);
  ctx.strokeStyle="#202c31";ctx.fillStyle="#5a6b70";ctx.lineWidth=1;ctx.font='11px "IBM Plex Mono"';ctx.textAlign="right";ctx.textBaseline="middle";
  for(let g=0;g<=ymax;g+=ymax/4){ctx.beginPath();ctx.moveTo(x0,Y(g));ctx.lineTo(x1,Y(g));ctx.stroke();ctx.fillText(g.toFixed(0),x0-6,Y(g));}
  ctx.textAlign="center";ctx.textBaseline="top";for(let d=0;d<=365;d+=73)ctx.fillText(d,X(d),y0+6);
  ctx.beginPath();ctx.moveTo(X(fc.t[0]),Y(fc.lo[0]));
  for(let i=1;i<fc.t.length;i++)ctx.lineTo(X(fc.t[i]),Y(fc.lo[i]));
  for(let i=fc.t.length-1;i>=0;i--)ctx.lineTo(X(fc.t[i]),Y(fc.hi[i]));ctx.closePath();ctx.fillStyle="rgba(255,122,89,0.16)";ctx.fill();
  ctx.beginPath();for(let i=0;i<fc.t.length;i++){const px=X(fc.t[i]),py=Y(fc.median[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}ctx.strokeStyle="#FF7A59";ctx.lineWidth=2.4;ctx.stroke();
  (fc.points||[]).forEach(pt=>{ctx.beginPath();ctx.arc(X(pt.day),Y(pt.vol),4.5,0,7);ctx.fillStyle="#35D0BA";ctx.fill();ctx.lineWidth=2;ctx.strokeStyle="#0E1416";ctx.stroke();});
}

async function loadPatients(){const ps=await (await fetch('/api/patients')).json();
  document.getElementById('plist').innerHTML=ps.map(p=>`<div class="pat${p.id===current?' sel':''}" onclick="selectPatient(${p.id})">${p.name} <span class="s">· age ${p.age} · stage ${p.stage} · ${p.n_scans} scan(s)</span></div>`).join('')||'<div class="hint">No patients yet — add one above.</div>';}
async function addPatient(){const fd=new FormData();
  fd.append('name',document.getElementById('pname').value||'Patient');
  fd.append('age',document.getElementById('page').value||63);fd.append('stage',document.getElementById('pstage').value||3);
  const r=await (await fetch('/api/patients',{method:'POST',body:fd})).json();
  document.getElementById('pname').value='';await loadPatients();selectPatient(r.id);}
async function selectPatient(id){current=id;await loadPatients();
  document.getElementById('upcard').style.display='block';
  const d=await (await fetch('/api/patients/'+id)).json();
  document.getElementById('who').textContent=d.patient.name+' · age '+d.patient.age+' · stage '+d.patient.stage;
  if(!document.getElementById('sdate').value)document.getElementById('sdate').value=new Date().toISOString().slice(0,10);
  const t=d.twin;
  if(t){fc=t;tTarget=volToScale(t.baseline_volume);
    document.getElementById('chDbl').textContent=Math.round(t.doubling_time_days)+' d';
    document.getElementById('chResp').textContent=t.chemo_sensitivity>0.05?'responder':t.chemo_sensitivity>0.03?'partial':'resistant';
    document.getElementById('chVer').textContent='v'+t.version;
    document.getElementById('roK').textContent=t.n_scans+' scan(s) · v'+t.version;
    document.getElementById('roVol').innerHTML=t.baseline_volume+'<small> cm³</small>';
    document.getElementById('fcMed').textContent=t.forecast_median_1yr+' cm³';
    document.getElementById('fcCI').textContent=t.forecast_ci90[0]+'–'+t.forecast_ci90[1];
    document.getElementById('scancard').style.display='block';
    document.querySelector('#scantbl tbody').innerHTML=d.scans.map(s=>`<tr><td>${s.scan_date}</td><td>${s.day??0}</td><td>${s.volume_cm3.toFixed(1)} cm³</td><td>${(s.heterogeneity??0).toFixed(3)}</td></tr>`).join('');
  }else{fc=null;tTarget=0.4;['chDbl','chResp','fcMed','fcCI'].forEach(i=>document.getElementById(i).textContent='—');document.getElementById('chVer').textContent='v0';document.getElementById('roVol').innerHTML='—<small> cm³</small>';document.getElementById('scancard').style.display='none';}
  drawChart();}
async function uploadScan(){const f=document.getElementById('sfile').files[0];const msg=document.getElementById('upmsg');
  if(!f){msg.className='msg warn';msg.textContent='Choose a scan file first.';return;}
  msg.className='msg';msg.textContent='Analysing… this can take a moment.';
  const fd=new FormData();fd.append('scan_date',document.getElementById('sdate').value);fd.append('file',f);
  const m=document.getElementById('smask').files[0];if(m)fd.append('mask',m);
  const d=await (await fetch('/api/patients/'+current+'/scans',{method:'POST',body:fd})).json();
  if(d.error){msg.className='msg warn';msg.textContent=d.error;return;}
  msg.className='msg';msg.textContent='Done: tumour volume '+d.radiomics.volume_cm3.toFixed(1)+' cm³';
  document.getElementById('sfile').value='';document.getElementById('smask').value='';
  selectPatient(current);}
window.addEventListener('resize',()=>{resizeThree();drawChart();});
initThree();drawChart();loadPatients();
</script>
</body></html>"""

if __name__ == "__main__":
    init_db()
    print("OncoTwin Scan App -> http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

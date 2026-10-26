"""3D surfaces and volumes of the heart chambers from a dense 4D intensity volume.

Pipeline (no network inside this module; it works on any dense volume, e.g. the
neural field queried on a fine grid, or the scanner stack interpolated):

  1. Blood threshold. In cine-MRI (bSSFP) blood is bright and the heart muscle is
     dark. The threshold is the midpoint between the mean intensity of the
     expert-labelled LV cavity and of the myocardium at end-diastole.
  2. Region of interest per chamber: the expert label of that chamber (union of
     ED and ES), dilated in-plane, so neighbouring bright structures are excluded.
     Between two scanned slices the ROI is the union of both neighbours.
  3. Per time point: voxels brighter than the threshold inside the ROI, largest
     connected 3D component -> chamber mask -> volume in ml.
  4. Marching cubes on the slightly smoothed mask -> triangle mesh in millimetres.

Validation: the same procedure on the scanner data gives volumes that can be
compared with the volumes of the expert labels (end-diastolic / end-systolic volume
and ejection fraction).
"""

from __future__ import annotations

import base64
import json

import numpy as np

CHAMBERS = {"LV": 3, "RV": 1}       # ACDC labels: 1 RV cavity, 2 myocardium, 3 LV cavity
STRUCTURES = ("MYO", "LV", "RV")     # drawing order; MYO = left-ventricular muscle (solid, cut open)
COLORS = {"MYO": "#d39a84", "LV": "#c62828", "RV": "#3b6fd1"}
NAMES = {"MYO": "Heart muscle (left ventricle), cut open", "LV": "Blood in the left ventricle",
         "RV": "Blood in the right ventricle"}


def blood_threshold(frame_vol, labels_frame):
    """frame_vol, labels_frame: [H, W, D] on the scanner grid."""
    blood = frame_vol[labels_frame == 3]
    muscle = frame_vol[labels_frame == 2]
    if blood.size == 0 or muscle.size == 0:
        raise ValueError("labels have no LV cavity / myocardium")
    return 0.5 * (float(np.median(blood)) + float(np.median(muscle)))


def chamber_rois(labels: dict, crop, dilate=4):
    """{chamber: bool [h, w, D]} on the scanner slices (inside crop)."""
    from skimage.morphology import dilation, disk

    r0, r1, c0, c1 = crop
    labs = [lab[r0:r1, c0:c1] for lab in labels.values()]
    out = {}
    for name, code in CHAMBERS.items():
        own = np.zeros(labs[0].shape, bool)
        other = np.zeros(labs[0].shape, bool)
        for lab in labs:
            own |= lab == code
            other |= (lab > 0) & (lab != code) & (lab != 2)   # the other cavity
        roi = np.zeros_like(own)
        fp = disk(dilate)
        for z in range(own.shape[2]):
            if own[:, :, z].any():
                roi[:, :, z] = dilation(own[:, :, z].astype(np.uint8), fp) > 0
        out[name] = roi & ~other
    return out


def muscle_band(frame_vol, labels_frame, thr):
    """Intensity range of the heart muscle: from symmetric-below-median to the blood threshold."""
    med = float(np.median(frame_vol[labels_frame == 2]))
    return 2 * med - thr, thr


def wall_roi(labels: dict, crop, dilate=2):
    """Where the LV wall can be: expert LV cavity + myocardium (ED and ES), dilated in-plane."""
    from skimage.morphology import dilation, disk

    r0, r1, c0, c1 = crop
    labs = [lab[r0:r1, c0:c1] for lab in labels.values()]
    u = np.zeros(labs[0].shape, bool)
    for lab in labs:
        u |= (lab == 2) | (lab == 3)
    out = np.zeros_like(u)
    for z in range(u.shape[2]):
        if u[:, :, z].any():
            out[:, :, z] = dilation(u[:, :, z].astype(np.uint8), disk(dilate)) > 0
    return out


def lv_solid(vol3d, lv_blood, roi3d, band):
    """Left ventricle as a solid (blood + surrounding muscle): its outer surface is the outside of
    the heart wall. Muscle = voxels in the muscle intensity band inside the wall ROI."""
    from scipy import ndimage

    lo, hi = band
    # the blood grown by 2 voxels is always part of the solid: the muscle surface then encloses
    # the blood surface everywhere (no blood poking through a thin wall after smoothing)
    solid = ndimage.binary_dilation(lv_blood, iterations=2) | (roi3d & (vol3d > lo) & (vol3d <= hi))
    solid = ndimage.binary_closing(solid, iterations=2)
    for z in range(solid.shape[2]):
        solid[:, :, z] = ndimage.binary_fill_holes(solid[:, :, z])
    lab, n = ndimage.label(solid)
    if n == 0:
        return solid
    sizes = ndimage.sum(solid, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def roi_interp(roi, z_positions, empty=15.0):
    """Shape-based interpolation of a slice-wise mask to fractional slice positions:
    the signed distance to the mask boundary is interpolated linearly between slices, so the
    region changes smoothly from one scanned slice to the next (no steps every 10 mm).
    Slices without the structure count as 'far outside' (distance `empty` px), so the region
    tapers off towards the apex instead of ending with a flat cut."""
    from scipy import ndimage

    D = roi.shape[2]
    sdf = np.empty(roi.shape, np.float32)
    for z in range(D):
        m = roi[:, :, z]
        if m.any():
            sdf[:, :, z] = ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)
        else:
            sdf[:, :, z] = empty
    z = np.clip(np.asarray(z_positions, float), 0, D - 1)
    lo = np.floor(z).astype(int)
    hi = np.minimum(lo + 1, D - 1)
    w = (z - lo).astype(np.float32)
    return ((1 - w) * sdf[:, :, lo] + w * sdf[:, :, hi]) < 0


def roi_at(roi, z_positions):
    """ROI [h, w, D] sampled at fractional slice positions (union of both neighbours)."""
    D = roi.shape[2]
    z = np.asarray(z_positions, float)
    lo = np.clip(np.floor(z).astype(int), 0, D - 1)
    hi = np.clip(np.ceil(z).astype(int), 0, D - 1)
    return roi[:, :, lo] | roi[:, :, hi]


def segment(vol3d, roi3d, thr):
    """Largest bright connected component inside the ROI."""
    from scipy import ndimage

    mask = (vol3d > thr) & roi3d
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def taubin(verts, faces, iterations=10, lam=0.5, mu=-0.53):
    """Taubin smoothing: removes the staircase of the voxel grid without shrinking the surface."""
    from scipy import sparse

    n = len(verts)
    if n == 0:
        return verts
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    v = verts.astype(np.float64)
    for _ in range(iterations):
        for f in (lam, mu):
            v = v + f * (A @ v / deg[:, None] - v)
    return v.astype(np.float32)


def mesh(mask, spacing, step=1, smooth=1.0, polish=10):
    """Marching cubes -> (verts [n, 3] in mm, faces [m, 3])."""
    from scipy import ndimage
    from skimage.measure import marching_cubes

    if mask.sum() < 10:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    m = np.pad(mask.astype(np.float32), 2)
    if smooth > 0:
        m = ndimage.gaussian_filter(m, smooth)
    verts, faces, _, _ = marching_cubes(m, 0.5, spacing=tuple(float(s) for s in spacing), step_size=step)
    verts -= 2 * np.asarray(spacing, np.float32)
    if polish:
        verts = taubin(verts, faces, polish)
    return verts.astype(np.float32), faces.astype(np.int32)


def display_coords(verts):
    """(row, col, slice) mm -> display frame: x = row, y = up = -slice (base on top, apex down),
    z = col (towards the viewer)."""
    v = np.asarray(verts, np.float32)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) if len(v) else v


def save_obj(path, verts, faces):
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
        for t in faces + 1:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")


def ejection_fraction(volumes):
    v = np.asarray(volumes, float)
    edv, esv = v.max(), v.min()
    return edv, esv, 100.0 * (edv - esv) / edv if edv > 0 else float("nan")


# --------------------------------------------------------------------------- HTML viewer

def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def write_viewer(path, frames, volumes, times, title, subtitle, stats, caps=None, cap_geom=None):
    """frames: list over time of {chamber: (verts, faces)}; volumes: {chamber: [ml per frame]}.
    caps: per frame (image uint8 [h, nz], alpha uint8 [h, nz]) = the neural-field image on the cut
    plane, shown where the muscle wall is; cap_geom = dict(col_mm, sx, dz).
    Writes a self-contained HTML page (WebGL, no external libraries, works offline)."""
    frames = [{k: (display_coords(v), f) for k, (v, f) in fr.items()} for fr in frames]
    allv = np.concatenate([v for fr in frames for v, _ in fr.values() if len(v)])
    center = (allv.min(0) + allv.max(0)) / 2
    cap = None
    if caps:
        center[2] = cap_geom["col_mm"]                 # the cut plane is display z = 0
        h, nz = caps[0][0].shape
        sx, dz = cap_geom["sx"], cap_geom["dz"]
        cap = {"w": int(h), "h": int(nz),
               "quad": [float(-0.5 * sx - center[0]), float((h - 0.5) * sx - center[0]),
                        float(0.5 * dz - center[1]), float(-(nz - 0.5) * dz - center[1])],
               "tex": [_b64(np.stack([img.T, a.T], -1).astype(np.uint8)) for img, a in caps]}
    radius = float(np.linalg.norm(allv.max(0) - allv.min(0)) / 2)
    data = []
    for fr in frames:
        item = {}
        for name, (v, f) in fr.items():
            q = np.round((v - center) * 20).astype(np.int16)          # 0.05 mm units
            small = len(v) < 65536
            item[name] = {"v": _b64(q), "f": _b64(f.astype(np.uint16 if small else np.uint32)),
                          "s": int(small), "n": int(len(f))}
        data.append(item)
    payload = {"frames": data, "volumes": {k: [round(float(x), 2) for x in v] for k, v in volumes.items()},
               "times": [round(float(t)) for t in times], "radius": radius,
               "colors": COLORS, "labels": NAMES, "cap": cap, "title": title, "subtitle": subtitle, "stats": stats}
    html = _VIEWER.replace("__DATA__", json.dumps(payload)).replace("__TITLE__", title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


_VIEWER = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--accent:#e0524a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--panel:#1d1d20;--ink:#ececef;--muted:#9a9aa2;--line:#2c2c31}}
:root[data-theme="dark"]{--bg:#141416;--panel:#1d1d20;--ink:#ececef;--muted:#9a9aa2;--line:#2c2c31}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 2px}p.sub{color:var(--muted);margin:0 0 16px}
.grid{display:grid;grid-template-columns:1fr 340px;gap:16px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
canvas#gl{width:100%;aspect-ratio:1/1;display:block;border-radius:8px;cursor:grab;touch-action:none;background:radial-gradient(circle at 50% 40%,#2a2a30,#0d0d10)}
.controls{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:14px;color:var(--muted)}
.legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:5px 4px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left}th{color:var(--muted);font-weight:500}
svg text{fill:var(--muted);font-size:11px}.hint{color:var(--muted);font-size:13px;margin-top:6px}
.big{font-size:28px;font-weight:600}
</style></head><body><main>
<h1 id="t"></h1><p class="sub" id="s"></p>
<div class="grid">
 <div class="card"><canvas id="gl"></canvas>
  <div class="controls"><button id="play">Pause</button><input id="slider" type="range" min="0" value="0">
   <span id="phase" style="min-width:110px;color:var(--muted)"></span></div>
  <div class="hint">Drag to rotate, scroll or pinch to zoom.</div></div>
 <div>
  <div class="card legend" id="legend"></div>
  <div class="card" style="margin-top:12px"><div style="color:var(--muted);font-size:13px">Chamber volume over one heartbeat, ml</div>
   <svg id="curve" viewBox="0 0 320 180" width="100%"></svg></div>
  <div class="card" style="margin-top:12px" id="stats"></div>
 </div>
</div></main>
<script>
const D=__DATA__;
document.getElementById('t').textContent=D.title;document.getElementById('s').textContent=D.subtitle;
const names=Object.keys(D.frames[0]);
document.getElementById('legend').innerHTML=names.map(n=>`<label style="display:flex;gap:8px;align-items:center;margin:3px 0"><input type="checkbox" checked data-k="${n}"><i style="background:${D.colors[n]}"></i>${D.labels[n]}</label>`).join('')+
 `<label style="display:flex;gap:8px;align-items:center;margin:8px 0 0"><input type="checkbox" checked id="cut">Cut the muscle open: the cut shows the network's own image of the wall</label>`;
// stats table
(function(){const s=D.stats;let h='<table><tr><th></th>'+s.cols.map(c=>`<th>${c}</th>`).join('')+'</tr>';
for(const r of s.rows)h+='<tr><td>'+r[0]+'</td>'+r.slice(1).map(x=>`<td>${x}</td>`).join('')+'</tr>';
document.getElementById('stats').innerHTML=h+'</table>'+(s.note?`<div class="hint">${s.note}</div>`:'')})();
function dec(b,T){const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer)}
const cv=document.getElementById('gl'),gl=cv.getContext('webgl');
if(!gl){cv.replaceWith(Object.assign(document.createElement('p'),{textContent:'WebGL is not available in this browser.'}))}
const vs=`attribute vec3 p;attribute vec3 n;uniform mat4 M;uniform mat4 P;varying vec3 vn;varying float vz;void main(){vn=mat3(M)*n;vz=p.z;gl_Position=P*M*vec4(p,1.);}`;
const fs=`precision mediump float;varying vec3 vn;varying float vz;uniform vec3 c;uniform float clip;void main(){if(clip>0.5&&vz>0.)discard;vec3 N=normalize(vn);if(!gl_FrontFacing)N=-N;
float d=max(dot(N,normalize(vec3(.4,.6,1.))),0.);float r=pow(max(dot(reflect(-normalize(vec3(.4,.6,1.)),N),vec3(0,0,1)),0.),24.);
gl_FragColor=vec4(c*(.28+.72*d)+.25*r,1.);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const ext=gl.getExtension('OES_element_index_uint');
const L={p:gl.getAttribLocation(pr,'p'),n:gl.getAttribLocation(pr,'n'),M:gl.getUniformLocation(pr,'M'),P:gl.getUniformLocation(pr,'P'),c:gl.getUniformLocation(pr,'c'),clip:gl.getUniformLocation(pr,'clip')};
const show={};for(const k of Object.keys(D.frames[0]))show[k]=true;let cut=true;
function hex(h){return[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)/255)}
const meshes=D.frames.map(fr=>{const o={};for(const k of names){const m=fr[k];const q=dec(m.v,Int16Array),f=dec(m.f,m.s?Uint16Array:Uint32Array);
 const p=new Float32Array(q.length);for(let i=0;i<q.length;i++)p[i]=q[i]/20;const n=new Float32Array(p.length);
 for(let i=0;i<f.length;i+=3){const a=f[i]*3,b=f[i+1]*3,c=f[i+2]*3;const ux=p[b]-p[a],uy=p[b+1]-p[a+1],uz=p[b+2]-p[a+2],vx=p[c]-p[a],vy=p[c+1]-p[a+1],vz=p[c+2]-p[a+2];
  const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;for(const j of[a,b,c]){n[j]+=nx;n[j+1]+=ny;n[j+2]+=nz}}
 const bp=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bp);gl.bufferData(gl.ARRAY_BUFFER,p,gl.STATIC_DRAW);
 const bn=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bn);gl.bufferData(gl.ARRAY_BUFFER,n,gl.STATIC_DRAW);
 const bi=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bi);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,f,gl.STATIC_DRAW);
 o[k]={bp,bn,bi,count:f.length,type:m.s?gl.UNSIGNED_SHORT:gl.UNSIGNED_INT}}return o});
// cut plane: the network's image on the cut, shown where the muscle wall is
let capP=null,capT=[],capB=null,CL={};
if(D.cap){const v2=`attribute vec3 a;attribute vec2 u;uniform mat4 M;uniform mat4 P;varying vec2 vu;void main(){vu=u;gl_Position=P*M*vec4(a,1.);}`;
 const f2=`precision mediump float;varying vec2 vu;uniform sampler2D t;void main(){vec4 s=texture2D(t,vu);if(s.a<0.5)discard;gl_FragColor=vec4(vec3(s.r)*vec3(1.,.93,.9),1.);}`;
 capP=gl.createProgram();gl.attachShader(capP,sh(gl.VERTEX_SHADER,v2));gl.attachShader(capP,sh(gl.FRAGMENT_SHADER,f2));gl.linkProgram(capP);
 CL={a:gl.getAttribLocation(capP,'a'),u:gl.getAttribLocation(capP,'u'),M:gl.getUniformLocation(capP,'M'),P:gl.getUniformLocation(capP,'P'),t:gl.getUniformLocation(capP,'t')};
 const q=D.cap.quad;capB=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,capB);
 gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([q[0],q[2],0,0,0, q[1],q[2],0,1,0, q[0],q[3],0,0,1, q[1],q[3],0,1,1]),gl.STATIC_DRAW);
 gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
 capT=D.cap.tex.map(b=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.LUMINANCE_ALPHA,D.cap.w,D.cap.h,0,gl.LUMINANCE_ALPHA,gl.UNSIGNED_BYTE,dec(b,Uint8Array));
  for(const [k,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_2D,k,v);return t})}
function noAttribs(){for(let i=0;i<4;i++)gl.disableVertexAttribArray(i)}
let rx=0.25,ry=-0.5,zoom=2.4,frame=0,playing=true,last=0;
function mul(a,b){const r=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];r[i*4+j]=s}return r}
function rotX(t){const c=Math.cos(t),s=Math.sin(t);return new Float32Array([1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1])}
function rotY(t){const c=Math.cos(t),s=Math.sin(t);return new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1])}
function draw(){const w=cv.clientWidth*devicePixelRatio,h=cv.clientHeight*devicePixelRatio;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}
 gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
 const R=D.radius,dist=R*zoom,f=1/Math.tan(0.35),a=w/h,nr=Math.max(dist-R*2,R*0.05),fr=dist+R*2;
 const P=new Float32Array([f/a,0,0,0,0,f,0,0,0,0,(fr+nr)/(nr-fr),-1,0,0,2*fr*nr/(nr-fr),0]);
 let M=mul(rotX(rx),rotY(ry));M[14]-=dist;gl.uniformMatrix4fv(L.P,false,P);gl.uniformMatrix4fv(L.M,false,M);
 gl.disable(gl.CULL_FACE);noAttribs();
 for(const k of names){const m=meshes[frame][k];if(!m.count||!show[k])continue;gl.uniform3fv(L.c,hex(D.colors[k]));gl.uniform1f(L.clip,((k==='MYO'||k==='LV')&&cut)?1:0);
  gl.bindBuffer(gl.ARRAY_BUFFER,m.bp);gl.enableVertexAttribArray(L.p);gl.vertexAttribPointer(L.p,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,m.bn);gl.enableVertexAttribArray(L.n);gl.vertexAttribPointer(L.n,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.bi);gl.drawElements(gl.TRIANGLES,m.count,m.type,0)}
 if(capP&&cut&&show.MYO){noAttribs();gl.useProgram(capP);gl.uniformMatrix4fv(CL.P,false,P);gl.uniformMatrix4fv(CL.M,false,M);
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,capT[frame]);gl.uniform1i(CL.t,0);gl.bindBuffer(gl.ARRAY_BUFFER,capB);
  gl.enableVertexAttribArray(CL.a);gl.vertexAttribPointer(CL.a,3,gl.FLOAT,false,20,0);gl.enableVertexAttribArray(CL.u);gl.vertexAttribPointer(CL.u,2,gl.FLOAT,false,20,12);
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);noAttribs();gl.useProgram(pr)}}
document.querySelectorAll('#legend input[data-k]').forEach(el=>el.onchange=()=>{show[el.dataset.k]=el.checked;draw()});
document.getElementById('cut').onchange=e=>{cut=e.target.checked;draw()};
const slider=document.getElementById('slider');slider.max=D.frames.length-1;
function setFrame(i){frame=i;slider.value=i;document.getElementById('phase').textContent=`${D.times[i]}% of cycle`;curveMark();draw()}
slider.oninput=e=>{playing=false;document.getElementById('play').textContent='Play';setFrame(+e.target.value)};
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
function loop(t){if(playing&&t-last>1000/14){last=t;setFrame((frame+1)%D.frames.length)}requestAnimationFrame(loop)}
let drag=null;cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)};cv.onpointerup=()=>drag=null;
cv.onpointermove=e=>{if(!drag)return;ry+=(e.clientX-drag[0])*0.01;rx+=(e.clientY-drag[1])*0.01;drag=[e.clientX,e.clientY];draw()};
cv.onwheel=e=>{e.preventDefault();zoom=Math.min(6,Math.max(1.4,zoom*(1+e.deltaY*0.001)));draw()};
// volume curve
const svg=document.getElementById('curve');const T=D.times.length;let vmax=0;const vn=Object.keys(D.volumes);for(const k of vn)vmax=Math.max(vmax,...D.volumes[k]);vmax=Math.ceil(vmax/50)*50;
const X=i=>34+i*(280/(T-1)),Y=v=>160-v/vmax*145;let s='';
for(let g=0;g<=vmax;g+=50)s+=`<line x1="34" x2="314" y1="${Y(g)}" y2="${Y(g)}" stroke="currentColor" opacity=".12"/><text x="28" y="${Y(g)+4}" text-anchor="end">${g}</text>`;
for(const k of vn)s+=`<polyline fill="none" stroke="${D.colors[k]}" stroke-width="2" points="${D.volumes[k].map((v,i)=>X(i)+','+Y(v)).join(' ')}"/>`;
s+=`<line id="mk" y1="12" y2="160" stroke="currentColor" opacity=".5"/><text x="314" y="176" text-anchor="end">time in cycle</text>`;svg.innerHTML=s;
function curveMark(){const m=document.getElementById('mk');m.setAttribute('x1',X(frame));m.setAttribute('x2',X(frame))}
setFrame(0);requestAnimationFrame(loop);addEventListener('resize',draw);
</script></body></html>
"""

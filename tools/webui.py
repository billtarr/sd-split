#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Image on v73 NPU —— 最小 WebUI（PC 端运行，零第三方依赖）

两种后端，页面上可切：

  chain   : adb 跑设备上的 tools/chain.sh（**分段链**，三进程，采样 5.1 s/it，全权重常驻 NPU）
  server  : 转发到设备上常驻的 sd-server（三段链，末段持有采样器）

用法：
  python tools/webui.py                     # 默认 http://127.0.0.1:7860
  python tools/webui.py --port 8080
  python tools/webui.py --adb D:/android-tools/sdk/platform-tools/adb.exe --server http://192.168.1.8:18080

浏览器打开 http://127.0.0.1:7860 即可。不需要 adb forward（走局域网直连 sd-server）。
"""

import argparse
import base64
import json
import os
import posixpath
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ADB = r"D:/android-tools/sdk/platform-tools/adb.exe"
REMOTE_DIR = "/data/local/tmp/qisplit"
SERVER_URL = "http://192.168.1.8:18080"
LOCAL_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".push")

_lock = threading.Lock()   # 设备同一时刻只跑一个生成任务


# ----------------------------------------------------------------------------- adb helpers
def adb(*args, timeout=60):
    p = subprocess.run([ADB] + list(args), capture_output=True, timeout=timeout)
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out, err


def sh(cmd, timeout=60):
    return adb("shell", cmd, timeout=timeout)


def sd_proc_count():
    rc, out, _ = sh("ps -A | grep -c 'sd-cli-hex' || true")
    try:
        return int(out.strip().split()[-1])
    except Exception:
        return 0


def kill_stale():
    """清掉设备上残留的 sd-cli-hex（chain 的末段和 worker 都是它）。
    只杀 sd-cli-hex，**不动 sd-server-hex** —— 那是 server 模式的常驻服务，
    杀掉它 WebUI 的 server 模式就得重启。"""
    sh("for p in $(ps -A | grep 'sd-cli-hex' | awk '{print $2}'); do kill -TERM $p; done; true")
    for _ in range(20):
        rc, out, _ = sh("ps -A | grep -c 'sd-cli-hex' || true")
        try:
            if int(out.strip().split()[-1]) == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ----------------------------------------------------------------------------- chain (分段)
def gen_chain(prompt, steps, w, h, seed, timeout=900):
    tag = "w%d" % int(time.time())
    remote_png = "%s/out_%s.png" % (REMOTE_DIR, tag)

    if not kill_stale():
        return {"ok": False, "error": "设备上还有 sd-* 残留进程，拒绝启动（会撞 OOM）"}

    q = "'" + prompt.replace("'", "'\\''") + "'"
    # 注意：不要用 `adb shell "... &"` —— windows 上 adb 会一直等后台子进程的管道，
    # 直到 timeout 才返回（踩过）。改成前台跑 + Popen 不等待，自己轮询产物。
    cmd = (
        "cd %s && rm -f out_%s.png && "
        "PROMPT=%s STEPS=%d SEED=%d W=%d H=%d "
        "sh chain.sh %s > log_%s.txt 2>&1"
    ) % (REMOTE_DIR, tag, q, steps, seed, w, h, tag, tag)
    proc = subprocess.Popen([ADB, "shell", cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0 = time.time()
    launched = False
    while time.time() - t0 < timeout:
        time.sleep(3)
        rc, out, _ = sh("ps -A | grep -c 'sd-cli-hex' || true")
        try:
            n = int(out.strip().split()[-1])
        except Exception:
            n = 0
        if n > 0:
            launched = True
        elif launched:
            break
        if proc.poll() is not None and not launched and time.time() - t0 > 30:
            break
    elapsed = time.time() - t0
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()

    rc, out, _ = sh("test -f %s && echo yes" % remote_png)
    if "yes" not in out:
        _, tail, _ = sh("tail -6 %s/log_%s.txt" % (REMOTE_DIR, tag))
        return {"ok": False, "error": "没拿到出图", "elapsed": round(elapsed, 1), "log": tail.strip()}

    local_png = os.path.abspath(os.path.join(LOCAL_TMP, "webui_%s.png" % tag))
    os.makedirs(os.path.dirname(local_png), exist_ok=True)
    rc, out, err = adb("pull", remote_png, local_png, timeout=180)
    if not os.path.exists(local_png) or os.path.getsize(local_png) < 1024:
        return {"ok": False, "error": "pull 出图失败: %s" % err.strip(), "elapsed": round(elapsed, 1)}

    _, s_it, _ = sh("grep -oE '[0-9.]+s/it' %s/log_%s.txt | tail -1" % (REMOTE_DIR, tag))
    _, log_tail, _ = sh("grep -E 'completed in|CHAIN exit' %s/log_%s.txt | tail -2" % (REMOTE_DIR, tag))
    with open(local_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {
        "ok": True,
        "mode": "chain",
        "image": b64,
        "elapsed": round(elapsed, 1),
        "s_it": s_it.strip(),
        "log": log_tail.strip(),
    }


# ----------------------------------------------------------------------------- server (单进程)
def gen_server(prompt, steps, w, h, seed, url, timeout=900):
    body = json.dumps({
        "prompt": prompt, "steps": steps, "width": w, "height": h,
        "seed": seed, "cfg_scale": 1.0, "batch_size": 1, "n_iter": 1,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/sdapi/v1/txt2img",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        return {"ok": False, "error": "请求 sd-server 失败: %s" % e}
    elapsed = time.time() - t0
    imgs = data.get("images") or []
    if not imgs:
        return {"ok": False, "error": "sd-server 返回空图", "elapsed": round(elapsed, 1)}
    img = imgs[0]
    if img.startswith("data:"):
        img = img.split(",", 1)[1]
    return {"ok": True, "mode": "server", "image": img, "elapsed": round(elapsed, 1)}


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "qs-webui/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path.startswith("/api/health"):
            info = {"chain": {"ok": False}, "server": {"ok": False}}
            rc, out, _ = adb("devices", timeout=15)
            info["device"] = "7686f72b" in out and "device" in out
            try:
                req = urllib.request.Request(self.server.sd_url + "/sdapi/v1/options")
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=8) as r:
                    info["server"]["ok"] = r.status == 200
            except Exception as e:
                info["server"]["error"] = str(e)
            if info["device"]:
                rc, out, _ = sh("test -x %s/sd-cli-hex && echo yes" % REMOTE_DIR)
                info["chain"]["ok"] = "yes" in out
            return self._send(200, json.dumps(info))
        self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        if not self.path.startswith("/api/gen"):
            return self._send(404, json.dumps({"ok": False, "error": "not found"}))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"ok": False, "error": "bad json"}))

        prompt = (req.get("prompt") or "cat").strip() or "cat"
        steps = max(1, min(50, int(req.get("steps") or 8)))
        w = int(req.get("width") or 256)
        h = int(req.get("height") or 256)
        seed = int(req.get("seed") if req.get("seed") not in (None, "") else 42)
        mode = req.get("mode") or "chain"

        if not _lock.acquire(blocking=False):
            return self._send(429, json.dumps({"ok": False, "error": "正在生成，等这一张结束"}))
        try:
            if mode == "server":
                res = gen_server(prompt, steps, w, h, seed, self.server.sd_url)
            else:
                res = gen_chain(prompt, steps, w, h, seed)
        except Exception as e:
            res = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        finally:
            _lock.release()
        self._send(200, json.dumps(res))


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen-Image · v73 NPU</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a3;--acc:#4c8dff;--ok:#35c46a;--bad:#ff5c5c}
@media (prefers-color-scheme:light){:root{--bg:#f5f6f8;--card:#fff;--line:#e3e6ec;--fg:#1a1d23;--dim:#6b7280;--acc:#2563eb}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:17px;margin:0 0 2px;font-weight:600}
.sub{color:var(--dim);font-size:12px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:16px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
label{display:block;font-size:12px;color:var(--dim);margin:10px 0 5px}
label:first-child{margin-top:0}
textarea,input,select{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:8px 10px;font:inherit;outline:none}
textarea{min-height:74px;resize:vertical}
textarea:focus,input:focus,select:focus{border-color:var(--acc)}
.row{display:flex;gap:10px}.row>*{flex:1}
button{width:100%;margin-top:14px;padding:11px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-weight:600;font-size:14px;cursor:pointer}
button:disabled{opacity:.55;cursor:default}
.seg{display:flex;gap:6px;margin-top:6px}
.seg button{margin:0;padding:7px 10px;font-size:12px;background:var(--bg);color:var(--dim);border:1px solid var(--line);font-weight:500}
.seg button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
#out{min-height:300px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px}
#img{max-width:100%;border-radius:8px;display:none}
.ph{color:var(--dim);font-size:13px;text-align:center;padding:30px 10px}
.meta{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.meta b{color:var(--fg);font-weight:600}
.log{white-space:pre-wrap;font:11px/1.5 ui-monospace,Consolas,monospace;color:var(--dim);margin-top:8px;max-height:120px;overflow:auto}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px;background:var(--dim)}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}
.bar{height:3px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:12px;display:none}
.bar i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
</style></head><body><div class="wrap">
<h1>Qwen-Image 2.1 Turbo · v73 HTP</h1>
<div class="sub"><span id="h1" class="dot"></span><span id="hstat">检查后端…</span></div>
<div class="grid">
  <div class="card">
    <label>模式</label>
    <div class="seg" id="mode">
      <button data-v="chain" class="on">分段链（快）</button>
      <button data-v="server">单进程常驻</button>
    </div>
    <label>提示词</label>
    <textarea id="prompt">a red fox in snow, cinematic lighting</textarea>
    <div class="row"><div><label>步数<span style="opacity:.6">（由 sigmas preset 定死）</span></label>
      <input id="steps" type="number" value="8" disabled title="Turbo 的步数由 --sigmas preset 决定，--steps 会被覆盖"></div>
    <div><label>seed</label><input id="seed" type="number" value="42"></div></div>
    <div class="row"><div><label>宽</label><input id="w" type="number" value="256" step="64"></div>
    <div><label>高</label><input id="h" type="number" value="256" step="64"></div></div>
    <button id="go">生成</button>
    <div class="bar" id="bar"><i></i></div>
  </div>
  <div class="card">
    <div id="out"><div class="ph">生成的图会出现在这里</div><img id="img"></div>
    <div class="meta" id="meta"></div>
    <div class="log" id="log"></div>
  </div>
</div></div>
<script>
let mode='chain', busy=false, t0=0, timer=null;
const $=id=>document.getElementById(id);
document.querySelectorAll('#mode button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#mode button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); mode=b.dataset.v;
});
function status(t){$('hstat').textContent=t}
function tick(){
  const s=((Date.now()-t0)/1000);
  $('meta').innerHTML='<b>'+s.toFixed(1)+'</b> s　模式 <b>'+mode+'</b>';
}
fetch('/api/health').then(r=>r.json()).then(h=>{
  const d=$('h1'), parts=[];
  d.className='dot '+((h.device||h.server.ok)?'ok':'bad');
  parts.push(h.device?'设备在线':'设备离线');
  parts.push(h.chain.ok?'chain 就绪':'chain 缺 sd-cli-hex');
  parts.push(h.server.ok?'server 就绪':'server 未启动');
  status(parts.join(' · '));
}).catch(e=>{status('后端检查失败: '+e)});

$('go').onclick=async()=>{
  if(busy) return;
  busy=true; $('go').disabled=true; $('go').textContent='生成中…';
  $('img').style.display='none'; $('log').textContent='';
  document.querySelector('#out .ph')?.remove();
  $('bar').style.display='block'; $('bar').firstElementChild.style.width='6%';
  t0=Date.now(); timer=setInterval(tick,100);
  tick();
  const body={mode,prompt:$('prompt').value,steps:+$('steps').value,
              width:+$('w').value,height:+$('h').value,seed:+$('seed').value};
  try{
    const r=await fetch('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    $('bar').firstElementChild.style.width='100%';
    if(j.ok){
      $('img').src='data:image/png;base64,'+j.image; $('img').style.display='block';
      $('meta').innerHTML='<b>'+j.elapsed+'</b> s　模式 <b>'+j.mode+'</b>'+(j.s_it?'　'+j.s_it:'');
      $('log').textContent=j.log||'';
      status('完成');
    }else{
      $('meta').innerHTML='<span style="color:#ff5c5c">失败</span>';
      $('log').textContent=(j.error||'')+'\n'+(j.log||'');
      status('失败：'+(j.error||''));
    }
  }catch(e){
    $('meta').innerHTML='<span style="color:#ff5c5c">请求异常</span>';
    $('log').textContent=String(e);
  }finally{
    clearInterval(timer); busy=false; $('go').disabled=false; $('go').textContent='生成';
    setTimeout(()=>{$('bar').style.display='none';$('bar').firstElementChild.style.width='0'},900);
  }
};
</script></body></html>"""


def main():
    global ADB, REMOTE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--adb", default=ADB)
    ap.add_argument("--server", default=SERVER_URL)
    ap.add_argument("--remote-dir", default=REMOTE_DIR)
    a = ap.parse_args()

    ADB, REMOTE_DIR = a.adb, a.remote_dir
    if not os.path.exists(ADB):
        print("!! adb 不存在: %s（用 --adb 指定）" % ADB)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.sd_url = a.server.rstrip("/")
    srv.daemon_threads = True
    print("WebUI  ->  http://%s:%d" % (a.host, a.port))
    print("  chain  模式: adb 跑 %s/tools/chain.sh（分段链）" % REMOTE_DIR)
    print("  server 模式: 转发 %s" % srv.sd_url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()

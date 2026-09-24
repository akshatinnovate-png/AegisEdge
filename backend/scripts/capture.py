"""Drive a real node + console and capture what it looks like in action.

Every screenshot is of the real frontend talking to a real node over HTTP.
Nothing is mocked and no state is faked: the corpus is ingested through the
same path a deployed device uses, and the degraded frames are produced by
actually pinning the degradation ladder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import urllib.request

CORPUS = [
    ("sensor", "Bay 3 conveyor vibration crossed 4.2 mm/s at 02:14; the bearing signature matches the pre-failure cluster from March"),
    ("sensor", "Coolant pressure read 1.74 bar for 96 seconds before the interlock fired"),
    ("sensor", "Ambient temperature in the cell climbed to 61 C during the night shift"),
    ("sensor", "Spindle SP-9920 drew 41 A on start-up, twice the nominal inrush"),
    ("sensor", "Line 2 torque peaked at 47 nm during the tool change"),
    ("semantic", "Coolant pressure below 1.8 bar for over 90 seconds is treated as a hard stop condition on this cell"),
    ("semantic", "Bearing vibration above 4.0 mm/s is an early indicator of raceway spalling"),
    ("semantic", "Restricted-class memories never leave this device under any sync policy"),
    ("semantic", "The torque limit on line 2 is 42 nm after the bearing was replaced"),
    ("procedural", "Recovery: isolate the drive, purge the line, re-home the gantry, then release the interlock in that order"),
    ("procedural", "To clear a torque fault: cut servo power, rotate the spindle by hand, confirm free travel, then re-enable"),
    ("procedural", "Weekly: inspect the bay 3 conveyor belt tension and log the reading against the work order"),
    ("episodic", "Operator acknowledged the torque alarm and switched line 2 to manual feed for eleven minutes"),
    ("episodic", "Uplink dropped for 47 minutes during the night shift; operations queued locally and replayed on reconnect"),
    ("episodic", "Maintenance replaced the bay 3 bearing housing BX-7741-Q and logged the part number on the work order"),
    ("episodic", "The gantry was re-homed after the interlock released on line 2"),
]


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../testlogs/images")
    parser.add_argument("--data", default="/tmp/aegis-capture")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--web", type=int, default=5180)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    backend = Path(__file__).resolve().parents[1]
    frontend = backend.parent / "frontend"
    subprocess.run(["rm", "-rf", args.data], check=False)

    env = {**__import__("os").environ, "AEGIS_DATA_DIR": args.data, "PYTHONUNBUFFERED": "1"}
    node = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "aegis.main:app", "--host", "127.0.0.1",
         "--port", str(args.port), "--log-level", "warning"],
        cwd=backend, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web = subprocess.Popen([sys.executable, "-m", "http.server", str(args.web)],
                           cwd=frontend, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    api = f"http://127.0.0.1:{args.port}/api/v1"
    try:
        for _ in range(90):
            try:
                get(f"{api}/health")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("node did not come up")

        for collection, text in CORPUS:
            post(f"{api}/memory/ingest", {"text": text, "collection": collection})
        # a restricted memory, so the policy path is visible in the event stream
        post(f"{api}/memory/ingest",
             {"text": "Operator 4471 (jo.reyes@plant.io) overrode the interlock from console 2",
              "collection": "episodic"})
        # warm the caches and the event log with real queries
        for query in ("coolant pressure hard stop", "how do i recover the drive",
                      "bearing vibration", "what happened on line 2"):
            post(f"{api}/search", {"query": query, "k": 5})
        post(f"{api}/sync/trigger", {"reason": "capture"})
        print("node ready:", json.dumps(get(f"{api}/health"))[:120])

        script = out / "_shots.js"
        script.write_text(SHOTS % {"web": args.web, "api": api, "out": str(out),
                                   "base": f"http://127.0.0.1:{args.port}"}, encoding="utf-8")
        subprocess.run(["node", str(script)], check=True)
    finally:
        node.terminate()
        web.terminate()
        node.wait(timeout=20)
        web.wait(timeout=20)
    print("captured →", out)


SHOTS = r"""
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const API = '%(api)s';

async function post(page, path, body) {
  return page.evaluate(async ([api, path, body]) => {
    const r = await fetch(api + path, {method:'POST', headers:{'content-type':'application/json'},
                                       body: JSON.stringify(body)});
    return r.json();
  }, [API, path, body]);
}

(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5 });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));

  // The console defaults to :8000; this capture runs the node elsewhere so it
  // does not collide with a real one. Without this the page renders perfectly
  // and shows NO NODE with every figure blank — which is the frontend being
  // honest about having no backend, and a useless screenshot.
  await page.addInitScript((api) => { window.AEGIS_API = api; }, '%(base)s');

  await page.goto('http://localhost:%(web)d/index.html');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '%(out)s/01-cold-boot.png' });

  await page.waitForTimeout(9000);
  await page.screenshot({ path: '%(out)s/02-hero.png' });

  // Refuse to produce a gallery of a disconnected console.
  const link = await page.evaluate(() => (document.body.innerText.match(/NO NODE/) ? 'down' : 'up'));
  if (link === 'down') { console.log(JSON.stringify({ fatal: 'console never reached the node' })); await b.close(); process.exit(3); }

  await page.fill('#searchInput', 'colent presure hard stop');
  await page.click('.btn--go');
  await page.waitForTimeout(1200);
  await page.evaluate(() => document.getElementById('console').scrollIntoView());
  await page.waitForTimeout(600);
  await page.screenshot({ path: '%(out)s/03-console-live.png' });

  // pin the degradation ladder and show the console still answering
  await post(page, '/slo/override', { level: 'SURVIVAL' });
  await page.fill('#searchInput', 'bearing vibration raceway');
  await page.click('.btn--go');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '%(out)s/04-degraded-survival.png' });
  await post(page, '/slo/override', { level: null });

  // take the uplink down and show the node still serving
  await post(page, '/chaos/link_drop', { duration_s: 25 });
  await page.waitForTimeout(6000);
  await page.fill('#searchInput', 'what happened on line 2');
  await page.click('.btn--go');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '%(out)s/05-offline-still-serving.png' });

  await page.waitForTimeout(22000);   // let the link come back
  await page.evaluate(() => document.getElementById('console').scrollIntoView());
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '%(out)s/06-reconnected.png' });

  console.log(JSON.stringify({ errors }));
  await b.close();
})();
"""


if __name__ == "__main__":
    main()

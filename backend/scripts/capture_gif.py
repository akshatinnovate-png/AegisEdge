"""The durability claim, as motion.

    python3 scripts/capture_gif.py --out ../testlogs/images

A README can assert that a node survives SIGKILL. It cannot assert it
convincingly to somebody who will spend ninety seconds on the page and will
not clone anything. So this records the claim being tested: the memory count,
the kill, the gap, and the same count afterwards.

The node runs under `scripts/supervise.py`, which is what makes the kill
button honest — SIGKILL runs no handler, flushes no buffer and tidies nothing,
and something outside the process has to bring it back.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CORPUS = [
    "Conveyor 3 vibration spike on the night shift, raceway bearing suspected",
    "Coolant pressure dropped to 2.1 bar before the hard stop on line 2",
    "Gantry encoder drift of 0.4 mm after the tool change",
    "Hydraulic valve 7 slow to seat, replaced the seal",
    "Spindle torque limit raised to 42 Nm on the night shift",
    "Inverter fault F031 cleared after a power cycle",
]

FRAMES = r"""
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const API = '%(base)s';

(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 860, height: 1200 }, deviceScaleFactor: 2 });
  await page.addInitScript((api) => { window.AEGIS_API = api; }, API);
  await page.goto('http://localhost:%(web)d/index.html');
  await page.waitForTimeout(2500);

  await page.evaluate(() => document.querySelector('.mode-switch button[data-mode="prove"]')?.click());
  await page.waitForTimeout(1800);
  await page.evaluate(() => document.getElementById('durVerdict')?.scrollIntoView({block:'center'}));

  // Wait for the count to arrive rather than guessing. The first attempt used
  // a fixed delay and recorded a panel whose headline figure was still a dash,
  // which is a film of a system not answering.
  await page.waitForFunction(
    () => { const n = document.getElementById('durCount');
            return n && /[0-9]/.test(n.textContent || ''); },
    undefined, { timeout: 20000 });
  await page.waitForTimeout(600);

  const card = await page.$('#prove .proof');
  const shot = async (n) => { await card.screenshot({ path: `%(out)s/_gif/${String(n).padStart(2,'0')}.png` }); };

  // Before: the count, sitting there.
  for (let i = 0; i < 3; i++) { await shot(i); await page.waitForTimeout(500); }

  // The kill.
  await page.click('#durKill');
  for (let i = 3; i < 22; i++) { await shot(i); await page.waitForTimeout(700); }

  console.log(JSON.stringify({ ok: true }));
  await b.close();
})();
"""


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def assemble(frames_dir: Path, out: Path, width: int = 700,
             colors: int = 48, contrast: float = 1.6) -> None:
    """Frames to a GIF, held on the first and last so both read.

    A README GIF has a weight budget as real as any other asset: a reader on a
    phone will not wait for ten megabytes, and GitHub will still render it, so
    nothing tells you it was a bad idea. The dark gradient behind this panel is
    what costs — it dithers into thousands of near-identical colours — so the
    palette is small and dithering is off, which on flat UI chrome is
    invisible and on a gradient is the whole saving.
    """
    from PIL import Image, ImageEnhance

    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise RuntimeError("no frames captured")
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)),
                                 Image.LANCZOS)
        # Contrast is lifted before quantizing, not after. The console's
        # background is a dark gradient over a noise texture, which a small
        # palette spends itself on and leaves nothing for the text — at 32
        # colours the panel came out brown on brown and unreadable. Separating
        # the ink from the paper first means 48 colours is enough for both.
        image = ImageEnhance.Contrast(image).enhance(contrast)
        # Consecutive frames that are identical add bytes and say nothing; the
        # hold at either end is expressed as duration instead.
        if images and image.tobytes() == images[-1].tobytes():
            continue
        images.append(image)

    # A uniform size is required or PIL silently crops to the first frame, and
    # the panel changes height as the verdict text grows.
    tallest = max(i.height for i in images)
    canvas = []
    for image in images:
        if image.height != tallest:
            padded = Image.new("RGB", (image.width, tallest), (11, 11, 13))
            padded.paste(image, (0, 0))
            image = padded
        canvas.append(image.convert("P", palette=Image.ADAPTIVE, colors=colors,
                                    dither=Image.Dither.NONE))

    durations = [700] * len(canvas)
    durations[0] = 1400                       # hold on "before"
    durations[-1] = 2600                      # and on "recovered"
    canvas[0].save(out, save_all=True, append_images=canvas[1:], loop=0,
                   duration=durations, optimize=True, disposal=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../testlogs/images")
    parser.add_argument("--data", default="/tmp/aegis-gif")
    parser.add_argument("--port", type=int, default=8140)
    parser.add_argument("--web", type=int, default=5190)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    (out / "_gif").mkdir(parents=True, exist_ok=True)
    for stale in (out / "_gif").glob("*.png"):
        stale.unlink()
    backend = Path(__file__).resolve().parents[1]
    frontend = backend.parent / "frontend"
    subprocess.run(["rm", "-rf", args.data], check=False)

    supervisor = subprocess.Popen(
        [sys.executable, "scripts/supervise.py", "--port", str(args.port),
         "--data-dir", args.data, "--host", "127.0.0.1"],
        cwd=backend, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web = subprocess.Popen([sys.executable, "-m", "http.server", str(args.web)],
                           cwd=frontend, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    api = f"http://127.0.0.1:{args.port}/api/v1"
    try:
        for _ in range(120):
            try:
                get(f"{api}/health")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("node did not come up under the supervisor")

        for text in CORPUS:
            post(f"{api}/memory/ingest", {"text": text, "collection": "episodic"})
        print("node ready with", get(f"{api}/health")["points"], "memories")

        script = out / "_frames.js"
        script.write_text(FRAMES % {"web": args.web, "out": str(out),
                                    "base": f"http://127.0.0.1:{args.port}"},
                          encoding="utf-8")
        subprocess.run(["node", str(script)], check=True)

        target = out / "durability.gif"
        assemble(out / "_gif", target)
        print(f"wrote {target}  ({target.stat().st_size / 1e6:.2f} MB)")
    finally:
        supervisor.terminate()
        web.terminate()
        try:
            supervisor.wait(timeout=20)
            web.wait(timeout=20)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

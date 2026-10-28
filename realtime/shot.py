"""Captura o canvas do index.html no Chrome headless (GPU real) para conferência.

Uso: python realtime/shot.py saida.png "grid=3&th=70" [largura altura]
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

page = Path(__file__).resolve().parent / "index.html"
out, query = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
w, h = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("1280", "720")
url = page.as_uri() + "?hide&capture=3&" + query
html = subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--use-gl=angle", "--use-angle=gl",
                       f"--window-size={w},{h}", "--virtual-time-budget=60000", "--dump-dom", url],
                      capture_output=True, text=True, timeout=240).stdout
m = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html)
if not m:
    sys.exit("sem captura")
Path(out).write_bytes(base64.b64decode(m.group(1)))
print(out)

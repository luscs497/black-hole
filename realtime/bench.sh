#!/bin/sh
# Mede o custo por quadro (1920x1080) na GPU via Chrome headless.
cd "$(dirname "$0")"
google-chrome --headless=new --no-sandbox --use-gl=angle --use-angle=gl \
  --dump-dom "file://$PWD/index.html?bench" 2>/dev/null |
  grep -o '<pre id="validation">.*</pre>' | sed 's/<[^>]*>//g;s/&quot;/"/g' |
  python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d["renderer"])
for r in d["results"]:
    print(f"{r.get("caso"):<22} {r.get("res"):>10} {r.get("ms"):7.2f} ms")
'

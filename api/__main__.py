import logging
import os
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
port = int(os.environ.get("PORT", "8000"))
url = f"http://127.0.0.1:{port}"
print(f"\n  DetectAI is starting: open {url}  (the first start after a reboot can take a minute)\n")


def open_when_ready() -> None:
    """Open the browser only once the server answers, so it never shows a connection error."""
    for _ in range(300):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=2)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(1)


if "--open" in sys.argv:
    threading.Thread(target=open_when_ready, daemon=True).start()
uvicorn.run("api.main:app", host="127.0.0.1", port=port)

import logging
import os
import sys
import threading
import webbrowser

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
port = int(os.environ.get("PORT", "8000"))
url = f"http://127.0.0.1:{port}"
print(f"\n  DetectAI is starting: open {url}\n")  # ASCII only: Windows consoles may not be UTF-8
if "--open" in sys.argv:
    threading.Timer(2.0, webbrowser.open, args=(url,)).start()
uvicorn.run("api.main:app", host="127.0.0.1", port=port)

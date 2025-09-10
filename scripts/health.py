#!/usr/bin/env python3
import http.server, socketserver, json, os

ART = os.getenv("ARTIFACTS_DIR","artifacts")
PORT=int(os.getenv("PORT","8080"))

class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        hb_crypto = os.path.exists(os.path.join(ART,"crypto_heartbeat.json"))
        hb_equity = os.path.exists(os.path.join(ART,"equity_heartbeat.json"))
        out = {"ok": hb_crypto or hb_equity, "crypto": hb_crypto, "equity": hb_equity}
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

socketserver.TCPServer(("",PORT), H).serve_forever()

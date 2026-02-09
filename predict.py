import traceback
from io import BytesIO
from ultralytics import YOLO
from model_loader import load_best
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import json

HOST = "localhost"
PORT = 3445

model: YOLO | None = None

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global model
        try:
            img_data = self.rfile.read(int(self.headers["Content-Length"]))
            img = Image.open(BytesIO(img_data))

            results = model.predict(img)
            data = f"[{",".join([r.to_json(normalize=True) for r in results])}]".encode()

            self.send_response(200)
            self.send_header("Content-Type", "text/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            traceback.print_exc()
            data = str(e).encode()

            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

def main():
    global model
    server = HTTPServer((HOST, PORT), Handler)
    try:
        model = YOLO(load_best())
        print(f"Server started http://{HOST}:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
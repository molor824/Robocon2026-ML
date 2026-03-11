import socket
import struct
import multiprocessing as mp
import queue
import cv2 as cv
import numpy as np

from ultralytics import YOLO

HOST = "molor-pi"
PORT = 9000
RECV_PORT = 9001
MAX_PAYLOAD = 65000
HEADER_STRUCT = struct.Struct("!QII")
MAX_PACKET_SIZE = MAX_PAYLOAD + HEADER_STRUCT.size
MAX_TIMEOUT = 1.0

def handle_model_process(frame_queue: mp.Queue, result_queue: mp.Queue):
    model = YOLO("runs/detect/train/weights/best.pt")
    try:
        while True:
            frame = frame_queue.get()
            print("Got frame")
            results = model.predict(frame, stream=True)
            result_queue.put(list(results))
    except KeyboardInterrupt: pass

def main():
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    conn.connect((HOST, PORT))
    conn.send(struct.pack("!H", RECV_PORT))

    udp_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)  # 4MB buffer allocated by OS
    udp_conn.bind(("0.0.0.0", RECV_PORT))
    udp_conn.settimeout(MAX_TIMEOUT)
    print(f"Listening on 0.0.0.0:{RECV_PORT}")

    frame_buffers: dict[int, dict[int, bytes]] = {}

    frame = None
    results = None

    frame_queue = mp.Queue(1)
    results_queue = mp.Queue(1)

    computing = False

    model_process = mp.Process(target=handle_model_process, args=(frame_queue, results_queue,))
    model_process.start()

    try:
        while True:
            try:
                packet, _ = udp_conn.recvfrom(MAX_PACKET_SIZE)
                frame_id, chunk_id, total_chunks = HEADER_STRUCT.unpack(packet[:HEADER_STRUCT.size])
                chunk_data = packet[HEADER_STRUCT.size:]

                if frame_id not in frame_buffers:
                    frame_buffers[frame_id] = {}

                chunks = frame_buffers[frame_id]
                chunks[chunk_id] = chunk_data
                if len(chunks) >= total_chunks:
                    raw = b''.join(chunks[i] for i in range(total_chunks))
                    np_arr = np.frombuffer(raw, dtype=np.uint8)
                    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR_BGR)
                    frame_buffers = dict((k, v) for k, v in frame_buffers.items() if k > frame_id)
            except socket.timeout: pass

            if frame is None: continue

            try:
                if computing:
                    results = results_queue.get_nowait()
                    computing = False
                if not computing:
                    frame_queue.put_nowait(frame)
                    computing = True
            except queue.Empty: pass
            except queue.Full: pass

            output = frame.copy()
            if results is not None:
                for r in results:
                    names = r.names
                    for cls, conf, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(
                            output,
                            f"{names[int(cls.item())]}: {conf.item()*100:.0f}%",
                            (x1, y1 - 20),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0)
                        )

            cv.imshow("Output", output)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt: pass
    finally:
        conn.close()
        udp_conn.close()
        cv.destroyAllWindows()
        frame_queue.close()
        results_queue.close()
        model_process.kill()
        model_process.join()

if __name__ == "__main__":
    main()
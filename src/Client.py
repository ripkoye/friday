import pyaudio, numpy as np, time, os, json, socket, struct, threading, sys

# ---------- Mic capture settings ----------
IN_SAMPLE_RATE = 16000
IN_FRAMES_PER_CHUNK = 1600   # 100 ms @ 16kHz
IN_FORMAT = pyaudio.paInt16
IN_CHANNELS = 1
INPUT_DEVICE_INDEX = None     # set to an int if you want a specific mic

# ---------- Server endpoints ----------
SERVER_HOST = "99.172.2.182"
UPSTREAM_PORT = 8080   # mic -> server (your existing ASR port)
DOWNSTREAM_PORT = 8081 # server -> client (TTS audio port; make your server send here)

# ---------- Helpers for framing ----------
def pack_frame(header_obj: dict, payload: bytes) -> bytes:
    hb = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload

def read_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data

def recv_frame(sock: socket.socket):
    # [uint32_be header_len][header_json][payload]
    raw = read_exact(sock, 4)
    (hlen,) = struct.unpack(">I", raw)
    hbytes = read_exact(sock, hlen)
    header = json.loads(hbytes.decode("utf-8"))
    # optional EOS frame without payload
    if header.get("type") == "eos":
        return header, b""
    length = int(header.get("length", 0))
    payload = read_exact(sock, length) if length > 0 else b""
    return header, payload

# ---------- Audio I/O ----------
def setup_microphone(pa: pyaudio.PyAudio):
    if INPUT_DEVICE_INDEX is None:
        # Optional: print input devices
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                print(i, "-", info["name"])
    stream = pa.open(
        format=IN_FORMAT,
        channels=IN_CHANNELS,
        rate=IN_SAMPLE_RATE,
        input=True,
        frames_per_buffer=IN_FRAMES_PER_CHUNK,
        input_device_index=INPUT_DEVICE_INDEX,
        start=False,  # we start explicitly
    )
    return stream

def mic_sender(pa: pyaudio.PyAudio, sock: socket.socket, stop_evt: threading.Event):
    stream = setup_microphone(pa)
    stream.start_stream()
    print("[Mic] streaming mic → server")
    try:
        while not stop_evt.is_set():
            audio_bytes = stream.read(IN_FRAMES_PER_CHUNK, exception_on_overflow=False)
            header_obj = {
                "type": "audio_chunk",
                "length": len(audio_bytes),
                "sample_rate": IN_SAMPLE_RATE,
                "channels": IN_CHANNELS,
                "sample_width": 2,  # 16-bit
                "timestamp": time.time(),
                # optionally add auth/HMAC fields here
            }
            frame = pack_frame(header_obj, audio_bytes)
            sock.sendall(frame)
            # print(".", end="", flush=True)  # uncomment for heartbeat
    except Exception as e:
        print(f"[Mic] error: {e}")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        print("[Mic] stopped")

def player_receiver(pa: pyaudio.PyAudio, sock: socket.socket, stop_evt: threading.Event):
    playback = None
    print("[Play] waiting for TTS audio from server")
    try:
        while not stop_evt.is_set():
            header, payload = recv_frame(sock)
            if header.get("type") == "eos":
                print("[Play] received EOS")
                break

            sr = int(header.get("sample_rate", 24000))
            ch = int(header.get("channels", 1))
            sw = int(header.get("sample_width", 2))  # bytes per sample
            if playback is None:
                fmt = pa.get_format_from_width(sw)
                playback = pa.open(format=fmt, channels=ch, rate=sr, output=True, frames_per_buffer=1024)
                print(f"[Play] opened output: {sr} Hz, {ch} ch, {sw*8}-bit")

            if payload:
                playback.write(payload)
    except Exception as e:
        print(f"[Play] error: {e}")
    finally:
        try:
            if playback is not None:
                playback.stop_stream()
                playback.close()
        except Exception:
            pass
        print("[Play] stopped")

def main():
    pa = pyaudio.PyAudio()
    stop_evt = threading.Event()

    # Upstream socket (mic -> server)
    up_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    up_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    up_sock.connect((SERVER_HOST, UPSTREAM_PORT))
    print(f"[Up] connected to {SERVER_HOST}:{UPSTREAM_PORT}")

    # Downstream socket (server -> client TTS)
    down_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    down_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    down_sock.connect((SERVER_HOST, DOWNSTREAM_PORT))
    print(f"[Down] connected to {SERVER_HOST}:{DOWNSTREAM_PORT}")

    t1 = threading.Thread(target=mic_sender, args=(pa, up_sock, stop_evt), daemon=True)
    t2 = threading.Thread(target=player_receiver, args=(pa, down_sock, stop_evt), daemon=True)
    t1.start(); t2.start()

    print("Listening & playing…  Ctrl+C to quit.")
    try:
        while t1.is_alive() and t2.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[Main] stopping…")
    finally:
        stop_evt.set()
        try:
            up_sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            down_sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        up_sock.close()
        down_sock.close()
        pa.terminate()
        print("[Main] done.")

if __name__ == "__main__":
    main()

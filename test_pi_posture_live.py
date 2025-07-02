# test_pi_posture_live.py
import cv2
import base64
import requests
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access Pi Camera.")
    exit()

print("📡 Real-time posture detection started...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed.")
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        payload = {"image": f"data:image/jpeg;base64,{image_base64}"}

        try:
            res = requests.post("http://127.0.0.1:5002/detect-posture", json=payload, timeout=10)
            result = res.json()
            print(f"📸 Posture: {result['posture']} | Stable: {result['stable_posture']} | Freeze: {result['freeze_count']} | Visibility: {result['visibility_count']}")
        except Exception as e:
            print("❌ Flask API error:", e)

        time.sleep(3)
except KeyboardInterrupt:
    print("\n🛑 Stopped.")
finally:
    cap.release()

from ultralytics import YOLO
import cv2
import requests

# ====== CONFIG ======
BOT_TOKEN = "8451324583:AAH2lXw3fGTaPUinlHXc7T6FW28g2fBT9xo"
CHAT_ID = "7219023636"
CONF_THRESHOLD = 0.1
TARGET_CLASSES = ["0","1"]  # Coin Object Classes
# ====================

model = YOLO("/Users/tunt/Downloads/leaf_data/best.pt")

# โหลดภาพ
img_path = "/Users/tunt/Downloads/leaf_data/leaf_detection/valid/images/frame_000063_png.rf.fb0c461b31f141d5985b1dc0594b156c.jpg"
results = model(img_path)[0]

detected = False
img = cv2.imread(img_path)

for box in results.boxes:
    conf = float(box.conf)
    cls = int(box.cls)
    label = model.names[cls]

    if conf > CONF_THRESHOLD and label in TARGET_CLASSES:
        detected = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

if detected:
    cv2.imwrite("detected.jpg", img)

    url = f"https://api.telegram.org/bot8451324583:AAH2lXw3fGTaPUinlHXc7T6FW28g2fBT9xo/sendPhoto"
    files = {"photo": open("detected.jpg", "rb")}
    data = {
        "chat_id": CHAT_ID,
        "caption": "🚨 YOLO ตรวจพบวัตถุ"
    }

    requests.post(url, data=data, files=files)
    print("📨 Sent to Telegram")
else:
    print("❌ No target detected")
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# --------- Config ----------
CANVAS = 280          # drawing surface (square)
BRUSH = 14            # stroke radius (thicker = easier to see)
OUT = 20              # downsample to 20x20
PREVIEW = 220         # magnified preview size
LIVE_PRED = True      # show live prediction in window title

# >>> update this path to your weights if different <<<
MODEL_WEIGHTS = "./crossvalidate_9.pyt"

# --------- Your CNN (MATCHES TRAINING) ----------
class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # this name MUST match what you used in training: meta_layer1
        self.meta_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(3, 1, 1), nn.Dropout(0.05),

            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.10),

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.15),

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.20),
        )

        # same head as in training
        self.fc_layer1 = nn.Linear(512, 64)
        self.dropout_fc = nn.Dropout(0.1)
        self.fc_layer2 = nn.Linear(64, 10)

    def forward(self, xss):
        # xss is [batch, 400]; reshape to [batch, 1, 20, 20]
        xss = xss.view(-1, 1, 20, 20)
        xss = self.meta_layer1(xss)
        xss = xss.view(-1, 512)
        xss = torch.relu(self.fc_layer1(xss))
        xss = self.dropout_fc(xss)
        xss = self.fc_layer2(xss)   # logits
        return xss


def load_weights(model, path):
    state = torch.load(path, map_location="cpu")
    # handle case where you saved with something like {'state_dict': ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

# --------- Init canvas & model ----------
img = np.full((CANVAS, CANVAS), 255, np.uint8)   # white background
drawing = False
last = None

model = None
if Path(MODEL_WEIGHTS).exists():
    try:
        m = ConvolutionalModel()
        load_weights(m, MODEL_WEIGHTS)
        m.eval()
        model = m
        print("[OK] Model loaded.")
    except Exception as e:
        print("[Info] Model not loaded:", e)
else:
    print("[Info] No weights file found at:", MODEL_WEIGHTS)

# --------- Helpers ----------
def downsample_20x20(gray):
    # Canvas: white background (255), black digit (0)
    # Training: black background (0), white digit (255)
    # → invert here so the model sees what it was trained on
    small = cv2.resize(gray, (OUT, OUT), interpolation=cv2.INTER_AREA)
    small = 255 - small          # NOW: black background, white digit
    return small

def tensor_from_20x20(small):
    arr = small.astype(np.float32) / 255.0   # 0–1, white digit ~1
    flat = arr.reshape(1, -1)                # [1, 400]
    return torch.from_numpy(flat)

def update_preview():
    small = downsample_20x20(img)
    big = cv2.resize(small, (PREVIEW, PREVIEW), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("20x20 Preview", big)         # shows black bg, white digit

def predict_from_canvas():
    if model is None:
        return None
    small = downsample_20x20(img)
    tens = tensor_from_20x20(small)
    with torch.no_grad():
        logits = model(tens)
        pred = int(logits.argmax(1).item())
    return pred

def safe_set_title(win, text):
    # some OpenCV builds on macOS don't have setWindowTitle
    if hasattr(cv2, "setWindowTitle"):
        cv2.setWindowTitle(win, text)

def put_title_with_pred():
    base = "Draw (Left drag) | Keys: C=Clear, P=Predict, Q/Esc=Quit"
    if model is None or not LIVE_PRED:
        safe_set_title("Draw", base)
        return
    pred = predict_from_canvas()
    safe_set_title("Draw", f"{base} | Pred: {pred}")

# --------- Mouse callback ----------
def on_mouse(event, x, y, flags, param):
    global drawing, last, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last = (x, y)
        cv2.circle(img, (x, y), BRUSH, 0, -1)
        update_preview(); put_title_with_pred()
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(img, last, (x, y), 0, BRUSH*2)
        cv2.circle(img, (x, y), BRUSH, 0, -1)
        last = (x, y)
        update_preview(); put_title_with_pred()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last = None
        update_preview(); put_title_with_pred()

# --------- Windows ----------
cv2.namedWindow("Draw", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Draw", on_mouse)
cv2.namedWindow("20x20 Preview", cv2.WINDOW_AUTOSIZE)

# Initial preview
update_preview(); put_title_with_pred()

print("Instructions:")
print(" - Left-click and drag to draw")
print(" - Press 'c' to clear")
print(" - Press 'p' to predict once (prints to console, updates title)")
print(" - Press 'q' or 'Esc' to quit")

# --------- Main loop ----------
while True:
    cv2.imshow("Draw", img)
    key = cv2.waitKey(10) & 0xFF
    if key in (27, ord('q')):      # ESC or q
        break
    elif key == ord('c'):
        img[:] = 255
        update_preview(); put_title_with_pred()
    elif key == ord('p'):
        if model is None:
            print("(No model loaded; set MODEL_WEIGHTS correctly.)")
        else:
            p = predict_from_canvas()
            print("Predicted:", p)
            put_title_with_pred()

cv2.destroyAllWindows()


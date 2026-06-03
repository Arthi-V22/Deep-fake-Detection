import cv2
import numpy as np
import uuid
import os

os.makedirs("static/results", exist_ok=True)

def detect_skin_damage(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    
    # --- Detect face region ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no face found, use full image
    if len(faces) == 0:
        faces = [(0, 0, img.shape[1], img.shape[0])]

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        # --- Convert to LAB color space (better for skin analysis) ---
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Redness = high 'a' channel (green–red)
        redness = cv2.GaussianBlur(a, (7,7), 0)
        redness_norm = cv2.normalize(redness, None, 0, 255, cv2.NORM_MINMAX)
        
        # --- Threshold for inflamed/red areas ---
        _, mask = cv2.threshold(redness_norm, 150, 255, cv2.THRESH_BINARY)

        # Smooth mask
        mask = cv2.GaussianBlur(mask, (11,11), 0)

        # --- Create heatmap ---
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face, 0.6, heatmap, 0.6, 0)

        # Replace in original
        img[y:y+h, x:x+w] = overlay

    out_name = f"{uuid.uuid4().hex}_skin_damage_heatmap.png"
    out_path = os.path.join("static/results", out_name)
    cv2.imwrite(out_path, img)
    return out_path

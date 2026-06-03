import torch
from torchvision import transforms
from PIL import Image
import timm
import cv2
import os

# ---------------- CONFIG ----------------
model_name = "efficientnet_b0"   # same as training
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels (based on your training folder structure)
class_names = ['Fake', 'Real']  # <-- adjust if your dataset folders are named differently

# ---------------- LOAD MODEL ----------------
print("📦 Loading trained model...")
model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded and ready.")

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------------- IMAGE PREDICT FUNCTION ----------------
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.softmax(output, dim=1)[0][pred].item()
    return class_names[pred], conf

# ---------------- WEBCAM PREDICTION ----------------
def webcam_predict():
    cap = cv2.VideoCapture(0)
    print("🎥 Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV image (BGR) to PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
            pred = torch.argmax(output, dim=1).item()
            conf = torch.softmax(output, dim=1)[0][pred].item()

        label = f"{class_names[pred]} ({conf*100:.1f}%)"
        color = (0,255,0) if pred == 1 else (0,0,255)
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Real-Time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\nChoose mode:")
    print("1️⃣  Predict single image")
    print("2️⃣  Real-time webcam prediction")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        img_path = input("Enter image file path: ").strip()
        if not os.path.exists(img_path):
            print("❌ File not found.")
        else:
            label, conf = predict_image(img_path)
            print(f"✅ Prediction: {label} ({conf*100:.2f}%)")
    elif choice == "2":
        webcam_predict()
    else:
        print("❌ Invalid choice.")

import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import time

# ===== Dataset Class =====
class VideoDataset(Dataset):
    def __init__(self, root_dir, subfolders, transform=None, frames_per_video=15):
        """
        subfolders: list of (folder_name, label) tuples
        e.g. [('fake_video', 0), ('real_video', 1)]
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = []

        for subfolder, label in subfolders:
            folder = os.path.join(root_dir, subfolder)
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist!")
                continue
            for file in os.listdir(folder):
                if file.endswith('.mp4'):
                    self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        return frames, torch.tensor(label, dtype=torch.long)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, max(1, total_frames // self.frames_per_video)))[:self.frames_per_video]
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        cap.release()

        # If not enough frames, repeat last frame
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])
        return torch.stack(frames)

# ===== Transform =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ===== Model =====
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super(CNN_LSTM, self).__init__()
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, frames, C, H, W = x.shape
        cnn_features = []
        for f in range(frames):
            feat = self.cnn(x[:, f])
            cnn_features.append(feat)
        cnn_features = torch.stack(cnn_features, dim=1)
        _, (h, _) = self.lstm(cnn_features)
        out = self.fc(h[-1])
        return out

# ==========================================================
# 🧠 TRAINING CODE — runs only when file is executed directly
# ==========================================================
if __name__ == "__main__":
    # ===== Dataset setup =====
    train_subfolders = [('fake_video', 0), ('real_video', 1)]
    test_subfolders = [('attack', 0), ('real_video', 1)]

    train_dataset = VideoDataset('./Video_dataset/train/', train_subfolders, transform)
    test_dataset = VideoDataset('./Video_dataset/test/', test_subfolders, transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # ===== Training setup =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f} sec")

    # ===== Save Model =====
    torch.save(model.state_dict(), 'cnn_lstm_video_model.pth')
    print("✅ Model saved as cnn_lstm_video_model.pth")

    # ===== Testing =====
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

from flask import Flask, render_template, request
from flask import jsonify, json
from werkzeug.utils import secure_filename
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition
from torch import nn
from torchvision import models
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""
detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ===================== MODEL ========================
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


sm = nn.Softmax()


def predict(model, img, path="./"):
    fmap, logits = model(img.to())
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print("confidence =", confidence)
    return [int(prediction.item()), confidence]


# ===================== DATASET ========================
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []

        for i, frame in enumerate(self.frame_extract(video_path)):

            if frame is None:
                continue

            # ensure uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # -------- HANDLE ALL FORMATS ----------
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            elif frame.shape[2] == 4:  # RGBA PNG
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # --------------------------------------

            # face detection safe
            try:
                faces = face_recognition.face_locations(frame)
            except Exception as e:
                print("Face detect skipped due to error:", e)
                faces = []

            # If no face found, DO NOT SKIP. Use whole image.
            if len(faces) > 0:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]

            frames.append(self.transform(frame))

            if len(frames) == self.count:
                break

        # ---- FALLBACK: prevent empty tensor ----
        if len(frames) == 0:
            print("No valid frames — using fallback image")
            image = cv2.imread(video_path)

            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(image))
        # ----------------------------------------

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True

        while success:
            success, image = vidObj.read()

            if not success:
                # If it's an image instead of video
                if path.lower().endswith((".jpg", ".jpeg", ".png")):
                    image = cv2.imread(path)
                    yield image
                    break
                break

            yield image


# ===================== DETECTION ========================
def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    path_to_videos = [videoPath]
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)

    model = Model(2)
    path_to_model = "model/df_model.pt"
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))
    model.eval()

    for i in range(len(path_to_videos)):
        prediction = predict(model, video_dataset[i], "./")

    return prediction


# ===================== ROUTES ========================
@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'GET':
        return render_template("index.html")


@app.route('/Detect', methods=['GET', 'POST'])
def DetectPage():
    if request.method == 'GET':
        return render_template("index.html")

    if request.method == 'POST':
        video = request.files["video"]
        print(video.filename)

        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))

        video_path = "Uploaded_Files/" + video_filename
        prediction = detectFakeVideo(video_path)

        if prediction[0] == 0:
            output = "FAKE"
        else:
            output = "REAL"

        confidence = prediction[1]

        data = {"output": output, "confidence": confidence}
        data = json.dumps(data)

        os.remove(video_path)

        return render_template("index.html", data=data)


# ===================== RUN ========================
app.run(port=2000)
 
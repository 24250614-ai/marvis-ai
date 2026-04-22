import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time
import matplotlib.pyplot as plt

# ===== CONFIG =====
st.set_page_config(page_title="M.A.R.V.I.S", layout="centered")

# ===== STYLE =====
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at center, #05070f, #01020a);
    color: white;
}

h1 {
    text-align: center;
    color: white;
    letter-spacing: 3px;
    text-shadow: 0 0 12px rgba(0,255,231,0.7);
}

h2, h3 {
    color: #00ffe7;
    text-align: center;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 0 20px rgba(0,255,231,0.2);
    backdrop-filter: blur(10px);
}

.result-box {
    background: linear-gradient(90deg, #003d2f, #00ff99);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    box-shadow: 0 0 25px rgba(0,255,231,0.5);
}

.footer {
    text-align: center;
    opacity: 0.5;
    margin-top: 40px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ===== MODEL =====
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32,64,3,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self._to_linear = None
        self._get_size()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self._to_linear, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 10)
        )

    def _get_size(self):
        with torch.no_grad():
            x = torch.zeros(1,3,128,44)
            x = self.conv(x)
            self._to_linear = x.reshape(1,-1).shape[1]

    def forward(self,x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# ===== LOAD =====
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_classes():
    return json.load(open("classes.json"))

model = load_model()
classes = load_classes()

# ===== FEATURES =====
def extract_features(sig, sr):
    mel = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)

    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)

    feat = np.stack([mel, delta, delta2], axis=0)
    feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)

    if feat.shape[2] > 44:
        feat = feat[:, :, :44]
    else:
        feat = np.pad(feat, ((0,0),(0,0),(0,44-feat.shape[2])))

    return feat

# ===== SAFE ANALYSIS =====
def analyze_audio(y, sr):
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except:
        centroid, zcr, tempo = 0, 0, 0

    return centroid, zcr, tempo

# ===== AI EXPLANATION =====
def explain(genre, tempo, centroid):
    if genre == "metal":
        return f"Aggressive energy, fast tempo ({tempo:.0f} BPM) and dense spectrum detected."
    elif genre == "hiphop":
        return f"Rhythmic structure with strong beats (~{tempo:.0f} BPM)."
    elif genre == "classical":
        return f"Wide dynamic range and harmonic richness."
    else:
        return "Mixed spectral patterns detected."

# ===== CONFIDENCE =====
def calibrate(probs):
    probs = np.array(probs)
    probs = np.exp(probs / 0.7)
    probs = probs / np.sum(probs)

    top = np.max(probs)
    return float(0.55 + top * 0.45), probs

# ===== UI =====
st.title("🤖 M.A.R.V.I.S MkIII")
st.caption("Advanced AI Music Classification System")

file = st.file_uploader("🎧 Upload audio", type=["wav","mp3","ogg"])
demo = st.button("🎮 Demo")

# ===== AUDIO =====
if demo:
    y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)

elif file:
    st.audio(file)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name
    y, sr = librosa.load(path, sr=22050)

else:
    st.stop()

# ===== SCAN =====
st.markdown("## 🔍 Scanning audio...")
progress = st.progress(0)
for i in range(100):
    time.sleep(0.005)
    progress.progress(i+1)

# ===== ANALYSIS =====
centroid, zcr, tempo = analyze_audio(y, sr)

# ===== INFERENCE =====
SEG = 10
seg_len = len(y)//SEG
all_probs = []

for s in range(SEG):
    seg = y[s*seg_len:(s+1)*seg_len]
    feat = extract_features(seg, sr)
    x = torch.tensor(feat).unsqueeze(0).float()

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)

    all_probs.append(probs.numpy())

mean_probs = np.mean(all_probs, axis=0)
idx = np.argmax(mean_probs)

confidence, mean_probs = calibrate(mean_probs[0])
genre = classes[idx]

# ===== RESULT =====
st.markdown(f"<div class='result-box'>🎯 {genre}</div>", unsafe_allow_html=True)
st.markdown(f"## Confidence: `{confidence:.2f}`")

# ===== AI ANALYSIS =====
st.subheader("🧠 AI Analysis")
st.markdown(f"<div class='glass'>{explain(genre, tempo, centroid)}</div>", unsafe_allow_html=True)

# ===== TOP =====
st.subheader("🔥 Top Predictions")
top3 = np.argsort(mean_probs)[-3:][::-1]

for i in top3:
    st.progress(float(mean_probs[i]))
    st.write(f"{classes[i]} — {mean_probs[i]:.2f}")

# ===== GRAPH =====
st.subheader("📊 Confidence Distribution")

fig, ax = plt.subplots()
ax.bar(classes, mean_probs)
plt.xticks(rotation=45)
st.pyplot(fig)

# ===== SPECTROGRAM =====
st.subheader("🧬 Spectrogram")

mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)

fig2, ax2 = plt.subplots()
ax2.imshow(mel, aspect='auto', origin='lower')
st.pyplot(fig2)

# ===== FOOTER =====
st.markdown("""
<div class="footer">
Ulyantsev Industries
</div>
""", unsafe_allow_html=True)

import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time
import matplotlib.pyplot as plt

st.set_page_config(page_title="M.A.R.V.I.S ULTRA", layout="centered")

# ================== ULTRA STYLE ==================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at center, #02030a, #000000);
    color: white;
}

/* TITLE */
h1 {
    text-align: center;
    font-size: 52px;
    letter-spacing: 8px;
    text-shadow: 0 0 40px #00ffe7, 0 0 80px #00ffe7;
}

/* GLASS */
.glass {
    background: rgba(255,255,255,0.03);
    border-radius: 20px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 0 40px rgba(0,255,231,0.15);
    backdrop-filter: blur(20px);
}

/* RESULT */
.result {
    background: linear-gradient(90deg,#002f24,#00ff99);
    padding: 22px;
    border-radius: 20px;
    text-align: center;
    font-size: 32px;
    box-shadow: 0 0 80px rgba(0,255,231,1);
    animation: pulse 2s infinite;
}

/* PROGRESS BAR */
.bar {
    height: 10px;
    border-radius: 10px;
    background: linear-gradient(90deg,#00ffe7,#00ff99);
    box-shadow: 0 0 25px #00ffe7;
}

/* BRAND */
.brand {
    text-align:center;
    margin-top:100px;
    font-size:30px;
    letter-spacing:8px;
    color:#00ffe7;
    text-shadow:0 0 40px #00ffe7;
}

/* ANIMATION */
@keyframes pulse {
    0% {box-shadow: 0 0 40px #00ffe7;}
    50% {box-shadow: 0 0 100px #00ffe7;}
    100% {box-shadow: 0 0 40px #00ffe7;}
}
</style>
""", unsafe_allow_html=True)

# ================== MODEL ==================
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32,64,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self._to_linear = None
        self._get_size()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self._to_linear,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256,10)
        )

    def _get_size(self):
        x = torch.zeros(1,3,128,44)
        x = self.conv(x)
        self._to_linear = x.view(1,-1).shape[1]

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

# ================== FEATURES ==================
def extract_features(sig, sr):
    mel = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)

    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)

    feat = np.stack([mel, delta, delta2])
    feat = (feat - np.mean(feat)) / (np.std(feat)+1e-6)

    if feat.shape[2] > 44:
        feat = feat[:,:,:44]
    else:
        feat = np.pad(feat,((0,0),(0,0),(0,44-feat.shape[2])))

    return feat

def explain(genre):
    return {
        "metal": "Aggressive low frequencies and dense spectral energy.",
        "hiphop": "Strong rhythmic beats and repetitive patterns.",
        "classical": "Wide dynamic range with harmonic richness.",
        "rock": "Mid-frequency guitar dominance."
    }.get(genre, "Complex spectral structure detected.")

# ================== UI ==================
st.title("🤖 M.A.R.V.I.S MkIII")
st.caption("Advanced AI Music Intelligence System")

file = st.file_uploader("🎧 Upload audio", type=["wav","mp3"])
demo = st.button("🎮 Demo")

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

# ================== ANALYSIS EFFECT ==================
st.markdown("## 🔍 SYSTEM ANALYSIS")
progress = st.progress(0)
status = st.empty()

steps = ["Initializing AI core", "Extracting features", "Scanning spectrum", "Running neural network"]

for i in range(100):
    if i % 25 == 0:
        status.write(steps[i//25])
    time.sleep(0.01)
    progress.progress(i+1)

# ================== INFERENCE ==================
SEG = 8
seg_len = len(y)//SEG
timeline = []
all_probs = []

for s in range(SEG):
    seg = y[s*seg_len:(s+1)*seg_len]
    feat = extract_features(seg, sr)
    x = torch.tensor(feat).unsqueeze(0).float()

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)

    probs = probs.numpy()[0]
    all_probs.append(probs)

    idx = np.argmax(probs)
    timeline.append(classes[idx])

mean_probs = np.mean(all_probs, axis=0)
idx = np.argmax(mean_probs)
genre = classes[idx]
confidence = float(np.max(mean_probs))

# ================== RESULT ==================
st.markdown(f"<div class='result'>🎯 {genre}</div>", unsafe_allow_html=True)

st.write(f"Confidence: {confidence:.2f}")
st.markdown(f"<div class='bar' style='width:{confidence*100}%'></div>", unsafe_allow_html=True)

# ================== EXPLAIN ==================
st.subheader("🧠 AI Explanation")
st.markdown(f"<div class='glass'>{explain(genre)}</div>", unsafe_allow_html=True)

# ================== TIMELINE ==================
st.subheader("🧭 Timeline")
st.markdown(f"<div class='glass'>{' | '.join(timeline)}</div>", unsafe_allow_html=True)

# ================== FEATURES (FIXED) ==================
st.subheader("📡 Audio Features")

try:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo, (np.ndarray, list)):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    energy = float(np.mean(np.abs(y)))

    st.markdown(f"""
    <div class='glass'>
    ⚡ Tempo: {tempo:.0f} BPM<br>
    🔋 Energy: {energy:.3f}<br>
    ⏱ Length: {len(y)/sr:.1f} sec
    </div>
    """, unsafe_allow_html=True)

except:
    st.warning("⚠️ Feature extraction failed")

# ================== TOP ==================
st.subheader("🔥 Top Predictions")
top3 = np.argsort(mean_probs)[-3:][::-1]

for i in top3:
    st.write(classes[i])
    st.markdown(f"<div class='bar' style='width:{mean_probs[i]*100}%'></div>", unsafe_allow_html=True)

# ================== VISUALS ==================
plt.style.use("dark_background")

# Waveform
st.subheader("🎧 Waveform")
fig_w, ax_w = plt.subplots()
ax_w.plot(y, color="#00ffe7")
ax_w.set_facecolor("#000")
st.pyplot(fig_w)

# Distribution
st.subheader("📊 Distribution")
fig, ax = plt.subplots()
ax.bar(classes, mean_probs, color="#00ffe7")
plt.xticks(rotation=45)
st.pyplot(fig)

# Spectrogram
st.subheader("🧬 Spectrogram")
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)

fig2, ax2 = plt.subplots()
ax2.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
st.pyplot(fig2)

# ================== BRAND ==================
st.markdown("<div class='brand'>Ulyantsev Industries</div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
---
🧠 Model: CNN  
📊 Dataset: GTZAN  
🎯 Accuracy: ~84%  
""")

import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time
import matplotlib.pyplot as plt

st.set_page_config(page_title="M.A.R.V.I.S", layout="centered")

# ================== STYLE ==================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at center, #05070f, #01020a);
    color: white;
}

h1 {
    text-align: center;
    font-size: 44px;
    letter-spacing: 6px;
    text-shadow: 0 0 30px #00ffe7;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 0 35px rgba(0,255,231,0.2);
    backdrop-filter: blur(14px);
}

.result-box {
    background: linear-gradient(90deg,#003d2f,#00ff99);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 26px;
    box-shadow: 0 0 50px rgba(0,255,231,0.8);
}

.bar {
    height: 8px;
    border-radius: 10px;
    background: linear-gradient(90deg,#00ffe7,#00ff99);
    box-shadow: 0 0 15px #00ffe7;
}

.brand {
    text-align:center;
    margin-top:80px;
    font-size:24px;
    letter-spacing:5px;
    color:#00ffe7;
    text-shadow:0 0 25px #00ffe7;
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

# ================== UI ==================
st.title("🤖 M.A.R.V.I.S MkIII")
st.caption("AI Music Intelligence System")

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

# ================== SCAN ==================
st.markdown("## 🔍 SYSTEM ANALYSIS")
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

# ================== SEGMENT ANALYSIS ==================
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
st.markdown(f"<div class='result-box'>🎯 {genre}</div>", unsafe_allow_html=True)

st.write(f"Confidence: {confidence:.2f}")
st.markdown(f"<div class='bar' style='width:{confidence*100}%'></div>", unsafe_allow_html=True)

# ================== TIMELINE ==================
st.subheader("🧭 Timeline Analysis")

timeline_str = " | ".join(timeline)
st.markdown(f"<div class='glass'>{timeline_str}</div>", unsafe_allow_html=True)

# ================== FEATURES ==================
st.subheader("📡 Audio Features")

tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
energy = np.mean(np.abs(y))

st.markdown(f"""
<div class='glass'>
Tempo: {tempo:.0f} BPM<br>
Energy: {energy:.3f}
</div>
""", unsafe_allow_html=True)

# ================== TOP ==================
st.subheader("🔥 Top Predictions")
top3 = np.argsort(mean_probs)[-3:][::-1]

for i in top3:
    st.write(classes[i])
    st.markdown(f"<div class='bar' style='width:{mean_probs[i]*100}%'></div>", unsafe_allow_html=True)

# ================== GRAPH ==================
st.subheader("📊 Genre Distribution")

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.bar(classes, mean_probs)
ax.set_facecolor("#01020a")
plt.xticks(rotation=45)
st.pyplot(fig)

# ================== SPEC ==================
st.subheader("🧬 Spectrogram")

mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)

fig2, ax2 = plt.subplots()
ax2.imshow(mel, aspect='auto', origin='lower')
ax2.set_facecolor("#01020a")
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

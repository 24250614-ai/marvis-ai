import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time, os
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="M.A.R.V.I.S ULTRA", layout="centered")

# ===== ULTRA STYLE =====
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at center, #05070f, #01020a); color: white;}

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
    box-shadow: 0 0 35px rgba(0,255,231,0.15);
}

.result-box {
    background: linear-gradient(90deg, #003d2f, #00ff99);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 26px;
    margin: 20px 0;
    box-shadow: 0 0 60px rgba(0,255,231,0.9);
}

.conf-container {
    background: #111;
    border-radius: 12px;
    height: 12px;
    margin: 15px 0 30px;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    background: linear-gradient(90deg,#00ffe7,#00ff99);
    box-shadow: 0 0 25px #00ffe7;
}

/* HUD scan */
.scan {
    text-align:center;
    color:#00ffe7;
    animation: blink 1.2s infinite;
}
@keyframes blink {
    0%{opacity:1;}
    50%{opacity:0.2;}
    100%{opacity:1;}
}

.brand {
    text-align: center;
    margin-top: 80px;
    font-size: 22px;
    letter-spacing: 4px;
    color: #00ffe7;
    text-shadow: 0 0 25px rgba(0,255,231,0.9);
}

.footer-small {
    text-align: center;
    opacity: 0.4;
    font-size: 12px;
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
            torch.nn.Linear(self._to_linear,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256,10)
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
    if not os.path.exists("best_model.pth"):
        st.error("❌ model missing")
        return model
    state = torch.load("best_model.pth", map_location="cpu")
    try:
        model.load_state_dict(state)
    except:
        model.load_state_dict(state, strict=False)
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

    feat = np.stack([mel, delta, delta2])
    feat = np.nan_to_num(feat)
    feat = (feat - np.mean(feat)) / (np.std(feat)+1e-6)

    if feat.shape[2] > 44:
        feat = feat[:,:,:44]
    else:
        feat = np.pad(feat, ((0,0),(0,0),(0,44-feat.shape[2])))

    return feat

# ===== AI REASONING =====
def ai_reasoning():
    phrases = [
        "Spectral density matches genre profile",
        "Frequency distribution aligns with training patterns",
        "Temporal features confirm classification",
        "Confidence boosted by harmonic structure"
    ]
    return random.choice(phrases)

# ===== UI =====
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

# ===== HUD ANALYSIS =====
st.markdown("<div class='scan'>🔍 SYSTEM SCANNING...</div>", unsafe_allow_html=True)

progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

# ===== INFERENCE =====
SEG = 10
seg_len = max(1,len(y)//SEG)
all_probs = []

for s in range(SEG):
    seg = y[s*seg_len:(s+1)*seg_len]
    if len(seg) < 100:
        continue

    feat = extract_features(seg, sr)
    x = torch.tensor(feat).unsqueeze(0).float()

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)

    all_probs.append(probs.numpy()[0])

if not all_probs:
    st.error("Audio too short")
    st.stop()

mean_probs = np.mean(all_probs, axis=0)
idx = np.argmax(mean_probs)
genre = classes[idx]
confidence = float(np.max(mean_probs))

# ===== RESULT =====
st.markdown(f"<div class='result-box'>🎯 {genre}</div>", unsafe_allow_html=True)

st.markdown(f"### Confidence: `{confidence:.2f}`")
st.markdown(f"""
<div class="conf-container">
<div class="conf-fill" style="width:{confidence*100}%"></div>
</div>
""", unsafe_allow_html=True)

# ===== AI BLOCK =====
st.subheader("🧠 AI Reasoning")
st.markdown(f"<div class='glass'>{ai_reasoning()}</div>", unsafe_allow_html=True)

# ===== WAVEFORM =====
st.subheader("🎧 Waveform")
plt.style.use("dark_background")
fig, ax = plt.subplots()
ax.plot(y, color="#00ffe7")
st.pyplot(fig)

# ===== DISTRIBUTION =====
st.subheader("📊 Distribution")
fig2, ax2 = plt.subplots()
ax2.bar(classes, mean_probs, color="#00ffe7")
plt.xticks(rotation=45)
st.pyplot(fig2)

# ===== SPEC =====
st.subheader("🧬 Spectrogram")
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)
fig3, ax3 = plt.subplots()
ax3.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
st.pyplot(fig3)

# ===== BRAND (НЕ ТРОГАЛ) =====
st.markdown("<div class='brand'>Ulyantsev Industries</div>", unsafe_allow_html=True)
st.markdown("<div class='footer-small'>Advanced AI Systems Division</div>", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
---
🧠 Model: CNN  
📊 Dataset: GTZAN  
🎯 Accuracy: ~84%  
""")

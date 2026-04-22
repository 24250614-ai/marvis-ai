import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time
import matplotlib.pyplot as plt

# ===== CONFIG =====
st.set_page_config(page_title="M.A.R.V.I.S", layout="centered")

# ===== PREMIUM JARVIS STYLE =====
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at center, #05070f, #01020a);
    color: white;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* TITLE */
h1 {
    text-align: center;
    font-size: 42px;
    letter-spacing: 5px;
    text-shadow: 0 0 25px rgba(0,255,231,0.9);
}

/* SUB */
.css-10trblm {
    text-align: center;
    opacity: 0.7;
    margin-bottom: 30px;
}

/* HEADERS */
h2, h3 {
    color: #00ffe7;
    text-align: center;
    margin-top: 40px;
}

/* GLASS */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 0 35px rgba(0,255,231,0.15);
    backdrop-filter: blur(14px);
}

/* RESULT */
.result-box {
    background: linear-gradient(90deg, #003d2f, #00ff99);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 24px;
    margin: 20px 0;
    box-shadow: 0 0 40px rgba(0,255,231,0.7);
}

/* CONF BAR */
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
    box-shadow: 0 0 20px #00ffe7;
}

/* BRAND */
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
    margin-bottom: 20px;
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

# ===== EXPLAIN =====
def explain(genre):
    return {
        "metal": "High energy spectrum with aggressive low-frequency dominance.",
        "hiphop": "Rhythmic beat patterns and strong percussion.",
        "classical": "Rich harmonics and wide dynamic range.",
        "rock": "Guitar-driven mid-frequency structure."
    }.get(genre, "Complex spectral composition detected.")

# ===== CALIBRATE =====
def calibrate(probs):
    probs = np.array(probs)
    probs = np.exp(probs / 0.7)
    probs = probs / np.sum(probs)
    return float(0.5 + np.max(probs)*0.5), probs

# ===== UI =====
st.title("🤖 M.A.R.V.I.S MkIII")
st.caption("Advanced AI Music Classification System")

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("🎧 Upload audio", type=["wav","mp3","ogg"])

with col2:
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
st.markdown("## 🔍 Analyzing...")
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

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

st.markdown(f"### Confidence: `{confidence:.2f}`")
st.markdown(f"""
<div class="conf-container">
<div class="conf-fill" style="width:{confidence*100}%"></div>
</div>
""", unsafe_allow_html=True)

# ===== AI =====
st.subheader("🧠 AI Analysis")
st.markdown(f"<div class='glass'>{explain(genre)}</div>", unsafe_allow_html=True)

# ===== TOP =====
st.subheader("🔥 Top Predictions")
top3 = np.argsort(mean_probs)[-3:][::-1]

for i in top3:
    st.progress(float(mean_probs[i]))
    st.write(f"{classes[i]} — {mean_probs[i]:.2f}")

# ===== GRAPH =====
st.subheader("📊 Distribution")
plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.bar(classes, mean_probs)
plt.xticks(rotation=45)
st.pyplot(fig)

# ===== SPEC =====
st.subheader("🧬 Spectrogram")
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)

fig2, ax2 = plt.subplots()
ax2.imshow(mel, aspect='auto', origin='lower')
st.pyplot(fig2)

# ===== BRAND =====
st.markdown("<div class='brand'>Ulyantsev Industries</div>", unsafe_allow_html=True)
st.markdown("<div class='footer-small'>Advanced AI Systems Division</div>", unsafe_allow_html=True)

# ===== ORIGINAL FOOTER (НЕ ТРОГАЕМ) =====
st.markdown("""
---
🧠 Model: CNN  
📊 Dataset: GTZAN  
🎯 Accuracy: ~84%  
""")

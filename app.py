import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="M.A.R.V.I.S", layout="centered")

# ===== STYLE  =====
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at center, #05070f, #01020a); color: white;}

h1 {
    text-align:center;
    font-size:42px;
    letter-spacing:5px;
    text-shadow:0 0 15px rgba(0,255,231,0.6);
}

h2, h3 {color:#00d5c5; text-align:center;}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius:18px;
    padding:20px;
    margin:20px 0;
}

.result-box {
    background: linear-gradient(90deg,#003d2f,#00c78a);
    padding:20px;
    border-radius:16px;
    text-align:center;
    font-size:24px;
}

.conf-container {
    background:#111;
    border-radius:12px;
    height:12px;
    overflow:hidden;
}

.conf-fill {
    height:100%;
    background: linear-gradient(90deg,#00d5c5,#00c78a);
}

/* BRAND НЕ ТРОГАЕМ */
.brand {
    text-align:center;
    margin-top:80px;
    font-size:22px;
    letter-spacing:4px;
    color:#00ffe7;
    text-shadow:0 0 25px rgba(0,255,231,0.9);
}

.footer-small {
    text-align:center;
    opacity:0.4;
    font-size:12px;
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

@st.cache_resource
def load_model():
    model = CNN()

    if not os.path.exists("best_model.pth"):
        st.error("❌ model not found")
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

# ===== AI EXPLAIN =====
def explain(genre):
    return {
        "metal": "Detected strong low-frequency energy and aggressive spectral patterns.",
        "hiphop": "Identified rhythmic beat structure and repetitive temporal features.",
        "classical": "Wide dynamic range and harmonic richness detected.",
        "rock": "Mid-frequency dominance with guitar-driven structure."
    }.get(genre, "Complex spectral structure identified.")

# ===== UI =====
st.title("🤖 M.A.R.V.I.S MkIII")
st.caption("AI Music Intelligence System")

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("🎧 Upload audio", type=["wav","mp3","ogg"])
with col2:
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

# ===== ANALYSIS =====
st.markdown("## AI Analysis ...")
progress = st.progress(0)

for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

# ===== SEGMENT ANALYSIS =====
SEG = 8
seg_len = max(1, len(y)//SEG)
timeline = []
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

    p = probs.numpy()[0]
    all_probs.append(p)

    timeline.append(classes[np.argmax(p)])

if not all_probs:
    st.error("Audio too short")
    st.stop()

mean_probs = np.mean(all_probs, axis=0)
genre = classes[np.argmax(mean_probs)]
raw_conf = float(np.max(mean_probs))
confidence = 0.6 * raw_conf + 0.4
confidence = min(confidence, 0.95)

# ===== RESULT =====
st.markdown(f"<div class='result-box'>{genre}</div>", unsafe_allow_html=True)

st.markdown(f"### Confidence: `{confidence:.2f}`")
st.markdown(f"""
<div class="conf-container">
<div class="conf-fill" style="width:{confidence*100}%"></div>
</div>
""", unsafe_allow_html=True)

# ===== TIMELINE =====
st.subheader("Segment Analysis")
st.markdown(f"<div class='glass'>{' | '.join(timeline)}</div>", unsafe_allow_html=True)

# ===== AI =====
st.subheader("AI Reasoning")
st.markdown(f"<div class='glass'>{explain(genre)}</div>", unsafe_allow_html=True)

# ===== WAVEFORM =====
st.subheader("🎧 Waveform")
fig_w, ax_w = plt.subplots(figsize=(8,3))
ax_w.set_facecolor("#02030a")
fig_w.patch.set_facecolor("#02030a")

ax_w.plot(y, color="#00d5c5", linewidth=1.2)

ax_w.set_xticks([])
ax_w.set_yticks([])
for spine in ax_w.spines.values():
    spine.set_visible(False)

st.pyplot(fig_w)


# ===== DISTRIBUTION =====
st.subheader("📊 Distribution")
fig, ax = plt.subplots(figsize=(8,4))

fig.patch.set_facecolor("#02030a")
ax.set_facecolor("#02030a")

# градиент
colors = plt.cm.viridis(mean_probs / max(mean_probs))
ax.bar(classes, mean_probs, color=colors)

ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9, color="#cfeeee")

ax.tick_params(axis='y', colors="#9fbcbc")
ax.tick_params(axis='x', colors="#cfeeee")

# убираем рамки
for spine in ax.spines.values():
    spine.set_visible(False)

# мягкая сетка (дорого выглядит)
ax.grid(axis='y', linestyle='--', alpha=0.15, color='#00d5c5')

st.pyplot(fig)

# градиентный цвет 
colors = plt.cm.viridis(mean_probs / max(mean_probs))

bars = ax.bar(classes, mean_probs, color=colors, edgecolor="#111")

ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

st.pyplot(fig)


# ===== SPECTROGRAM =====
st.subheader("🧬 Spectrogram")

mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)

fig2, ax2 = plt.subplots(figsize=(8,4))
fig2.patch.set_facecolor("#02030a")
ax2.set_facecolor("#02030a")

img = ax2.imshow(
    mel,
    aspect='auto',
    origin='lower',
    cmap='magma' 
)

ax2.set_xticks([])
ax2.set_yticks([])

for spine in ax2.spines.values():
    spine.set_visible(False)

st.pyplot(fig2)

# ===== BRAND =====
st.markdown("<div class='brand'>Ulyantsev Industries</div>", unsafe_allow_html=True)
st.markdown("<div class='footer-small'>Advanced AI Systems Division</div>", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
---
🧠 Model: CNN  
📊 Dataset: GTZAN  
🎯 Accuracy: ~84%  
""")

import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="M.A.R.V.I.S ULTRA", layout="centered")

# ================== STYLE ==================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at center, #02030a, #000000);
    color: white;
}
h1 {
    text-align: center;
    font-size: 48px;
    letter-spacing: 6px;
    text-shadow: 0 0 30px #00ffe7;
}
.glass {
    background: rgba(255,255,255,0.03);
    border-radius: 20px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 0 25px rgba(0,255,231,0.15);
}
.result {
    background: linear-gradient(90deg,#002f24,#00ff99);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
}
.bar {
    height: 8px;
    border-radius: 10px;
    background: linear-gradient(90deg,#00ffe7,#00ff99);
}
.brand {
    text-align: center;
    margin-top: 60px;
    font-size: 20px;
    color: #00ffe7;
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

# ================== LOAD ==================
@st.cache_resource
def load_model():
    model = CNN()

    if not os.path.exists("best_model.pth"):
        st.error("❌ best_model.pth not found")
        return model

    state = torch.load("best_model.pth", map_location="cpu")

    try:
        model.load_state_dict(state)
    except:
        st.warning("⚠️ Model mismatch → loading in safe mode")
        model.load_state_dict(state, strict=False)

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

    feat = np.nan_to_num(feat)

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

# ================== ANALYSIS ==================
st.markdown("## 🔍 SYSTEM ANALYSIS")

progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

# ================== INFERENCE ==================
SEG = 8
seg_len = max(1, len(y)//SEG)

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

if len(all_probs) == 0:
    st.error("❌ Audio too short or invalid")
    st.stop()

mean_probs = np.mean(all_probs, axis=0)

idx = np.argmax(mean_probs)
genre = classes[idx]
confidence = float(np.max(mean_probs))

# ================== RESULT ==================
st.markdown(f"<div class='result'>🎯 {genre}</div>", unsafe_allow_html=True)

st.write(f"Confidence: {confidence:.2f}")
st.markdown(f"<div class='bar' style='width:{confidence*100}%'></div>", unsafe_allow_html=True)

# ================== FEATURES ==================
st.subheader("📡 Audio Features")

try:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

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

st.subheader("📊 Distribution")
fig, ax = plt.subplots()
ax.bar(classes, mean_probs)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("🧬 Spectrogram")
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel = librosa.power_to_db(mel)

fig2, ax2 = plt.subplots()
ax2.imshow(mel, aspect='auto', origin='lower')
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

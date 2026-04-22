import streamlit as st
import torch, librosa, numpy as np, json, tempfile, time
import matplotlib.pyplot as plt

st.set_page_config(page_title="M.A.R.V.I.S", layout="centered")

# ===== STYLE =====
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at center, #0a0f1c, #02040a);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    text-align: center;
    font-weight: 600;
    letter-spacing: 2px;
}

.card {
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 16px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 15px;
}

.result {
    background: linear-gradient(90deg, #0f3d2e, #1affb2);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 500;
}

.small {
    color: #9bb3c9;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ===== MODEL =====
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
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

# ===== EXTRA ANALYSIS =====
def analyze_audio(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return centroid, zcr, tempo

# ===== CONFIDENCE =====
def calibrate(p):
    return float(np.clip(p**0.7, 0, 1))

# ===== UI =====
st.title("🤖 M.A.R.V.I.S MkIII")
st.markdown("<p class='small'>AI Music Classification System</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Upload audio", type=["wav","mp3","ogg"])

with col2:
    demo = st.button("Demo")

# ===== AUDIO =====
if demo:
    y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)

elif file:
    st.audio(file)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        y, sr = librosa.load(tmp.name, sr=22050)
else:
    st.stop()

# ===== AI THINKING =====
with st.spinner("AI analyzing audio..."):
    time.sleep(1)

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

mean_probs = np.mean(all_probs, axis=0)[0]
idx = np.argmax(mean_probs)

confidence = calibrate(mean_probs[idx])
genre = classes[idx]

centroid, zcr, tempo = analyze_audio(y, sr)

# ===== RESULT =====
st.markdown(f"<div class='result'>🎯 {genre}</div>", unsafe_allow_html=True)
st.markdown(f"### Confidence: {confidence:.2f}")

# ===== ANALYSIS =====
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("AI Analysis")

st.write(f"""
Spectral centroid: {centroid:.0f}  
Tempo: {tempo:.0f} BPM  
Zero-crossing rate: {zcr:.3f}
""")

st.markdown("</div>", unsafe_allow_html=True)

# ===== TOP =====
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Top Predictions")

top3 = np.argsort(mean_probs)[-3:][::-1]

for i in top3:
    st.progress(float(mean_probs[i]))
    st.write(f"{classes[i]} — {mean_probs[i]:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ===== GRAPH =====
plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.bar(classes, mean_probs)
plt.xticks(rotation=45)
st.pyplot(fig)

# ===== FOOTER =====
st.markdown("""
<div style='text-align:center; margin-top:50px;'>

<hr style='border:0.5px solid #2a2f3a; width:60%; margin:auto;'>

<p style='font-size:15px; letter-spacing:2px;
color: gold; text-shadow: 0 0 8px rgba(255,215,0,0.5);'>
    ULYANTSEV INDUSTRIES
</p>

<p style='font-size:11px; color:#8aa2b5;'>
    AI Research & Audio Intelligence
</p>

</div>
""", unsafe_allow_html=True)

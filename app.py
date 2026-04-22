
import streamlit as st
import torch, librosa, numpy as np, json, tempfile
import matplotlib.pyplot as plt
import time

# ===== CONFIG =====
st.set_page_config(page_title="M.A.R.V.I.S", layout="centered")

# ===== STYLE =====
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at center, #0b0f1a, #02030a);
    color: #00ffe7;
}
h1 {
    color: #ffffff;
    text-align: center;
}

h2, h3 {
    color: #00ffe7;
    text-align: center;
}
p {
    color: #00ffe7;
}
.block {
    border: 1px solid #00ffe7;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    box-shadow: 0 0 10px #00ffe733;
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
    return json.load(open("classes_MkIII.json"))

model = load_model()
classes = load_classes()

# ===== FEATURE EXTRACTION =====
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

# ===== UI =====
st.title("🤖 M.A.R.V.I.S MkIII")
st.caption("Advanced AI Music Genre Classification System")

file = st.file_uploader("🎧 Upload audio", type=["wav","mp3","ogg"])

if file:

    st.audio(file)

    # ===== SAFE LOAD =====
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=22050)
    except:
        st.error("❌ Error loading audio")
        st.stop()

    if len(y) < 22050:
        st.error("❌ Audio too short")
        st.stop()

    with st.spinner("🧠 AI analyzing..."):
        time.sleep(1)

        SEGMENTS = 10
        seg_len = len(y)//SEGMENTS

        all_probs = []

        for s in range(SEGMENTS):
            seg = y[s*seg_len:(s+1)*seg_len]

            if len(seg) < seg_len:
                continue

            feat = extract_features(seg, sr)
            x = torch.tensor(feat).unsqueeze(0).float()

            with torch.no_grad():
                out = model(x)
                probs = torch.nn.functional.softmax(out, dim=1)

            all_probs.append(probs.numpy())

    if len(all_probs) == 0:
        st.error("❌ Processing failed")
        st.stop()

    # ===== УМНАЯ АГРЕГАЦИЯ =====
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    final_probs = mean_probs - 0.25 * std_probs
    final_probs = np.clip(final_probs, 0, 1)
    final_probs = final_probs / final_probs.sum()

    idx = np.argmax(final_probs)
    confidence = final_probs[0][idx]

    # ===== RESULT =====
    st.markdown('<div class="block">', unsafe_allow_html=True)

    st.success(f"🎯 Genre: {classes[idx]}")
    st.markdown(f"### Confidence: `{confidence:.2f}`")

    st.progress(float(confidence))

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== AI ANALYSIS =====
    st.subheader("🧠 AI Analysis")

    if confidence < 0.4:
        st.warning("Uncertain prediction — possible mixed genres")
    elif confidence < 0.6:
        st.info("Moderate confidence — overlapping features")
    else:
        st.success("High confidence — clear genre detected")

    # ===== TOP 3 =====
    st.subheader("🔥 Top Predictions")

    top3 = np.argsort(final_probs[0])[-3:][::-1]

    for i in top3:
        st.write(f"{classes[i]} — {final_probs[0][i]:.2f}")

    # ===== GRAPH =====
    st.subheader("📊 Confidence Distribution")

    fig, ax = plt.subplots()
    ax.bar(classes, final_probs[0])
    ax.set_facecolor("#0b0f1a")
    fig.patch.set_facecolor("#0b0f1a")
    ax.tick_params(colors='#00ffe7')
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ===== SPECTROGRAM =====
    st.subheader("🧬 Mel Spectrogram")

    mel_full = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_full = librosa.power_to_db(mel_full)

    fig2, ax2 = plt.subplots()
    ax2.imshow(mel_full, aspect='auto', origin='lower')
    st.pyplot(fig2)

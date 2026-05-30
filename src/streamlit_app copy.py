import time
import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs import DatasetMagnetometerDetection

# --- 1. STREAMLIT CONFIG & DARK THEME ---
st.set_page_config(
    page_title="Magnetometer Vehicle Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. CACHED MODEL & DATA INITIALIZATION ---
@st.cache_resource
def load_model():
    model = torch.load("results/rnn_model_detector/model_9.pt", weights_only=False, map_location='cpu')
    model.eval()
    return model

@st.cache_resource
def load_dataset():    
    return DatasetMagnetometerDetection("/users/michal/datasets/car_detection_2/Meranie_20_06_01-Lietavska_Lucka/01/")

model = load_model()
dataset = load_dataset()

# --- 3. UI HEADER ---
st.title("Magnetometer Car Detection Dashboard")
st.subheader("Real-time RNN Inference Window")

col1, col2 = st.columns(2)
with col1:
    fps_limit = st.slider("Simulation Speed (Delay in seconds)", 0.01, 0.1, 0.02, step=0.01)
with col2:
    car_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.7, step=0.05)

status_indicator = st.empty()
chart_placeholder = st.empty()

# --- 4. PERSISTENT PLOTLY BASE CONFIG ---
def create_base_figure():
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("Magnetometer Raw Signals (1024-Sample History Window)", "Car Detection State")
    )
    
    # Pre-populate empty traces
    fig.add_trace(go.Scatter(y=[], name="X-axis", line=dict(color="#FF4B4B")), row=1, col=1)
    fig.add_trace(go.Scatter(y=[], name="Y-axis", line=dict(color="#00CC96")), row=1, col=1)
    fig.add_trace(go.Scatter(y=[], name="Z-axis", line=dict(color="#636EFA")), row=1, col=1)
    fig.add_trace(go.Scatter(y=[], name="Detection", line=dict(color="#EF553B", width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(y=[], name="RNN Prediction", line=dict(color="#19D3F3", width=2.5)), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111723",
        plot_bgcolor="#0E1117",
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(range=[-3, 3], row=1, col=1)
    fig.update_yaxes(range=[-0.1, 1.2], row=2, col=1)
    fig.update_xaxes(range=[0, 1024])
    return fig

# --- 5. INFINITE STREAMING LOOP ---
history_x = np.zeros((0, 3))
#history_y_gt = np.zeros((0, 1))
history_y_detection = np.zeros((0, 1))
history_y_pred = np.zeros((0, 1))

base_fig = create_base_figure()
frame_counter = 0  # Counter to bypass the DuplicateElementId bug

while True:  
    for n in range(len(dataset)):
        # 1. Fetch data -> x shape: (512, 3), y_gt shape: (512, 1)
        x, y_gt = dataset.get(n*10, 1024) # [n*10]
        
        # 2. PyTorch Model Inference
        # Input shape expected: (batch, seq_len, 3) -> unsqueeze(0) gives (1, 512, 3)
        x_t = torch.from_numpy(x).float().unsqueeze(0)
        
        with torch.no_grad():
            y_pred_t = model(x_t)
            y_pred_t = torch.nn.functional.sigmoid(y_pred_t)
            
        # Output shape is (1, 512, 1) -> squeeze(0) returns it to (512, 1)
        y_pred = y_pred_t.cpu().numpy().squeeze(0)

        # 3. Concatenate and slide window to maintain 1024 history context
        history_x = np.vstack([history_x, x])
        #history_y_gt = np.vstack([history_y_gt, y_gt])
        history_y_detection = np.vstack([history_y_detection, y_pred > car_threshold])
        history_y_pred = np.vstack([history_y_pred, y_pred])
        
        if len(history_x) > 1024:
            history_x = history_x[-1024:]
            #history_y_gt = history_y_gt[-1024:]
            history_y_detection = history_y_detection[-1024:]
            history_y_pred = history_y_pred[-1024:]

        # Wait until history window is full (1024 items)
        if len(history_x) < 1024:
            continue

        # 4. MUTATE EXISTING PLOTLY TRACES
        base_fig.data[0].y = history_x[:, 0]
        base_fig.data[1].y = history_x[:, 1]
        base_fig.data[2].y = history_x[:, 2]
        base_fig.data[3].y = history_y_detection[:, 0]
        base_fig.data[4].y = history_y_pred[:, 0]

        # 5. UI Status Indicator Box
        is_car_present = history_y_pred[-1, 0] > car_threshold
        if is_car_present:
            status_indicator.error("🚨 CAR DETECTED IN MONITORING ZONE")
        else:
            status_indicator.success("🟢 ZONE CLEAR — SCANNING")

        # 6. RENDER WITH UNIQUE DYNAMIC KEY
        # Changing the key string slightly on each loop execution satisfies the Element ID requirement
        chart_placeholder.plotly_chart(
            base_fig, 
            use_container_width=True, 
            config={'displayModeBar': False, 'responsive': False},
            key=f"magnetometer_chart_{frame_counter}"
        )
        
        frame_counter += 1
        time.sleep(fps_limit)
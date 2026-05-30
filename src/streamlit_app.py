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
def load_detection_model():
    model = torch.load("results/rnn_model_detector/model_9.pt", weights_only=False, map_location='cpu')
    model.eval()
    return model

@st.cache_resource
def load_classification_model():
    model = torch.load("results/cnn_model_classifier/model_9.pt", weights_only=False, map_location='cpu')
    model.eval()
    return model

@st.cache_resource
def load_dataset():    
    return DatasetMagnetometerDetection("/users/michal/datasets/car_detection_2/Kysuce/")

# Vehicle mapping dictionary (Skipping index 5 based on your list, ignoring index 6)
VEHICLE_CLASSES = {
    0: "⚡ Motorcycle",
    1: "🚗 Car",
    2: "🚐 LGV",
    3: "🚜 ORV",
    4: "🚛 Heavy Truck",
    5: "❓ Unknown",
    6: "📦 Other"   
}

detection_model      = load_detection_model()
classification_model = load_classification_model()
dataset = load_dataset()

# --- 3. UI HEADER ---
st.title("Magnetometer Vehicle Analysis Dashboard")
st.subheader("Real-time RNN Detection & CNN Classification")

col1, col2 = st.columns(2)
with col1:
    fps_limit = st.slider("Simulation Speed (Delay in seconds)", 0.01, 0.1, 0.02, step=0.01)
with col2:
    car_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.7, step=0.05)

# Status indicators split into side-by-side columns
status_col, class_col = st.columns(2)
status_indicator = status_col.empty()
classification_indicator = class_col.empty()

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
    fig.add_trace(go.Scatter(y=[], name="Detection Thresholded", line=dict(color="#EF553B", width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(y=[], name="RNN Confidence", line=dict(color="#19D3F3", width=2.5)), row=2, col=1)

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
history_y_detection = np.zeros((0, 1))
history_y_pred = np.zeros((0, 1))

base_fig = create_base_figure()
frame_counter = 0  

while True:  
    for n in range(len(dataset)):
        window_size = 1024
        
        # Safe edge protection check for loop wrapping points
        start_idx = n * 10
        if start_idx - 256 < 0 or start_idx + window_size >= len(dataset) * 10:
            continue
            
        x, y_gt = dataset.get(start_idx, window_size) 
        
        # PyTorch Detection Model Inference
        x_t = torch.from_numpy(x).float().unsqueeze(0)
        with torch.no_grad():
            y_pred_t = detection_model(x_t)
            y_pred_t = torch.nn.functional.sigmoid(y_pred_t)
            
        y_pred = y_pred_t.cpu().numpy().squeeze(0)

        # Classification Segment
        detected_class_str = "🔍 Scanning..."
        is_car_present = np.max(y_pred) > car_threshold

        if is_car_present:
            # Using isolated variable names (x_class) preserves chart data arrays
            x_class, _ = dataset.get(start_idx - 256, 512)
            x_class_t = torch.from_numpy(x_class).float().unsqueeze(0)
            
            with torch.no_grad():
                logits = classification_model(x_class_t)
                # Compute probabilities from network logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
                predicted_class_idx = torch.argmax(probabilities).item()
            
            # Filter unassigned index spaces
            if predicted_class_idx == 6:
                detected_class_str = "⚠️ Class Ignored (Index 6)"
            else:
                detected_class_str = f"Type: {VEHICLE_CLASSES.get(predicted_class_idx, 'Unknown')}"
        else:
            detected_class_str = "💤 No Target Active"

        # Concatenate and slide window to maintain 1024 history context
        history_x = np.vstack([history_x, x])
        history_y_detection = np.vstack([history_y_detection, y_pred > car_threshold])
        history_y_pred = np.vstack([history_y_pred, y_pred])
        
        if len(history_x) > 1024:
            history_x = history_x[-1024:]
            history_y_detection = history_y_detection[-1024:]
            history_y_pred = history_y_pred[-1024:]

        # Wait until history window is full (1024 items)
        if len(history_x) < 1024:
            continue

        # Mutate traces inplace
        base_fig.data[0].y = history_x[:, 0]
        base_fig.data[1].y = history_x[:, 1]
        base_fig.data[2].y = history_x[:, 2]
        base_fig.data[3].y = history_y_detection[:, 0]
        base_fig.data[4].y = history_y_pred[:, 0]

        # Sync visual blocks
        if is_car_present:
            status_indicator.error("🚨 VEHICLE DETECTED IN MONITORING ZONE")
            classification_indicator.info(f"📋 {detected_class_str}")
        else:
            status_indicator.success("🟢 ZONE CLEAR — SCANNING")
            classification_indicator.empty()

        # Render step without flickering
        chart_placeholder.plotly_chart(
            base_fig, 
            use_container_width=True, 
            config={'displayModeBar': False, 'responsive': False},
            key=f"magnetometer_chart_{frame_counter}"
        )
        
        frame_counter += 1
        time.sleep(fps_limit)
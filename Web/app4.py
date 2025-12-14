import streamlit as st
import pandas as pd
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- 0. PAGE CONFIGURATION (Fixes Title Wrapping) ---
st.set_page_config(page_title="RDD System", layout="wide")

RDD_LOOKUP_TABLE = [
    # 1. B2_Full: Baseline (High Latency)
    {"name": "B2_Full", "miou": 0.8908, "gflops": 25.24, "latency_ms": 178.6, "channels": 1024, "mask": [0]*16},
    
    # 2. B2_a: Optimized Baseline (Skip 1 Block)
    # Mask: Skips last block of Stage 1
    {"name": "B2_a",    "miou": 0.8819, "gflops": 23.92, "latency_ms": 112.4, "channels": 980,  "mask": [0,0,1] + [0]*13},
    
    # 3. B2_b: Balanced (Skip 1 Block, Pruned Decoder)
    {"name": "B2_b",    "miou": 0.8819, "gflops": 23.92, "latency_ms": 108.7, "channels": 920,  "mask": [0,0,1] + [0]*13},
    
    # 4. B2_c: Efficiency Step (Skip 2 Blocks)
    # Mask: Skips last of Stage 1 & 2
    {"name": "B2_c",    "miou": 0.8635, "gflops": 22.88, "latency_ms": 107.5, "channels": 900,  "mask": [0,0,1] + [0,0,0,1] + [0]*9},
    
    # 5. B2_d: Aggressive Optimization (Skip 3 Blocks)
    # Mask: Skips last of Stage 1, 2 & 3
    {"name": "B2_d",    "miou": 0.8555, "gflops": 21.49, "latency_ms": 106.4, "channels": 880,  "mask": [0,0,1] + [0,0,0,1] + [0,0,0,0,0,1] + [0]*3},
    
    # 6. B2_e: "Super Eco" Test (Skip 4 Blocks)
    # Mask: Skips last of Stage 1, 2, 3 & 4
    {"name": "B2_e",    "miou": 0.8540, "gflops": 20.61, "latency_ms": 105.3, "channels": 832,  "mask": [0,0,1] + [0,0,0,1] + [0,0,0,0,0,1] + [0,0,1]},
    
    # 7. B2_f: "Super Eco" Floor (Skip 4 Blocks + Deep Pruning)
    {"name": "B2_f",    "miou": 0.8505, "gflops": 20.61, "latency_ms": 104.2, "channels": 768,  "mask": [0,0,1] + [0,0,0,1] + [0,0,0,0,0,1] + [0,0,1]},
]

# --- 3. MODEL ARCHITECTURE DEFINITIONS ---
class ExG_Attention(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        r, g, b = x[:,0,:,:], x[:,1,:,:], x[:,2,:,:]
        return torch.sigmoid((2*g)-r-b).unsqueeze(1)

class ParallelDetailDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes, embedding_dim=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, embedding_dim, 1) for c in encoder_channels])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim*4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
        self.prediction_head = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, features):
        projected = [l(f) for l, f in zip(self.lateral_convs, features)]
        target_size = projected[0].shape[-2:]
        upsampled = [F.interpolate(p, size=target_size, mode='bilinear', align_corners=False) for p in projected]
        concatenated = torch.cat(upsampled, dim=1)
        fused = self.linear_fuse(concatenated)
        fused = fused + projected[0]
        return self.prediction_head(fused)

class SegFormer_Research_Grade(nn.Module):
    def __init__(self, pretrained, num_classes):
        super().__init__()
        # FIX: ignore_mismatched_sizes=True prevents the crash
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
        self.exg_module = ExG_Attention()
        self.custom_decoder = ParallelDetailDecoder(self.base_model.config.hidden_sizes, num_classes)

    def forward(self, pixel_values, labels=None):
        exg_mask = self.exg_module(pixel_values)
        out = self.base_model.segformer(pixel_values, output_hidden_states=True)
        feats = list(out.hidden_states)[-4:]
        feats[-1] = feats[-1] * F.interpolate(exg_mask, size=feats[-1].shape[-2:], mode='nearest')
        logits = self.custom_decoder(feats)
        return F.interpolate(logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False)

# RDD Wrappers
class DynamicFusionWrapper(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.conv, self.bn, self.relu = original_module[0], original_module[1], original_module[2]
        self.weight, self.bias = self.conv.weight, self.conv.bias
        self.stride, self.padding = self.conv.stride, self.conv.padding
        self.active_in_channels = self.conv.in_channels

    def forward(self, x):
        x = x[:, :self.active_in_channels, :, :] # Slice Input
        sliced_w = self.weight[:, :self.active_in_channels, :, :] # Slice Weights
        x = F.conv2d(x, sliced_w, self.bias, self.stride, self.padding)
        return self.relu(self.bn(x))

class DynamicBlockWrapper(nn.Module):
    def __init__(self, block): 
        super().__init__()
        self.block = block
        self.skip = False
    def forward(self, x, h, w, out_att=False):
        return (x,) if self.skip else self.block(x, h, w, out_att)

# --- 4. MODEL LOADER (Cached) ---
@st.cache_resource
def load_rdd_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model on: {device}")
    
    model = SegFormer_Research_Grade("nvidia/segformer-b2-finetuned-ade-512-512", num_classes=3)
    
    
    weights_path = "models/custom_model_best.pth" 
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        st.error(f"Model file missing: {weights_path}")
        return None, None, None

    model.to(device)
    model.eval()

    model.custom_decoder.linear_fuse = DynamicFusionWrapper(model.custom_decoder.linear_fuse)
    
    encoder_wrappers = []
    for stage in model.base_model.segformer.encoder.block:
        for i in range(len(stage)):
            stage[i] = DynamicBlockWrapper(stage[i])
            encoder_wrappers.append(stage[i])
            
    return model, encoder_wrappers, device


model, wrappers, device = load_rdd_model()


def apply_rdd_surgery(cfg_name):
    cfg = next(c for c in RDD_LOOKUP_TABLE if c['name'] == cfg_name)
    model.custom_decoder.linear_fuse.active_in_channels = cfg['channels']
    for i, skip in enumerate(cfg['mask']):
        wrappers[i].skip = bool(skip)


def select_config_v3(manual_latency_limit, battery_level, velocity, temp):
    """
    Decides the optimal model based on:
    1. Battery (Energy Budget) -> Sets Max GFLOPs
    2. Velocity (Motion Blur/Coverage) -> Sets Max Latency
    3. Temperature (Throttling) -> Overrides GFLOPs to prevent crash
    """
    
    # --- 1. BASELINE: BATTERY LOGIC (Energy Budget) ---
    if battery_level >= 80:
        max_gflops = 26.0
        mode = "PERF"
        status_color = "green"
    elif battery_level >= 60:
        max_gflops = 24.0
        mode = "HIGH EFF"
        status_color = "green"
    elif battery_level >= 40:
        max_gflops = 23.0
        mode = "BALANCED"
        status_color = "orange"
    elif battery_level >= 20:
        max_gflops = 21.5
        mode = "ECO(Power Saving)"
        status_color = "orange"
    elif battery_level > 0:
        max_gflops = 20.7
        mode = "CRITICAL(Survival)"
        status_color = "red"
    else:
        return None, "SYSTEM OFFLINE", "gray"

    # --- 2. VELOCITY LOGIC (Motion Constraint) ---
    # Faster drone = less time per frame to avoid gaps/blur.
    # Logic: At 1 m/s we allow 300ms. At 10 m/s we only allow 30ms.
    # Formula: Allowable Latency = Constant / Velocity
    if velocity > 0.1:
        speed_latency_limit = 300 / velocity 
    else:
        speed_latency_limit = 1000 # Hovering = no strict limit
        
    # The actual constraint is the stricter of the two (Manual vs Physics)
    effective_latency_limit = min(manual_latency_limit, speed_latency_limit)

    # --- 3. THERMAL LOGIC (Throttling Prevention) ---
    # If GPU is cooking (>80°C), we MUST lower compute load immediately
    # to prevent hardware throttling (which causes massive lag spikes).
    if temp >= 80:
        max_gflops = 20.7 # Force lowest GFLOPs (B2_f)
        mode = "🔥 THERMAL THROTTLING ACTIVE"
        status_color = "red"
    elif temp >= 70:
        max_gflops = min(max_gflops, 22.0) # Cap at Medium Load
        mode = f"{mode} + 🔥 Warm"

    # --- FILTERING ---
    valid_options = []
    for cfg in RDD_LOOKUP_TABLE:
        # Check against the derived effective limits
        if cfg['latency_ms'] <= effective_latency_limit and cfg['gflops'] <= max_gflops:
            valid_options.append(cfg)

    # --- FINAL SELECTION ---
    if not valid_options:
        # Fail-safe: If velocity is crazy high (e.g. 20m/s -> 15ms limit),
        # even our fastest model (104ms) fails. In this case, pick the fastest 
        # and warn the user.
        best_config = sorted(RDD_LOOKUP_TABLE, key=lambda x: x['latency_ms'])[0]
        return best_config, "⚠️ OVERSPEED WARNING", "red"

    # Optimization Goal:
    # If Critical or Thermal, prioritize SPEED (lower heat/power duration)
    if battery_level < 20 or temp >= 75:
        valid_options.sort(key=lambda x: x['latency_ms']) 
    else:
        # Otherwise maximize ACCURACY
        valid_options.sort(key=lambda x: x['miou'], reverse=True)
    
    return valid_options[0], mode, status_color

# --- 7. UI ---
st.sidebar.header("Mission Control")

scenario = st.sidebar.selectbox(
    "Quick Scenarios", 
    [
        "Manual Control", 
        "Start of Mission (Hover)", 
        "High Speed Survey", 
        "Thermal Throttling", 
        "Low Battery Return", 
        "Critical Survival"
    ]
)

b_val, l_val, v_val, t_val = 100, 500, 0.0, 45

if scenario == "Start of Mission (Hover)":
    b_val, v_val, t_val = 100, 0.5, 40 # Hovering, cool, full battery
elif scenario == "High Speed Survey":
    b_val, v_val, t_val = 85, 8.0, 65 # Fast flying, getting warm
elif scenario == "Thermal Throttling":
    b_val, v_val, t_val = 60, 2.0, 85 # Hot day/Heavy load -> Trigger Thermal Logic
elif scenario == "Low Battery Return":
    b_val, v_val, t_val = 15, 4.0, 60 # Low battery, moderate speed return
elif scenario == "Critical Survival":
    b_val, v_val, t_val = 5, 1.0, 70 # Dying battery

# Render Sliders
battery = st.sidebar.slider("🔋 Battery Level (%)", 0, 100, b_val)
velocity = st.sidebar.slider("🚀 Drone Velocity (m/s)", 0.0, 15.0, v_val, step=0.5)
temp = st.sidebar.slider("🔥 GPU Temperature (°C)", 30, 95, t_val)
latency_manual = st.sidebar.slider("⚙️ Manual Latency Cap (ms)", 50, 500, l_val, step=10)

# --- 8. MAIN DISPLAY ---
st.title("Self-Aware RDD Segmentation System")

selected_cfg, mode, color = select_config_v3(latency_manual, battery, velocity, temp)

if selected_cfg is None:
    st.error("⚠️ SYSTEM FAILURE: BATTERY DEPLETED")
else:
    if model is not None:
        apply_rdd_surgery(selected_cfg['name'])

    speed_limit = int(300 / (velocity + 0.01)) if velocity > 0 else 1000
    effective_lat = min(latency_manual, speed_limit)

    # --- ROW 1: SYSTEM STATUS ---
    st.subheader("System Status")
    row1_1, row1_2, row1_3 = st.columns(3)
    
    row1_1.metric("System Mode", mode)
    row1_2.metric("Active Config", selected_cfg['name'])
    row1_3.metric("Energy Cost", f"{selected_cfg['gflops']} GFLOPs")

    # --- ROW 2: PHYSICS & CONSTRAINTS ---
    row2_1, row2_2, row2_3 = st.columns(3)

    # GPU Temp: Shows red delta if overheating
    row2_1.metric(
        "GPU Temp", 
        f"{temp}°C", 
        delta=f"{temp-80}°C" if temp > 80 else None, 
        delta_color="inverse"
    )

    # Latency Budget: The limit we must not exceed
    row2_2.metric(
        "Latency Budget", 
        f"<{effective_lat} ms", 
        help="The strictest limit between Manual setting and Velocity requirements"
    )

    # Model Latency: Shows how much headroom we have (green is good)
    margin = effective_lat - selected_cfg['latency_ms']
    row2_3.metric(
        "Actual Latency", 
        f"~{selected_cfg['latency_ms']} ms", 
        delta=f"{margin:.1f} ms margin"
    )

    st.markdown("---")

    st.subheader("Live Drone Feed")
    
    sample_options = [f"Sample {i}" for i in range(1, 31)]
    sample_choice = st.selectbox("Select Validation Sample:", sample_options)
    
    sample_id = sample_choice.split(" ")[1]
    fname = f"sample{sample_id}"
    img_path = f"validation_samples_5/{fname}.jpg"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Original Input")
        if os.path.exists(img_path):
            input_image = Image.open(img_path).convert("RGB")
            # UPDATED: Use the new parameter
            st.image(input_image, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_path}")

    with col2:
        st.caption(f"Real-Time Inference ({selected_cfg['name']})")
        if model is not None and os.path.exists(img_path):
            with st.spinner(f"Processing with {selected_cfg['name']}..."):
                processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
                inputs = processor(images=input_image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    logits = model(**inputs)
                    logits = F.interpolate(logits, size=input_image.size[::-1], mode="bilinear", align_corners=False)
                    pred = logits.argmax(dim=1).cpu().numpy()[0]

                # Colorize (0:Crop/Blue, 1:Weed/Green, 2:Soil/Brown)
                colors = np.array([[0, 0, 255], [0, 255, 0], [139, 69, 19]], dtype=np.uint8)
                pred_img = colors[pred % 3]
                
                # UPDATED: Use the new parameter
                st.image(pred_img, use_container_width=True)
        else:
            st.error("Model failed to load or Image missing.")

    # --- DECISION LOG (DEBUG) ---
    with st.expander("System Decision Logic"):
        c_log1, c_log2 = st.columns(2)
        
        with c_log1:
            st.markdown("### 1. Sensor Inputs")
            st.write(f"**Battery:** {battery}%")
            st.write(f"**Velocity:** {velocity} m/s")
            st.write(f"**GPU Temp:** {temp}°C")
            st.write(f"**Manual Latency:** {latency_manual} ms")

        with c_log2:
            st.markdown("### 2. Derived Constraints")
            # Recalculate for display logic clarity
            physics_lat = int(300 / (velocity + 0.01)) if velocity > 0.1 else 1000
            
            st.write(f"**Latency Limit:** {physics_lat} ms")
            if physics_lat < latency_manual:
                st.caption("⚠️Velocity overrides Manual Latency Capacity")
            
            st.write(f"🛑 **Throttling State:** {'ACTIVE' if temp >= 80 else 'Normal'}")
            
        st.markdown("---")
        st.markdown(f"### Final Decision: **{selected_cfg['name']}**")
        st.info(f"**Reasoning:** {mode}")
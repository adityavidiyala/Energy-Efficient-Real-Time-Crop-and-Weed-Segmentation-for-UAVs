import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


app = FastAPI(title="RDD Model Server")

# Enable CORS (Allows your React/Node app to talk to this server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace '*' with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class DynamicFusionWrapper(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.conv = original_module[0]
        self.bn = original_module[1]
        self.relu = original_module[2]
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.active_in_channels = self.conv.in_channels

    def forward(self, x):
        # Dynamic Slicing based on active_in_channels
        x = x[:, :self.active_in_channels, :, :] 
        sliced_w = self.weight[:, :self.active_in_channels, :, :] 
        x = F.conv2d(x, sliced_w, self.bias, self.stride, self.padding)
        return self.relu(self.bn(x))

class DynamicBlockWrapper(nn.Module):
    def __init__(self, block): 
        super().__init__()
        self.block = block
        self.skip = False # Default state
    def forward(self, x, h, w, out_att=False):
        return (x,) if self.skip else self.block(x, h, w, out_att)
    


device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
encoder_wrappers = []
processor = None

@app.on_event("startup")
async def load_model():
    global model, encoder_wrappers, processor
    print(f"Loading Model on {device}...")
    
    # Initialize Architecture
    model = SegFormer_Research_Grade("nvidia/segformer-b2-finetuned-ade-512-512", num_classes=3)
    
    # Load Weights
    try:
        weights_path = "models/custom_model_best.pth"
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Weights Loaded Successfully")
    except FileNotFoundError:
        print("Warning: 'custom_model_best.pth' not found. Using random initialization.")

    model.to(device)
    model.eval()

    # Apply Wrappers for Dynamic Inference
    model.custom_decoder.linear_fuse = DynamicFusionWrapper(model.custom_decoder.linear_fuse)
    
    encoder_wrappers = []
    for stage in model.base_model.segformer.encoder.block:
        for i in range(len(stage)):
            stage[i] = DynamicBlockWrapper(stage[i])
            encoder_wrappers.append(stage[i])
            
    # Load Processor
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    print("Server Ready!")


@app.get("/")
def health_check():
    return {"status": "active", "device": device}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    mask: str = Form("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"), # Default: Full Model
    channels: int = Form(1024)
):
    """
    Receives an image and RDD configuration parameters.
    1. Applies 'Surgery' (skipping blocks) based on the mask.
    2. Runs inference.
    3. Returns the colorized segmentation map.
    """
    
    # --- 1. APPLY RDD SURGERY ---
    # Parse the mask string "0,0,1..." into a list of integers
    mask_list = [int(x) for x in mask.split(',')]
    
    # Configure Encoder (Block Skipping)
    for i, skip_flag in enumerate(mask_list):
        if i < len(encoder_wrappers):
            encoder_wrappers[i].skip = bool(skip_flag)

    # Configure Decoder (Channel Pruning)
    model.custom_decoder.linear_fuse.active_in_channels = channels

    # --- 2. PREPROCESS IMAGE ---
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt").to(device)

    # --- 3. INFERENCE ---
    with torch.no_grad():
        logits = model(**inputs)
        # Upsample to original size
        logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).cpu().numpy()[0]

    # --- 4. COLORIZE OUTPUT ---
    # 0: Crop (Blue), 1: Weed (Green), 2: Soil (Brown)
    colors = np.array([
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [139, 69, 19]   # Brown
    ], dtype=np.uint8)
    
    pred_img_array = colors[pred % 3]
    result_image = Image.fromarray(pred_img_array)

    # --- 5. RETURN IMAGE ---
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
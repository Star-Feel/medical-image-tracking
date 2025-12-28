import torch
import numpy as np
import pycuda.driver as cuda
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from time import time

from .point_generator import gen_auto_queries


def warmup(
    args,
    track_model,
    depth_model,
    infer_params,
    gs2_params,
):
    with torch.no_grad():
        print("[Info] warming up automatic query generation...")
        warmup_image = Image.open('./video/warmup_frame.jpg').convert('RGB')
        warmup_np = np.array(warmup_image).astype(np.uint8)
        for _ in tqdm(range(20)):
            queries = gen_auto_queries(args.input_size, 'box.', warmup_np, args.device, 5.0, *gs2_params)
        print("[Info] automatic query generation warmup done.")
        
        print("[Info] warming up model...")
        warmup_np = warmup_np[None, ...]
        
        warmup_thwc = torch.from_numpy(warmup_np)
        warmup_tchw = warmup_thwc.permute(0, 3, 1, 2).to(dtype=torch.float32, device=args.device)
        
        resized_tchw = F.interpolate(warmup_tchw, (args.input_size, args.input_size), mode='bilinear', align_corners=False)

        track_infer_list = []
        for i in tqdm(range(50)):
            start = time()
            if args.track_model == 'litetracker':
                track_model(
                    resized_tchw,
                    queries=queries,
                )
            end = time()
            track_infer_list.append(end -  start)
        print(f"[Info] track infer min: {min(track_infer_list)}, max: {max(track_infer_list)}, mean: {sum(track_infer_list) / len(track_infer_list)}")
        print("[Info] model warmup done.")


def load_model(track_model_name, device):
    
    if track_model_name == 'cotracker':
        track_model = load_cotracker_model('./checkpoints/cotracker/scaled_online.pth', device=device)
    elif track_model_name == 'trackon':
        track_model = load_track_on_model(
            './checkpoints/track_on/trackon2_dinov2_checkpoint.pt', 
            './checkpoints/track_on/test_dinov2.yaml', 
            device=device,
        )
    elif track_model_name == 'litetracker':
        track_model = load_litetracker_model('./checkpoints/cotracker/scaled_online.pth', device=device)
    
    gs2_params = None
    # gs2_params = load_grounded_sam_2_model(device)
    
    return track_model, gs2_params


def load_track_on_model(model_path, model_config, device):
    from third_party.track_on import Predictor, load_args_from_yaml
    model_args = load_args_from_yaml(model_config)
    model = Predictor(model_args, checkpoint_path=model_path).to(device).eval()
    return model


def load_cotracker_model(model_path, device):
    from third_party.co_tracker_realtime.cotracker.predictor import CoTrackerOnlinePredictor
    model = CoTrackerOnlinePredictor(checkpoint=model_path).to(device).eval()
    return model


def load_litetracker_model(model_path, device):
    from src.lite_tracker import LiteTracker
    model = LiteTracker()
    with open(model_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


# def load_grounded_sam_2_model(device: str):
#     from third_party.Grounded_SAM_2.sam2.build_sam import build_sam2
#     from third_party.Grounded_SAM_2.sam2.sam2_image_predictor import SAM2ImagePredictor
#     from third_party.Grounded_SAM_2.grounding_dino.groundingdino.util.inference import load_model
#     import third_party.Grounded_SAM_2.grounding_dino.groundingdino.datasets.transforms as T
#     SAM2_CHECKPOINT = "third_party/Grounded_SAM_2/checkpoints/sam2.1_hiera_tiny.pt"
#     SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
#     GROUNDING_DINO_CONFIG = "third_party/Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     GROUNDING_DINO_CHECKPOINT = "third_party/Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    
#     # environment settings
#     # use bfloat16

#     # build SAM2 image predictor
#     sam2_checkpoint = SAM2_CHECKPOINT
#     model_cfg = SAM2_MODEL_CONFIG
#     sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
#     sam2_predictor = SAM2ImagePredictor(sam2_model)

#     # build grounding dino model
#     grounding_model = load_model(
#         model_config_path=GROUNDING_DINO_CONFIG, 
#         model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
#         device=device
#     )
#     # setup the input image and text prompt for SAM 2 and Grounding DINO
#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     return (transform, sam2_predictor, grounding_model)
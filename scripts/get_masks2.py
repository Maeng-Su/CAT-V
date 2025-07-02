#
# --- get_masks2.py ---
#

import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
from tqdm import tqdm
import sys

# 상위 폴더 경로 추가
sys.path.append("./")
from sam2.build_sam import build_sam2_video_predictor

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        line = f.read()
    x_min, y_min, x_max, y_max = map(int, line.strip().split(","))
    # SAMurai는 프레임 ID 0에 대한 프롬프트만 필요로 함
    return {0: ((x_min, y_min, x_max, y_max), 0)}

def determine_model_cfg(model_path):
    if "large" in model_path: return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in model_path: return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in model_path: return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in model_path: return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError("Unknown model size in path!")

def run_mask_generation(model_path, video_path, txt_path, video_output_path, device, save_to_video=True):
    """
    Mask generation 모델을 로드, 실행하고 메모리에서 해제합니다.
    """
    predictor = state = None
    try:
        print(f"  [Mask] 모델 로드 중: {model_path}")
        model_cfg = determine_model_cfg(model_path)
        predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
        
        prompts = load_txt(txt_path)
        print(f"  [Mask] 프롬프트 로드 완료: {prompts}")

        cap = cv2.VideoCapture(video_path)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            loaded_frames.append(frame)
        cap.release()
        if not loaded_frames: raise ValueError("비디오에서 프레임을 로드할 수 없습니다.")
        height, width = loaded_frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_output_path는 디렉터리가 아닌 전체 파일 경로여야 함
        out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))
        
        print(f"  [Mask] 추론 실행 중...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            bbox, _ = prompts[0]
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
            
            color = (255, 0, 0) # 단일 객체 색상
            for frame_idx, object_ids, masks in tqdm(predictor.propagate_in_video(state)):
                img = loaded_frames[frame_idx]
                for obj_id, mask in zip(object_ids, masks):
                    mask_np = mask[0].cpu().numpy() > 0.0
                    mask_img = np.zeros_like(img, dtype=np.uint8)
                    mask_img[mask_np] = color
                    img = cv2.addWeighted(img, 1, mask_img, 0.4, 0) # 투명도 조절
                out.write(img)
        
        out.release()
        print(f"  [Mask] 마스크 비디오 저장 완료: {video_output_path}")

    except Exception as e:
        print(f"Mask generation 중 오류 발생: {e}")
    finally:
        if predictor is not None:
            del predictor
        if state is not None:
            del state
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [Mask] 모델 메모리 해제 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--txt_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_output_path", required=True) # 전체 파일 경로
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_mask_generation(args.model_path, args.video_path, args.txt_path, args.video_output_path, args.device)

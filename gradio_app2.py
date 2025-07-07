#
# --- gradio_app2.py (수정 완료 버전) ---
#
# 메모리 관리 및 누락된 인자를 모두 수정한 최종 Gradio 애플리케이션
#

import os
from pathlib import Path
import gradio as gr
import json
import cv2
import numpy as np
import torch
import gc

# ==============================================================================
# 중요: 이 스크립트가 정상적으로 작동하려면, 이전 답변에서 제공한
#      'get_boundary2.py', 'get_masks2.py', 'get_caption2.py', 'get_vis2.py'
#      파일들이 'scripts' 폴더 안에 올바르게 위치해야 합니다.
# ==============================================================================
from scripts.get_boundary2 import run_boundary_detection
from scripts.get_masks2 import run_mask_generation
from scripts.get_caption2 import run_captioning
from scripts.get_vis2 import run_visualization

CONFIG = {
    "get_boundary_model_path": "Yongxin-Guo/trace-uni",
    "get_mask_model_path": "./checkpoints/sam2.1_hiera_tiny.pt",
    "get_caption_model_path": "OpenGVLab/InternVL2-8B",
    "output_folder": "./results/",
    "frame_count": 16,
}

def extract_first_frame(video_path):
    # (이 함수는 변경사항 없음)
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    if not ret:
        cap.release()
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    height, width = image.shape[:2]
    if height > 750:
        scale = 750 / height
        new_width = int(width * scale)
        new_height = 750
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def run_inference_pipeline(video_path, bbox):
    """
    메모리 관리 및 모든 인자를 포함하여 전체 추론 파이프라인을 실행합니다.
    """
    os.makedirs(CONFIG["output_folder"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=============================================")
    print(f"파이프라인 시작. 사용 디바이스: {device}")
    print(f"=============================================")

    # --- 파일 경로 준비 ---
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    
    object_bbox_path = Path(CONFIG['output_folder']) / f"{base_name}_bbox.txt"
    boundary_json_path = Path(CONFIG['output_folder']) / f"{base_name}_boundary.json"
    masked_video_path = Path(CONFIG['output_folder']) / f"{base_name}_mask.mp4"
    final_json_path = Path(CONFIG['output_folder']) / f"{base_name}_boundary_caption.json"
    final_video_path = Path(CONFIG['output_folder']) / f"{base_name}_boundary_caption.mp4"

    # Bbox 좌표 저장
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return None
    h, w = frame.shape[:2]
    cap.release()
    bbox_coords = [int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)]
    with open(object_bbox_path, "w") as f:
        f.write(','.join(map(str, bbox_coords)))
    
    try:
        # --- Step 1: Boundary Detection (수정됨) ---
        print("\n[Step 1/4] 경계 감지 실행 중...")
        question_text = '''You are a video analysis AI. Your sole task is to find all worker actions in the provided video and output the results as a valid JSON array.
        You MUST NOT include any text other than the JSON output (no explanations, no greetings).

        # Action List:
        - Consult
        - Picking
        - Assen (Assemble)
        - Take m (Take material)
        - Take sc (Take screw)
        - Put do (Put down)
        - Other

        # JSON Object Schema:
        {
        "action": "Action Label",
        "start_time": start_time_in_seconds,
        "end_time": end_time_in_seconds
        }

        # Instruction:
        Analyze the provided video and output all action events as a JSON array.'''
        run_boundary_detection(
            model_path=CONFIG['get_boundary_model_path'],
            video_path=video_path,
            question=question_text, # <-- 누락되었던 '질문' 인자 추가
            output_path=boundary_json_path,
            device=device
        )
        print("[Step 1/4] 경계 감지 완료.")

        # --- Step 2: Mask Generation ---
        print("\n[Step 2/4] 마스크 생성 실행 중...")
        run_mask_generation(
            model_path=CONFIG['get_mask_model_path'],
            video_path=video_path,
            txt_path=object_bbox_path,
            video_output_path=masked_video_path, # 인자 이름 명확화
            device=device
        )
        print("[Step 2/4] 마스크 생성 완료.")

        # --- Step 3: Captioning (수정됨) ---
        print("\n[Step 3/4] 캡션 생성 실행 중...")
        run_captioning(
            model_path=CONFIG['get_caption_model_path'],
            QA_file_path=boundary_json_path,
            video_folder=CONFIG['output_folder'],
            answers_output_folder=CONFIG['output_folder'], # <-- 누락되었던 인자 추가
            final_json_path=final_json_path, # <-- 누락되었던 인자 추가
            extract_frames_method='max_frames_num', # <-- 누락되었던 인자 추가
            max_frames_num=CONFIG['frame_count'], # <-- 누락되었던 인자 추가
            frames_from='video', # <-- 누락되었던 인자 추가
            provide_boundaries=True, # <-- 누락되었던 인자 추가
            device=device
        )
        print("[Step 3/4] 캡션 생성 완료.")

        # --- Step 4: Visualization ---
        print("\n[Step 4/4] 시각화 자료 생성 중...")
        run_visualization(
            video_input_path=masked_video_path, 
            json_path=final_json_path, 
            video_output_path=final_video_path
        )
        print("[Step 4/4] 시각화 완료.")

    except Exception as e:
        print(f"\n파이프라인 중간에 오류가 발생했습니다: {e}")
        return None

    # 최종 결과 반환
    try:
        with open(final_json_path, "r") as f:
            results = json.load(f)
        return {"captions": results, "final_video": final_video_path}
    except Exception as e:
        print(f"최종 결과 파일을 읽는 중 오류 발생: {e}")
        return None

def get_bounding_box(image):
    # (이 함수는 변경사항 없음)
    alpha_channel = image[:, :, 3]
    y_coords, x_coords = np.where(alpha_channel > 0)
    if y_coords.size == 0 or x_coords.size == 0:
        return None
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min_ratio = x_min / image.shape[1]
    x_max_ratio = x_max / image.shape[1]
    y_min_ratio = y_min / image.shape[0]
    y_max_ratio = y_max / image.shape[0]
    return x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio

def caption_video(video, edited_image):
    # (이 함수는 변경사항 없음)
    if video is None:
        return "먼저 비디오를 업로드해주세요.", None
    if edited_image is None or not edited_image.get('layers'):
        return "이미지에 Bounding Box를 그려주세요.", None
        
    layer_0 = edited_image['layers'][0]
    bbox = get_bounding_box(layer_0)
    if bbox is None:
        return "이미지에 Bounding Box를 그려주세요.", None

    results = run_inference_pipeline(video, bbox)
    if results is None:
        return "처리 중 오류가 발생했습니다. 터미널 로그를 확인해주세요.", None

    captions_text = "\n\n".join(
        [
            f"Event {i+1} (Time: {event.get('timestamp', 'N/A')}):\n{event.get('model_answer', 'No caption')}"
            for i, event in enumerate(results.get("captions", []))
        ]
    )
    return captions_text, results.get("final_video")

def create_demo():
    # (이 함수는 변경사항 없음)
    DESCRIPTION = """# CAT2:
    This is a demo for our 'CAT2' [paper](https://github.com/yunlong10/CAT-2). Code is available [here](https://github.com/yunlong10/CAT-2).
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Caption Anything Demo v2 (Memory Optimized)")
        gr.Markdown(DESCRIPTION)
        gr.Markdown("Upload a video and draw a rectangle on the first frame to provide a bounding box.")
        
        with gr.Row():
            video_input = gr.Video(label="Upload Video", height=800)
            first_frame_editor = gr.ImageEditor(label="Draw a rectangle on the First Frame", height=800, type="numpy")
        
        video_input.change(fn=extract_first_frame, inputs=video_input, outputs=first_frame_editor)
        
        caption_button = gr.Button("Generate Captions")
        
        output_text = gr.Textbox(label="Video Captions", lines=10)
        output_video = gr.Video(label="Processed Video")
        
        caption_button.click(
            fn=caption_video,
            inputs=[video_input, first_frame_editor],
            outputs=[output_text, output_video],
        )
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8889,
        debug=True,
    )

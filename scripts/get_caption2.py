import os
import sys
import gc
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import argparse
import traceback

# --- 이 파일이 독립적으로 작동하는 데 필요한 모든 헬퍼 함수를 여기에 직접 포함 ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = float(bound[0]), float(bound[1])
    else:
        start, end = 0, max_frame / fps
    start_idx = max(first_idx, int(start * fps))
    end_idx = min(int(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        # .asnumpy()는 decord의 VideoReader 프레임 객체(vr[...])에 사용하는 것이 맞습니다.
        img = Image.fromarray(vr[frame_index].cpu().numpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# --- 메인 함수 ---
def run_captioning(model_path, QA_file_path, video_folder, answers_output_folder, final_json_path, device, max_frames_num, extract_frames_method, frames_from, provide_boundaries):
    model = tokenizer = None
    try:
        print(f"  [Caption] 모델 로드 중: {model_path}")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        print(f"  [Caption] 경계 파일 로드 중: {QA_file_path}")
        with open(QA_file_path, 'r', encoding='utf-8') as f:
            events_data = json.load(f)

        answers = []
        if not events_data:
            print("  [Caption] 경계 파일에 처리할 이벤트가 없습니다.")
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4, ensure_ascii=False)
            return

        # 비디오는 한 번만 로드합니다.
        video_name = events_data[0]['video']
        video_path = os.path.join(video_folder, video_name)
        print(f"  [Caption] 비디오 로딩 중: {video_path}")
        pixel_values, num_patches_list = load_video(video_path, num_segments=max_frames_num, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(next(model.parameters()).device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        
        print(f"  [Caption] 각 이벤트에 대해 개별 추론 실행 중...")
        # for 루프를 돌면서 각 이벤트에 대해 개별적으로 질문합니다.
        for event in tqdm(events_data):
            # 각 이벤트에 맞는 구체적인 질문 생성
            event_desc = f"From {event['segment'].replace('_', ' to ')}s, {event['answer']}"
            question_tmp = video_prefix + event_desc + "\nDescribe this specific event in detail."

            # 모델을 각 이벤트마다 호출
            with torch.inference_mode():
                response = model.chat(tokenizer, pixel_values, question_tmp, generation_config)
            
            # 각 이벤트의 결과를 answers 리스트에 추가
            segment_str = event['segment'].replace('_', 's - ') + 's'
            answers.append({
                "video": event['video'],
                "segment": event['segment'].split('_'),
                "timestamp": segment_str,
                "question": event['question'],
                "short_answer": event['answer'],
                "model_answer": response
            })

        print(f"  [Caption] 최종 결과 저장 중: {final_json_path}")
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"Captioning 중 오류 발생: {e}")
        traceback.print_exc()
    finally:
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [Caption] 모델 메모리 해제 완료.")


# 이 스크립트를 단독으로 실행할 때를 위한 부분
if __name__ == "__main__":
    # basic_parser를 직접 정의하거나, eval_utils에서 가져와야 함
    # 여기서는 간단한 ArgumentParser로 대체
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--QA_file_path", required=True)
    parser.add_argument("--video_folder", default='./results')
    parser.add_argument("--answers_output_folder", default='./results')
    parser.add_argument("--final_json_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames_num", type=int, default=16)
    parser.add_argument("--extract_frames_method", default="max_frames_num")
    parser.add_argument("--frames_from", default="video")
    parser.add_argument("--provide_boundaries", action='store_true')
    args = parser.parse_args()

    run_captioning(
        model_path=args.model_path,
        QA_file_path=args.QA_file_path,
        video_folder=args.video_folder,
        answers_output_folder=args.answers_output_folder,
        final_json_path=args.final_json_path,
        device=args.device,
        max_frames_num=args.max_frames_num,
        extract_frames_method=args.extract_frames_method,
        frames_from=args.frames_from,
        provide_boundaries=args.provide_boundaries
    )
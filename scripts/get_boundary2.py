#
# --- get_boundary2.py (완성된 버전) ---
#

import torch
import gc
import transformers
import json
import sys
import os
import argparse

# 상위 폴더를 경로에 추가하여 trace 모듈을 찾을 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trace.conversation import conv_templates, SeparatorStyle
from trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from trace.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token_all, process_video, KeywordsStoppingCriteria
from trace.model.builder import load_pretrained_model

def run_boundary_detection(model_path, video_path, question, output_path, device):
    """
    Boundary detection 모델을 로드, 실행하고 메모리에서 해제합니다.
    """
    model = tokenizer = processor = None # 변수 초기화
    try:
        print(f"  [Boundary] 모델 로드 중: {model_path}")
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
        model.eval()

        print(f"  [Boundary] 비디오 처리 및 프롬프트 생성 중...")
        tensor, video_timestamps = process_video(video_path, processor, model.config.image_aspect_ratio, num_frames=64)
        tensor = tensor.to(dtype=torch.float16, device=device, non_blocking=True)

        question_prompt = DEFAULT_MMODAL_TOKEN["VIDEO"] + "\n" + question
        conv = conv_templates['llama_2'].copy()
        conv.append_message(conv.roles[0], question_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + '<sync>'
        
        input_ids = tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)
        
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        print(f"  [Boundary] 추론 실행 중...")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images_or_videos=[tensor],
                modal_list=['video'],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                video_timestamps=[video_timestamps],
                heads=[1]
            )

        # --- [추가된 부분 시작] ---
        # 모델 출력(output_ids)을 파싱하여 의미있는 데이터로 변환
        outputs = {
            'timestamps': [],
            'scores': [],
            'captions': [],
        }
        cur_timestamps = []
        cur_timestamp = []
        cur_scores = []
        cur_score = []
        cur_caption = []
        for idx in output_ids[0]:
            if idx <= 32000:
                if idx == 32000:
                    new_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                    outputs['captions'].append(new_caption)
                    cur_caption = []
                else:
                    cur_caption.append(idx)
            elif idx <= 32013:
                if idx == 32001:
                    if len(cur_timestamp) > 0:
                        cur_timestamps.append(float(''.join(cur_timestamp)))
                    outputs['timestamps'].append(cur_timestamps)
                    cur_timestamps = []
                    cur_timestamp = []
                elif idx == 32002:
                    if len(cur_timestamp) > 0:
                        cur_timestamps.append(float(''.join(cur_timestamp)))
                    cur_timestamp = []
                else:
                    cur_timestamp.append(model.get_model().time_tokenizer.decode(idx - 32001))
            else:
                if idx == 32014:
                    if len(cur_score) > 0:
                        cur_scores.append(float(''.join(cur_score)))
                    outputs['scores'].append(cur_scores)
                    cur_scores = []
                    cur_score = []
                elif idx == 32015:
                    if len(cur_score) > 0:
                        cur_scores.append(float(''.join(cur_score)))
                    cur_score = []
                else:
                    cur_score.append(model.get_model().score_tokenizer.decode(idx - 32014))
        if len(cur_caption):
            outputs['captions'].append(tokenizer.decode(cur_caption, skip_special_tokens=True).strip())
        
        # 파싱된 결과를 JSON 형식에 맞게 정리
        results = []
        for i in range(len(outputs['timestamps'])):
            # 타임스탬프가 비어있거나, 캡션이 없는 경우는 제외
            if not outputs['timestamps'][i] or i >= len(outputs['captions']):
                continue
            
            output_item = {
                'video': os.path.basename(video_path).replace(os.path.splitext(video_path)[1], "") + "_mask.mp4",
                'segment': f"{outputs['timestamps'][i][0]}_{outputs['timestamps'][i][1]}",
                'question': question,
                'answer': outputs['captions'][i],
            }
            results.append(output_item)
        
        print(f"  [Boundary] 결과 저장 중: {output_path}")
        # 최종 결과를 JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        # --- [추가된 부분 끝] ---

    except Exception as e:
        print(f"Boundary detection 중 오류 발생: {e}")
        # 오류 발생 시에도 빈 파일 생성하여 파이프라인이 멈추지 않도록 함
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:
                json.dump([], f)
    finally:
        if model is not None:
            del model
            del tokenizer
            del processor
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  [Boundary] 모델 메모리 해제 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for boundary detection.")
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    parser.add_argument("--question", required=True, help="Question for video inference.")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model.")
    parser.add_argument("--output_path", required=True, help="Path to save the output json file.")
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    args = parser.parse_args()
    
    run_boundary_detection(args.model_path, args.video_path, args.question, args.output_path, args.device)
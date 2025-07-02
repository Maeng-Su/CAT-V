#
# --- get_vis2.py ---
#

import cv2
import json
import sys
from tqdm import tqdm

# ... 기존 get_vis.py의 헬퍼 함수들 (is_frame_in_segment, wrap_text 등)을 여기에 그대로 붙여넣으세요 ...

def run_visualization(video_input_path, json_path, video_output_path):
    """
    주어진 JSON 데이터의 캡션을 비디오에 추가합니다.
    """
    print(f"  [Vis] 캡션 시각화 시작: {video_input_path}")
    try:
        # add_captions_to_video 함수의 내용을 여기에 그대로 붙여넣습니다.
        # ... (생략) ...
        print(f"  [Vis] 캡션 비디오 저장 완료: {video_output_path}")
    except Exception as e:
        print(f"Visualization 중 오류 발생: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python get_vis2.py <input_video_path> <json_path> <output_video_path>")
        sys.exit(1)
    
    video_input_path = sys.argv[1]
    json_path = sys.argv[2]
    video_output_path = sys.argv[3]

    run_visualization(video_input_path, json_path, video_output_path)

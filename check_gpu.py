import torch

def check_gpu_status():
    """
    PyTorch를 사용하여 GPU 상태를 확인하고 간단한 텐서 연산을 테스트합니다.
    """
    print("=============================================")
    print("      PyTorch GPU 상태 확인 스크립트")
    print("=============================================")

    # 1. CUDA 사용 가능 여부 확인
    is_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능 여부: {is_available}")

    if not is_available:
        print("\n[알림] PyTorch가 GPU(CUDA)를 찾을 수 없습니다.")
        print("CUDA 드라이버나 PyTorch가 올바르게 설치되었는지 확인하세요.")
        print("=============================================")
        return

    # 2. 사용 가능한 GPU 개수 확인
    device_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {device_count}")

    # 3. 현재 GPU 정보 확인
    current_device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device_id)
    print(f"현재 사용 중인 GPU ID: {current_device_id}")
    print(f"현재 GPU 이름: {gpu_name}")

    # 4. GPU 메모리 정보 확인 (전체/할당된/캐시)
    total_mem = torch.cuda.get_device_properties(current_device_id).total_memory / (1024**3)
    allocated_mem = torch.cuda.memory_allocated(current_device_id) / (1024**3)
    cached_mem = torch.cuda.memory_reserved(current_device_id) / (1024**3)
    print(f"전체 VRAM: {total_mem:.2f} GB")
    print(f"현재 할당된 VRAM: {allocated_mem:.2f} GB")
    print(f"현재 캐시된 VRAM: {cached_mem:.2f} GB")

    # 5. 간단한 텐서 연산을 통해 GPU가 실제로 동작하는지 테스트
    try:
        print("\n[테스트] 간단한 텐서를 GPU로 보내는 중...")
        # 'cuda' 디바이스를 명시적으로 지정
        device = torch.device("cuda")
        
        # CPU에 텐서 생성
        cpu_tensor = torch.randn(3, 3)
        print(f"  - CPU 텐서: {cpu_tensor.device}")
        
        # GPU로 텐서 이동
        gpu_tensor = cpu_tensor.to(device)
        print(f"  - GPU로 이동된 텐서: {gpu_tensor.device}")

        # GPU에서 간단한 연산 수행
        result_tensor = gpu_tensor * gpu_tensor
        print("  - GPU에서 행렬 곱셈 연산 성공!")
        print("\n[결론] PyTorch가 GPU를 성공적으로 인식하고 사용하고 있습니다.")

    except Exception as e:
        print("\n[오류] GPU 테스트 중 문제가 발생했습니다.")
        print(e)
        
    print("=============================================")


if __name__ == "__main__":
    check_gpu_status()

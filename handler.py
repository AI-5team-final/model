import os
import runpod
from LLMManager import NLLBManager

# 환경 변수에서 설정값 로드
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "Youseff1987/nllb-200-finetuning-20250305")
CPU_ONLY = os.getenv("CPU_ONLY", "False").lower() == "true"
print("HF_TOKEN:", HF_TOKEN)
print("MODEL_ID:", MODEL_ID)
print("CPU_ONLY:", CPU_ONLY)

# NLLBManager 인스턴스 초기화
manager = NLLBManager(model_id=MODEL_ID, hf_token=HF_TOKEN, cpu_only=CPU_ONLY)

def handler(job):
    """
    RunPod 서버리스 핸들러 함수.
    입력 데이터를 받아 번역 결과를 반환합니다.

    Parameters:
    job (dict): 실행할 작업의 정보가 담긴 딕셔너리.

    Returns:
    dict: 번역 결과 또는 에러 메시지를 포함한 딕셔너리.
    """
    try:
        input_data = job["input"]
        text = input_data["text"]
        lang_code = input_data.get("lang_code", "eng_Latn")

        # 번역 실행
        translation = manager.invoke(text=text, lang_code=lang_code)
        return {"translation": translation}

    except Exception as e:
        return {"error": str(e)}

# 서버리스 핸들러 시작
runpod.serverless.start({"handler": handler})
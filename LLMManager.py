import json
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login


# 번역용 Class인데 LLM용 클래스로 변경해서 사용할 것
class NLLBManager:
    def __init__(self, model_id: str, hf_token: str, cpu_only: bool = False):
        self.model_id = model_id
        self.hf_token = hf_token
        self.cpu_only = cpu_only
        self.device = torch.device("cpu" if cpu_only or not torch.cuda.is_available() else "cuda")

        try:
            print("[INFO] Logging into HuggingFace...")
            if self.hf_token:
                login(self.hf_token)
        except Exception as e:
            print(f"[ERROR] HuggingFace login failed: {e}")

        try:
            self.initialize()
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            raise e  # raise 해서 container가 죽더라도 로그는 남음

    def initialize(self):
        print("[INFO] Loading language code map...")
        with open("./bcp47_to_flores.json", "r", encoding="utf-8") as f:
            self.bcp47_to_flores = json.load(f)

        print(f"[INFO] Loading model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.device)
        print("[INFO] Model and tokenizer loaded.")

    def get_language_code(self, user_input_code: str) -> str:
        key = user_input_code.lower()
        if key in ["zh-hans", "zh-hant", "zh-cn", "zh-tw"]:
            return self.bcp47_to_flores.get(key, "eng_Latn")
        if "-" in key:
            parts = key.split("-")
            return self.bcp47_to_flores.get(parts[0], "eng_Latn")
        return self.bcp47_to_flores.get(key, "eng_Latn")

    def invoke(self, text: str, lang_code: str = "eng_Latn") -> str:
        try:
            print(f"[INFO] Invoking translation: '{text}' → {lang_code}")
            nllb_code = self.get_language_code(lang_code)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(nllb_code)
            )

            result = self.tokenizer.decode(output[0], skip_special_tokens=True)

						# 메모리 삭제
            del inputs
            del output
            
            # cuda vram memory 정리
            if not self.cpu_only:
                torch.cuda.empty_cache()

            return result
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            raise
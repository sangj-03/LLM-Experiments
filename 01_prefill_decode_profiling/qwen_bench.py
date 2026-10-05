import ollama
import json
import time
import random
import platform
import psutil

# 1. 설정 및 메타데이터 준비
MODEL = "qwen2.5vl:7b" 
ITERATIONS = 100 

# 시스템 환경 정보 (데이터 엔지니어링: 환경 변수 기록)
SYSTEM_INFO = {
    "os": platform.system(),
    "cpu": platform.processor(),
    "gpu": "NVIDIA RTX 4080 (16GB)",
    "driver_version": "535.261.03",
    "python_version": platform.python_version()
}

# 시나리오별 프롬프트 설정
LONG_IN_PROMPT = """
오늘 날씨는 어떤지 설명해주는데 다음과 같이 설명해줘. 
먼저, 오늘 기온은 어느 정도이고 객관적으로 추운건지 더운건지를 알려줘. 
그리고 날씨에 맞춰서 옷을 어떻게 입으면 좋을 지도 추가로 알려줘. 
오늘 강수 확률은 어느 정도이고, 강수가 온다면 눈인지 비인지도 알려줘. 
현재 기압은 어느정도이고 고기압인지 저기압인지도 알려주고 
1시간 단위로 기온이 어느정도 되는 지도 알려줘.
""" * 30

SHORT_IN_PROMPT = "오늘 날씨 어때?"

scenarios = [
    {"name": "Short_In / Short_Out", "prompt": SHORT_IN_PROMPT + " 한 문장으로 짧게 답해줘.", "max_tokens": 150},
    {"name": "Short_In / Long_Out", "prompt": SHORT_IN_PROMPT + " 300자 이상 아주 상세하게 답해줘.", "max_tokens": 1024},
    {"name": "Long_In / Short_Out", "prompt": LONG_IN_PROMPT + " 위 내용을 요약해서 한 문장으로 답해줘.", "max_tokens": 150},
    {"name": "Long_In / Long_Out", "prompt": LONG_IN_PROMPT + " 위 내용을 바탕으로 300자 이상 상세히 답해줘.", "max_tokens": 1024},
]

# 2. 자동 채점기 함수 (품질 관리 로직)
def auto_grade(text, scenario_name):
    score = 0
    char_count = len(text)
    
    # 규칙 1: 분량 체크 (50점)
    if "Long_Out" in scenario_name:
        is_valid_len = char_count >= 300
    else:
        is_valid_len = char_count <= 200 # Short_Out은 너무 길지 않아야 함
    
    if is_valid_len: score += 50
    
    # 규칙 2: 키워드 포함 체크 (50점)
    keywords = ["기온", "날씨", "옷", "기압", "도"]
    found = [kw for kw in keywords if kw in text]
    keyword_score = (len(found) / len(keywords)) * 50
    score += int(keyword_score)
    
    return {
        "final_score": score,
        "char_length": char_count,
        "keywords_found": found,
        "pass_fail": "PASS" if score >= 70 else "FAIL"
    }

# 3. 메인 실행 루프
raw_data = []
print(f"🔥 모델 웜업 및 벤치마크 시작 (모델: {MODEL})...")
ollama.generate(model=MODEL, prompt="warmup", options={"num_predict": 1})



for i in range(ITERATIONS):
    current_round = scenarios.copy()
    random.shuffle(current_round) # 순서 섞기 (편향 제거)
    
    print(f"\n[반복 {i+1}/{ITERATIONS}]")
    
    for sc in current_round:
        try:
            time.sleep(0.5) # GPU 쿨다운
            
            start_time = time.time()
            response = ollama.generate(
                model=MODEL,
                prompt=sc["prompt"],
                options={
                    "num_predict": sc["max_tokens"],
                    "temperature": 0.7,
                    "num_ctx": 8192
                }
            )
            
            # 응답 데이터 가공
            answer_text = response['response']
            prefill_time = response['prompt_eval_duration'] / 1e9
            decode_time = response['eval_duration'] / 1e9
            total_tokens = response['eval_count']
            
            # 자동 채점 수행
            grade_results = auto_grade(answer_text, sc["name"])
            
            # 데이터 엔지니어링: 통합 레코드 생성
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "iteration": i + 1,
                "scenario": sc["name"],
                "metrics": {
                    "prefill_s": round(prefill_time, 4),
                    "decode_s": round(decode_time, 4),
                    "tps": round(total_tokens / decode_time, 2) if decode_time > 0 else 0,
                    "total_tokens": total_tokens
                },
                "quality": grade_results, # 채점 결과 통합
                "system": SYSTEM_INFO,     # 메타데이터 통합
                "response": answer_text
            }
            
            raw_data.append(record)
            print(f"  ✅ {sc['name']} 완료 | 총 토큰: {record['metrics']['total_tokens']} | Score: {grade_results['final_score']} | TPS: {record['metrics']['tps']}")
            
        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")

# 4. 결과 저장
filename = "qwen_benchmark_refined.json"
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=4)

print(f"\n✨ 벤치마크 완료! 데이터가 {filename}에 저장되었습니다.")

import ollama
import time
import pandas as pd
import tiktoken
from pathlib import Path

def get_llama3_token_count(text):
    """
    Llama 3의 토큰 수를 계산하기 위한 함수입니다.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base") 
        return len(encoding.encode(text))
    except:
        return len(text)

def query_llama(prompt, temp):
    """
    Top-p를 제외하고 오직 온도(Temperature)만 조절하여 호출합니다.
    (Top-p는 기본값인 1.0으로 작동하게 됩니다.)
    """
    response = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
        options={
            'temperature': temp,
            'num_predict': 1024,
            'repeat_penalty': 1.2,
            # 'top_p': 1.0  # 명시적으로 1.0을 주거나 아예 삭제하면 필터링이 풀립니다.
        }
    )
    return response['response']

# 실험 변수 설정: 온도만 변경
temperatures = [0.1, 0.6, 1.0, 1.8] 
iterations = 300

# The puzzle prompt is kept outside the repository (third-party quiz text).
# Put the full prompt in prompts/detective_case.txt before running.
test_prompt = (Path(__file__).parent / "prompts" / "detective_case.txt").read_text(encoding="utf-8")

results = []

try:
    print("Temperature 변수 집중 실험을 시작합니다. (Top-p 필터링 해제)")
    for t in temperatures:
        print(f"\n--- 현재 설정: Temperature {t} 테스트 시작 ---")
        for i in range(1, iterations + 1):
            try:
                start_time = time.time()
                res = query_llama(test_prompt, t)
                duration = time.time() - start_time
                token_count = get_llama3_token_count(res)
                
                results.append({
                    "temperature": t,
                    "iteration": i,
                    "response": res.strip().replace('\n', ' '),
                    "duration": duration,
                    "token_count": token_count,
                    "status": "success"
                })
                print(f"  [{t}도] {i}/{iterations} | {duration:.2f}s | {token_count}tk")
            except Exception as e:
                print(f"  [{t}도] {i}/{iterations} 오류: {e}")
                results.append({
                    "temperature": t, "iteration": i, "response": "ERR", 
                    "duration": 0, "token_count": 0, "status": "error"
                })
                time.sleep(1)
finally:
    df = pd.DataFrame(results)
    df.to_csv('test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n[저장 완료] 총 {len(df)}개의 실험 결과가 test_results.csv에 저장되었습니다.")

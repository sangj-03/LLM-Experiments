import ollama
import pandas as pd
import re
import json

def evaluate_and_append(row):
    """
    LLM의 답변을 분석하여 100점 만점 기준 점수만 반환합니다.
    이유(reason) 생성은 제외되었습니다.
    """
    response_text = str(row['response'])
    
    if "ERR" in response_text or len(response_text) < 10:
        return 0, 0, 0, False, 0

    judge_prompt = f"""
    당신은 전문 수사 검증관입니다. 아래 [학생 답변]을 엄격하게 채점하세요.

    [배점 기준]
    1. 범인 특정 (40점): '4번 임상수' 지목 여부
    2. 핵심 논리 (40점): '노랑머리 = 이름 모르는 외부인' 설명 여부
    3. 형식 (20점): 10문장 이내 및 가독성

    [감점]
    - 환각(Hallucination) 발생 시 최종 점수에서 **30점 차감** (예: 본문에 없는 CCTV, 지문 등)

    [학생 답변]
    {response_text}

    반드시 아래 JSON 형식으로만 답변하세요:
    {{
        "criminal": 40,
        "logic": 40,
        "format": 20,
        "hallucination": true
    }}
    """
    
    try:
        # JSON 모드로 결과 생성
        res = ollama.generate(model='llama3.1:8b', prompt=judge_prompt, 
                              format='json', options={'temperature': 0})
        data = json.loads(res['response'])
        
        # 점수 계산 및 환각 감점 적용
        base = data.get('criminal', 0) + data.get('logic', 0) + data.get('format', 0)
        penalty = 30 if data.get('hallucination', False) else 0
        final_score = max(0, base - penalty)
        
        return (data.get('criminal', 0), data.get('logic', 0), data.get('format', 0), 
                data.get('hallucination', False), final_score)
    except:
        return 0, 0, 0, False, 0

def start_scoring():
    print("데이터 로드 중: test_results.csv")
    df = pd.read_csv('test_results.csv')
    
    print("정밀 채점 시작 (이유 제외, 점수 중심)...")
    
    # 새로운 컬럼 생성을 위한 리스트 적용 (reason 컬럼 제외)
    results = df.apply(evaluate_and_append, axis=1, result_type='expand')
    results.columns = ['score_crim', 'score_logic', 'score_format', 'is_hallucination', 'final_score']
    
    # 기존 데이터프레임에 병합
    final_df = pd.concat([df, results], axis=1)
    
    # 결과 저장
    final_df.to_csv('final_scored_results.csv', index=False, encoding='utf-8-sig')
    print(f"채점 완료! 총 {len(final_df)}행의 데이터가 저장되었습니다.")

if __name__ == "__main__":
    start_scoring()

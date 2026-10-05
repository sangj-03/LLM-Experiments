import ollama
import json
import time
from tqdm import tqdm

# 1. 설정
DATA_FILE = "qwen_benchmark_refined.json"  # 분석할 원본 데이터
JUDGE_MODEL = "llama3.1:8b"                   # 채점용 고성능 모델
OUTPUT_FILE = "qwen_benchmark_judged.json"

def get_judge_prompt(scenario, response):
    """
    채점 모델에게 전달할 구체적인 채점 가이드(Rubric)를 정의합니다.
    """
    return f"""
    당신은 LLM 응답 품질을 평가하는 전문 평가관입니다.
    다음 시나리오에 따른 인공지능의 답변을 객관적으로 채점해주세요.

    [시나리오 유형]: {scenario}
    [인공지능의 답변]: 
    ---
    {response}
    ---

    [채점 기준]:
    1. 정보의 구체성: 날씨, 기온, 의복 추천 등 요구사항을 구체적으로 설명했는가? (40점)
    2. 논리적 일관성: 답변의 내용이 모순되지 않고 흐름이 자연스러운가? (30점)
    3. 가독성 및 형식: 문장 구조가 깔끔하고 읽기 편한가? (30점)

    [주의]: 다른 설명은 생략하고 오직 아래 JSON 형식으로만 답변하세요.
    {{
        "logic_score": 0,
        "detail_score": 0,
        "readability_score": 0,
        "final_score": 0,
        "reason": "한 문장으로 요약된 채점 이유"
    }}
    """

def run_judge():
    # 데이터 로드
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ {DATA_FILE} 파일을 찾을 수 없습니다. 먼저 벤치마크를 실행하세요.")
        return

    print(f"🚀 {JUDGE_MODEL} 모델을 사용하여 채점을 시작합니다... (총 {len(data)}개 항목)")
    
    judged_data = []

    for entry in tqdm(data):
        # 채점용 프롬프트 생성
        prompt = get_judge_prompt(entry['scenario'], entry['response'])
        
        try:
            # Llama 3.1에게 채점 요청
            response = ollama.generate(model=JUDGE_MODEL, prompt=prompt, format="json")
            judge_result = json.loads(response['response'])
            
            # 기존 데이터에 채점 결과 통합
            entry['quality_llm'] = judge_result
            # 기존 score를 LLM 점수로 대체 (비교 분석용)
            entry['quality']['final_score'] = judge_result['final_score']
            
            judged_data.append(entry)
            
        except Exception as e:
            print(f"⚠️ 채점 중 오류 발생: {e}")
            continue

    # 결과 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(judged_data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 채점 완료! 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    run_judge()

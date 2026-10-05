# LLM 실습

[English](README.md)

2026년 상반기에 개인 실습으로 진행한 로컬 LLM 실험 두 가지입니다. 두 실험 모두 [Ollama](https://ollama.com)로 띄운 모델을 사용했습니다.

| 실험 | 모델 | 질문 | 실행 횟수 |
| --- | --- | --- | --- |
| 01 Prefill·Decode 프로파일링 | Qwen2.5-VL 7B | 입력·출력 길이에 따라 지연시간, 처리량, 답변 품질이 어떻게 달라지는가? | 시나리오 4개 × 100회 |
| 02 샘플링 온도와 추론 | Llama 3.1 8B | 온도에 따라 추론 정확도, 응답 시간, 출력 길이가 어떻게 달라지는가? | 온도 4개 × 300회 = 1,200회 |

원본 결과 파일(JSON, CSV)은 포함하지 않았습니다. 각 `results/` 폴더의 그림은 시각화 스크립트의 출력입니다.

## 01 Prefill·Decode 프로파일링

`01_prefill_decode_profiling/`

**설정.** NVIDIA RTX 4080(16 GB)에서 Qwen2.5-VL 7B(`qwen2.5vl:7b`), temperature 0.7, 컨텍스트 길이 8,192를 사용했습니다. 짧은·긴 입력과 짧은·긴 출력을 교차해 시나리오 4개를 만들었습니다.

- 짧은 입력: 오늘 날씨를 묻는 한 줄 질문. 긴 입력: 여러 항목을 요구하는 날씨 질문을 30번 반복.
- 짧은 출력: 한 문장, 최대 150토큰. 긴 출력: 300자 이상, 최대 1,024토큰.

웜업 호출 뒤 100라운드 동안 매 라운드 네 시나리오를 무작위 순서로 실행했고, 호출 사이에 0.5초씩 쉬었습니다.

**지표.**

- Prefill·Decode 시간은 Ollama의 `prompt_eval_duration`, `eval_duration`에서 가져왔고, 초당 토큰 수는 생성 토큰 수를 decode 시간으로 나눈 값입니다.
- `qwen_bench.py`는 규칙으로 답변을 채점합니다. 길이 조건 50점, 키워드 포함 50점입니다.
- `judge_results.py`로 Llama 3.1 8B를 채점 모델로 써서 다시 채점할 수 있습니다(구체성 40, 일관성 30, 가독성 30).
- `plot_json.py`는 품질과 속도를 합친 품질 가중 효율 점수를 계산합니다: (점수 / 100)³ × 100 / (prefill + decode + 0.5초).

**결과** (시나리오별 평균, [`results/exponential_performance_dashboard.png`](01_prefill_decode_profiling/results/exponential_performance_dashboard.png)):

| 시나리오 | Prefill (초) | Decode (초) | 토큰/초 | 품질 점수 | 가중 효율 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 짧은 입력 / 짧은 출력 | 0.01 | 0.10 | 137.12 | 59.8 | 35.01 |
| 짧은 입력 / 긴 출력 | 0.01 | 1.96 | 128.69 | 54.1 | 10.06 |
| 긴 입력 / 짧은 출력 | 0.52 | 0.69 | 119.70 | 93.6 | 51.82 |
| 긴 입력 / 긴 출력 | 0.47 | 3.95 | 117.84 | 92.8 | 19.23 |

- 이 실험 조건에서는 출력 길이가 지연을 지배했습니다. 긴 답변의 decode는 최대 3.95초가 걸렸지만, 긴 프롬프트를 읽는 데는 약 0.5초면 됐습니다.
- 처리량은 118–137토큰/초의 좁은 범위에 있었고, 입력이 길 때 조금 낮았습니다.

**한계.** 질문 주제 하나, 모델 하나, GPU 하나로만 실험했습니다. 규칙 기반 품질 점수는 키워드와 길이만 세는 대략적인 지표이고, 모델에게 실시간 날씨 정보가 없어 사실 정확도는 측정하지 않았습니다.

## 02 샘플링 온도와 추론

`02_temperature_reasoning/`

**설정.** Llama 3.1 8B(`llama3.1:8b`)가 한국어 추리 퀴즈 하나를 풉니다. 용의자 7명 중 범인을 지목하고 핵심 단서를 8문장 이내로 설명해야 합니다. 온도 0.1, 0.6, 1.0, 1.8에서 각각 300번씩 풀게 했고, top-p는 기본값, repeat penalty 1.2, 최대 생성 토큰 1,024로 두었습니다. 매 실행마다 걸린 시간과 출력 길이(Llama용 근사로 tiktoken `cl100k_base` 사용)를 기록했습니다.

**채점.** `judge_answers.py`가 Llama 3.1 8B(온도 0, JSON 출력)에게 답변을 채점하게 합니다. 범인 지목 40점, 논리 40점, 형식 20점이며, 퀴즈에 없는 증거를 지어내면 30점을 뺍니다.

**그래프.** `plot_results.py`가 1,200회 전체에 걸친 점수·시간·토큰 수의 20회 이동 평균을 그립니다([`results/performance_matrix_result_1200.png`](02_temperature_reasoning/results/performance_matrix_result_1200.png)).

**결과** (그래프 기준): 채점 점수는 온도 0.6 부근에서 가장 높았고 1.8에서 가장 낮았으며, 1.8에서는 답변도 더 길고 느렸습니다.

**프롬프트.** 퀴즈 지문은 외부 자료라 저장소에 넣지 않았습니다. 실험을 다시 돌리려면 전체 프롬프트를 `prompts/detective_case.txt`에 넣으면 됩니다.

**한계.** 퀴즈 하나로만 실험했습니다. 채점 모델이 답변 모델과 같은 크기·계열이라 같은 편향을 가질 수 있습니다. 토큰 수는 Llama가 아닌 토크나이저로 셌고, 그래프는 온도별 통계가 아니라 이동 평균입니다.

## 실행 방법

```bash
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
ollama pull llama3.1:8b

# 01
cd 01_prefill_decode_profiling
python qwen_bench.py        # qwen_benchmark_refined.json 생성
python judge_results.py     # 선택: LLM 채점으로 다시 채점
python plot_json.py         # exponential_performance_dashboard.png 생성

# 02
cd ../02_temperature_reasoning
python run_temperature_sweep.py   # test_results.csv 생성
python judge_answers.py           # final_scored_results.csv 생성
python plot_results.py            # performance_matrix_result_1200.png 생성
```

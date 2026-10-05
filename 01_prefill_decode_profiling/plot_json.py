import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def draw_comprehensive_analysis(json_file):
    # 1. 데이터 로드 및 지수 가중치 지표 계산
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ '{json_file}' 파일을 찾을 수 없습니다.")
        return

    flat_data = []
    for entry in raw_data:
        prefill = entry["metrics"]["prefill_s"]
        decode = entry["metrics"]["decode_s"]
        total_time = prefill + decode
        score = entry["quality"]["final_score"]
        
        # [핵심 수정] 지수 가중치 적용 (세제곱)
        # 점수가 100점에 가까울수록 가치가 보존되고, 낮을수록 급격히 가치가 훼손됨
        quality_weight = (score / 100) ** 3
        weighted_score = quality_weight * 100
        
        # 보정된 QPS (시간 오버헤드 0.5초 포함)
        refined_qps = weighted_score / (total_time + 0.5) 
        
        flat_data.append({
            "scenario": entry["scenario"],
            "prefill_s": prefill,
            "decode_s": decode,
            "tps": entry["metrics"]["tps"],
            "score": score,
            "qps": refined_qps 
        })
    df = pd.DataFrame(flat_data)

    # 2. 시각화 설정 (5단 구성)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 5, figsize=(32, 7))

    metrics = [
        ('prefill_s', '1. Prefill Time', 'Seconds', 'Blues_d'),
        ('decode_s', '2. Decode Time', 'Seconds', 'Reds_d'),
        ('tps', '3. Tokens/Sec (TPS)', 'Tokens/Sec', 'magma'),
        ('score', '4. Quality Score', 'Score (0-100)', 'Greens_d'),
        ('qps', '5. Exponential Efficiency', 'Weighted QPS', 'rocket')
    ]

    for i, (col, title, ylabel, palette) in enumerate(metrics):
        sns.barplot(x='scenario', y=col, data=df, ax=axes[i], 
                    hue='scenario', palette=palette, legend=False, errorbar=None)
        axes[i].set_title(title, fontsize=13, fontweight='bold')
        axes[i].set_ylabel(ylabel)
        axes[i].tick_params(axis='x', rotation=45)
        
        for p in axes[i].patches:
            height = p.get_height()
            axes[i].annotate(f'{height:.2f}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 7), 
                            textcoords='offset points', fontsize=10, fontweight='bold')

    axes[3].set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig('exponential_performance_dashboard.png')
    
    # 분석 리포트 출력
    print("\n" + "="*75)
    print("🏆 [지수 가중치 반영] 최종 가성비(QPS) 순위")
    print("="*75)
    summary = df.groupby('scenario')['qps'].mean().sort_values(ascending=False)
    print(summary)
    print("\n💡 분석 결과: 품질이 낮은 시나리오는 지수 가중치에 의해 효율성 점수가 크게 삭감되었습니다.")
    print("="*75)

if __name__ == "__main__":
    draw_comprehensive_analysis('qwen_benchmark_refined.json')

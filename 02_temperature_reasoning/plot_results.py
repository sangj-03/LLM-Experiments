import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize():
    file_name = 'final_scored_results.csv'
    if not os.path.exists(file_name):
        print(f"'{file_name}' 파일이 없습니다. 채점 코드를 먼저 실행해주세요.")
        return

    df = pd.read_csv(file_name)
    df = df.sort_values(by=['temperature', 'iteration']).reset_index(drop=True)
    df['total_index'] = df.index + 1

    # 전체 시도 횟수 확인 (예: 1200)
    total_trials = len(df)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    window_size = 20 # 데이터가 많아졌으므로 윈도우 크기를 약간 키워 부드럽게 표현

    # 1. 정확성
    target_col = 'accuracy' if 'accuracy' in df.columns else 'final_score'
    acc_smooth = df[target_col].rolling(window=window_size, min_periods=1, center=True).mean()
    ax1.plot(df['total_index'], acc_smooth, color='#1f77b4', linewidth=2.5)
    ax1.set_ylabel('Reasoning Accuracy', fontsize=11, fontweight='bold')
    ax1.autoscale(enable=True, axis='y')

    # 2. 시간
    time_col = 'decoding_time' if 'decoding_time' in df.columns else 'duration'
    dur_smooth = df[time_col].rolling(window=window_size, min_periods=1, center=True).mean()
    ax2.plot(df['total_index'], dur_smooth, color='#d62728', linewidth=2)
    ax2.set_ylabel('Time (sec)', fontsize=11, fontweight='bold')

    # 3. 토큰 수
    token_smooth = df['token_count'].rolling(window=window_size, min_periods=1, center=True).mean()
    ax3.plot(df['total_index'], token_smooth, color='#2ca02c', linewidth=2)
    ax3.set_ylabel('Tokens', fontsize=11, fontweight='bold')
    ax3.set_xlabel(f'Total Trials (1-{total_trials})', fontsize=11)

    # --- 수정된 핵심 로직: 온도 구간 동적 계산 ---
    temps = sorted(df['temperature'].unique())
    # 온도당 시도 횟수를 동적으로 계산 (1200 / 4 = 300)
    trials_per_temp = total_trials // len(temps)
    
    colors = ['#f0f9ff', '#e0f2f1', '#fff3e0', '#ffebee']
    for i in range(len(temps)):
        # 고정값 100 대신 trials_per_temp 사용
        start = i * trials_per_temp
        end = (i + 1) * trials_per_temp
        
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(start, end, facecolor=colors[i % len(colors)], alpha=0.4)
            ax.axvline(x=start, color='black', alpha=0.1)
            
        # 텍스트 라벨 위치 조정
        label_pos = (start + end) / 2
        ax1.text(label_pos, ax1.get_ylim()[1], f"Temp {temps[i]}", 
                 ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Performance Matrix (Total {total_trials} Trials)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('performance_matrix_result_1200.png', dpi=300)
    print(f"그래프가 성공적으로 저장되었습니다. (총 {total_trials}회 데이터 반영)")
    plt.show()

if __name__ == "__main__":
    visualize()

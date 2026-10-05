# LLM Experiments

[한국어](README.ko.md)

Two small experiments with local LLMs that I ran as individual practice in the first half of 2026. Both use models served by [Ollama](https://ollama.com).

| Experiment | Model | Question | Runs |
| --- | --- | --- | --- |
| [01 Prefill vs. decode profiling](#01-prefill-vs-decode-profiling) | Qwen2.5-VL 7B | How do input and output length change latency, throughput and answer quality? | 4 scenarios × 100 rounds |
| [02 Sampling temperature vs. reasoning](#02-sampling-temperature-vs-reasoning) | Llama 3.1 8B | How does temperature change reasoning accuracy, answer time and output length? | 4 temperatures × 300 = 1,200 |

The raw result files (JSON and CSV) are not included. The figures in each `results/` folder are the outputs of the plotting scripts.

## 01 Prefill vs. decode profiling

`01_prefill_decode_profiling/`

**Setup.** Qwen2.5-VL 7B (`qwen2.5vl:7b`) on an NVIDIA RTX 4080 (16 GB), temperature 0.7, context length 8,192. Four scenarios cross a short or long input with a short or long output:

- Short input: a one-line question about today's weather. Long input: a multi-part weather request repeated 30 times.
- Short output: one sentence, at most 150 tokens. Long output: at least 300 characters, at most 1,024 tokens.

After a warm-up call, each of 100 rounds runs the four scenarios in random order, with a 0.5 s pause between calls.

**Metrics.**

- Prefill and decode time come from Ollama's `prompt_eval_duration` and `eval_duration`; tokens per second is generated tokens divided by decode time.
- `qwen_bench.py` grades each answer with simple rules: 50 points for meeting the length requirement and 50 points for keyword coverage.
- `judge_results.py` can re-grade the answers with Llama 3.1 8B as a judge (specificity 40, consistency 30, readability 30).
- `plot_json.py` combines quality and speed into a quality-weighted efficiency score: (score / 100)³ × 100 / (prefill + decode + 0.5 s).

**Results** (mean per scenario, from [`results/exponential_performance_dashboard.png`](01_prefill_decode_profiling/results/exponential_performance_dashboard.png)):

| Scenario | Prefill (s) | Decode (s) | Tokens/s | Quality score | Weighted efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short in / short out | 0.01 | 0.10 | 137.12 | 59.8 | 35.01 |
| Short in / long out | 0.01 | 1.96 | 128.69 | 54.1 | 10.06 |
| Long in / short out | 0.52 | 0.69 | 119.70 | 93.6 | 51.82 |
| Long in / long out | 0.47 | 3.95 | 117.84 | 92.8 | 19.23 |

- In these tests, output length dominated latency: decoding a long answer took up to 3.95 s, while reading even the long prompt took about 0.5 s.
- Throughput stayed in a narrow band (118–137 tokens/s) and was slightly lower with long inputs.

**Limitations.** One prompt topic, one model and one GPU. The rule-based quality score is a rough proxy (it counts keywords and length), and the model has no live weather data, so factual accuracy is not measured.

## 02 Sampling temperature vs. reasoning

`02_temperature_reasoning/`

**Setup.** Llama 3.1 8B (`llama3.1:8b`) answers one Korean detective-style logic puzzle: from seven suspects, it must name the culprit and explain the key clue in at most eight sentences. The puzzle is answered 300 times at each of four temperatures (0.1, 0.6, 1.0 and 1.8), with top-p at its default, a repeat penalty of 1.2 and at most 1,024 new tokens. Each run logs wall-clock time and output length (counted with tiktoken `cl100k_base`, an approximation for Llama).

**Scoring.** `judge_answers.py` asks Llama 3.1 8B (temperature 0, JSON output) to grade each answer: culprit 40 points, reasoning 40 points, format 20 points, minus 30 points if the answer invents evidence that is not in the puzzle.

**Plot.** `plot_results.py` draws a 20-run rolling mean of the score, time and token count across all 1,200 runs ([`results/performance_matrix_result_1200.png`](02_temperature_reasoning/results/performance_matrix_result_1200.png)).

**Results** (read from the plot): the judged score was highest around temperature 0.6 and lowest at 1.8, where answers were also longer and slower.

**Prompt.** The puzzle text comes from an external quiz and is not included in this repository. To rerun the experiment, put the full prompt in `prompts/detective_case.txt`.

**Limitations.** One puzzle only. The judge is a model of the same size and family, so it may share the answering model's biases. Token counts use a non-Llama tokenizer, and the plot shows rolling means rather than per-temperature statistics.

## Running

```bash
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
ollama pull llama3.1:8b

# 01
cd 01_prefill_decode_profiling
python qwen_bench.py        # writes qwen_benchmark_refined.json
python judge_results.py     # optional: LLM-judge re-grading
python plot_json.py         # writes exponential_performance_dashboard.png

# 02
cd ../02_temperature_reasoning
python run_temperature_sweep.py   # writes test_results.csv
python judge_answers.py           # writes final_scored_results.csv
python plot_results.py            # writes performance_matrix_result_1200.png
```

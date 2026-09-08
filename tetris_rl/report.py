"""Create a human-readable report and scientific plot from saved measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def create_report(evaluation: Path, run_dir: Path, budget_dir: Path):
    data = json.loads(evaluation.read_text(encoding="utf-8-sig"))
    results = data["results"]
    model = results["model"]
    summary = model["summary"]
    budget = json.loads((budget_dir / "budget.json").read_text(encoding="utf-8-sig"))
    best = json.loads((run_dir / "best_validation.json").read_text(encoding="utf-8-sig"))
    initial = json.loads((run_dir / "initial_validation.json").read_text(encoding="utf-8-sig"))
    freeze_path = evaluation.parent / "model_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    target_met = summary["episodes"] == 200 and summary["success_rate"] >= .99
    names = {"random": "유효 행동 무작위", "heuristic": "고정 휴리스틱", "untrained": "학습 전 무작위 가중치",
             "initialized": "전문가 초기 정책", "model": "CEM 학습 정책", "ppo": "PPO 학습 정책"}

    def f(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"

    rows, means = [], {}
    for name, entry in results.items():
        m = entry["summary"]
        successful = [e["pieces"] for e in entry["episodes"] if e["success"]]
        means[name] = float(np.mean(successful)) if successful else None
        rows.append(f"| {names.get(name,name)} | {m['successes']}/{m['episodes']} ({100*m['success_rate']:.1f}%) | "
                    f"{f(means[name],3)} | {f(m['successful_pieces_median'])} | {f(m['successful_pieces_p90'])} | "
                    f"{f(m['successful_game_time_seconds_median'])} | {f(entry['evaluation_wall_seconds'])} |")
    lo, hi = summary["success_rate_wilson_95"]
    model_episodes = {e["seed"]: e for e in model["episodes"]}
    init_episodes = {e["seed"]: e for e in results["initialized"]["episodes"]}
    diffs = [init_episodes[s]["pieces"] - model_episodes[s]["pieces"] for s in model_episodes
             if model_episodes[s]["success"] and init_episodes[s]["success"]]
    paired_gain = float(np.mean(diffs)) if diffs else None
    failed = []
    for name, entry in results.items():
        m = entry["summary"]
        failed.append(f"- {names.get(name,name)}: 실패 원인 `{json.dumps(m['failure_reasons'],ensure_ascii=False)}`, "
                      f"실패 판 제거 줄 중앙값 {f(m['failure_lines_median'])}, 90백분위수 {f(m['failure_lines_p90'])}.")
    meta = best["metadata"]
    validation_gain = initial["metrics"]["success_pieces_mean"] - best["metrics"]["success_pieces_mean"]
    text = f"""# 실제 실행 결과

독립 테스트 **{summary['successes']}/{summary['episodes']}판 성공 ({100*summary['success_rate']:.1f}%)**. 요청한 200판 성공률 99% 목표는 **{'달성' if target_met else '미달'}**입니다.

| 에이전트 | 성공률 | 성공 배치 평균 | 중앙값 | 90백분위수 | 합성 게임 시간 중앙값(초) | 전체 평가 실행(초) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

배치 수는 조각 고정 횟수입니다. 합성 게임 시간은 입력 토큰 수/60이며 중력·DAS/ARR·lock delay·줄 삭제 지연이 없는 로컬 규칙의 값입니다. 표의 실행 시간은 모든 성공/실패 판의 평가 루프 경과 시간이며 게임 시간과 다릅니다. 각 판의 실제 실행·추론 시간은 `{evaluation.name}`과 JSONL에 있습니다.

## 실제 학습과 초기값 구분

- 전문가 계수로 초기화한 정책은 검증 {initial['metrics']['successes']}/{initial['metrics']['episodes']}판 성공, 평균 {initial['metrics']['success_pieces_mean']:.2f}개 배치였습니다.
- 선택된 CEM 정책은 검증 {best['metrics']['successes']}/{best['metrics']['episodes']}판 성공, 평균 {best['metrics']['success_pieces_mean']:.2f}개 배치입니다. 검증 배치 평균 차이는 {validation_gain:.2f}개입니다.
- 최종 테스트에서 두 정책이 모두 성공한 동일 시드 {len(diffs)}판의 평균 배치 감소량은 {f(paired_gain,3)}개입니다. 작은 차이를 큰 성능 향상으로 해석하지 않습니다.
- 선택 세대: {meta['generation']}, 후보: `{meta['candidate']}`. 정규화한 초기 가중치로부터 L2 변화량은 {meta['parameter_l2_drift_from_normalized_initial']:.8f}입니다.
- 실제 환경 보상으로 CEM 분포를 갱신하고 보상으로 후보를 선택했습니다. 저장한 가중치로만 PyTorch 추론을 수행하며 휴리스틱 대체 경로가 없습니다. 완전한 무작위 초기화에서 이 성능을 학습한 것은 아닙니다.
- MaskablePPO 경로도 실제 최적화·저장·재로드를 수행했습니다. 별도의 PPO 결과는 `ppo_smoke_stdout.txt` 및 PPO 실험 폴더에 있습니다.

## 예산과 재현성

- 공통 원장 집계: **{budget['transitions']:,} / {budget['max_transitions']:,}전이**. 학습·검증·보수적인 초기 교정분을 포함합니다.
- 마지막 학습 종료까지 공통 예산 경과 시간: **{budget['elapsed_seconds']:.2f}초 ({budget['elapsed_seconds']/60:.2f}분)** / {budget['max_seconds']/3600:.0f}시간. 실험 사이 제어 시간도 포함하는 보수적인 시계입니다.
- CEM 구성과 각 후보 보상·가중치·학습 시드는 `{run_dir.name}/config_*.json`, `{run_dir.name}/training.jsonl`에 있습니다.
- 모델 고정 시각: `{freeze.get('frozen_at_utc','not recorded')}`. 최종 평가 전 모델을 고정했고 최종 테스트로 재튜닝하지 않았습니다.
- 최종 체크포인트 SHA256: `{model['checkpoint_sha256']}`.
- 최종 시드 목록 SHA256: `{data['seed_sha256']}`.
- 테스트 시드: 2000000..2000199. 검증 시드는 1000000..1000049이며 학습 시드 범위와 분리했습니다.
- 관측 성공률의 Wilson 95% 구간은 **{100*lo:.2f}%–{100*hi:.2f}%**입니다. 200판 측정 결과가 전체 가능한 시퀀스의 성공률을 보장하지는 않습니다.

## 실패 기록

{chr(10).join(failed)}

## 산출물과 한계

- `final_model.pt`: 최종 평가 전에 고정한 학습 정책. `model_freeze.json`은 선택 근거와 해시입니다.
- `final_test.json`, `final_test_*_episodes.jsonl`: 전체 200판 결과를 에이전트별로 보존합니다.
- `replays/model_2000000.html`: 첫 테스트 판의 오프라인 리플레이. 실패 판을 걸러낸 선택이 아닙니다.
- `training_and_evaluation.png`: 저장된 측정에서 만든 학습/평가 그래프.
- 현재 정책은 사람이 설계한 배치 특징을 사용하는 선형 정책이며 다단계 next lookahead는 없습니다. 다음 개선은 별도 검증 시드를 준비한 뒤 제한된 lookahead나 학습된 afterstate value를 비교하는 것입니다. 실제 스프린트 시간 최적화에는 대상 게임의 시간 규칙을 갖춘 별도 환경이 필요합니다.
"""
    out = evaluation.parent / "RESULTS.md"
    out.write_text(text, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    events = [json.loads(line) for line in (run_dir / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [e for e in events if e["event"] == "cem_validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for kind, marker in [("mean","o"),("reward_selected_elite","s")]:
        group = [e for e in val if e["candidate"] == kind]
        axes[0].plot([e["generation"] for e in group], [e["validation"]["success_pieces_mean"] for e in group],
                     marker=marker, markersize=3, alpha=.8, label=kind.replace("_"," "))
    axes[0].axhline(initial["metrics"]["success_pieces_mean"], color="#a45a22", ls="--", label="initial policy")
    axes[0].set(xlabel="CEM generation", ylabel="Mean placements (successful validation episodes)", title="Validation: 50 fixed seeds")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    labels = list(results)
    rates = [results[n]["summary"]["success_rate"]*100 for n in labels]
    axes[1].bar(labels, rates, color=["#708090" if n!="model" else "#087e8b" for n in labels])
    axes[1].axhline(99, color="#a45a22", ls="--", lw=1, label="99% target")
    for i,n in enumerate(labels):
        m=results[n]["summary"]
        axes[1].text(i,rates[i]+1.5,f"{m['successes']}/{m['episodes']}",ha="center",fontsize=9)
    axes[1].set(ylabel="Success rate (%)", ylim=(0,113), title="Held-out final test: 200 paired seeds")
    axes[1].tick_params(axis="x",labelrotation=25)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y",alpha=.2)
    fig.savefig(evaluation.parent / "training_and_evaluation.png", dpi=160)
    plt.close(fig)
    print(out.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation",type=Path,default=Path("artifacts/final_test.json"))
    p.add_argument("--run-dir",type=Path,default=Path("artifacts/cem"))
    p.add_argument("--budget-dir",type=Path,default=Path("artifacts/run"))
    a=p.parse_args()
    create_report(a.evaluation,a.run_dir,a.budget_dir)


if __name__ == "__main__":
    main()

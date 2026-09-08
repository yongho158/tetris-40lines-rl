# Tetris 40 Lines 강화학습

빈 보드에서 40줄을 제거하는 로컬 에이전트입니다. 실제 이동 경로가 있는 배치만 선택하고, 학습된 PyTorch 정책으로 플레이합니다. 목표는 **독립 테스트 200판에서 성공률 99% 이상**, 성공률이 같으면 더 적은 배치 수입니다.

학습된 체크포인트와 평가 기록, 실시간 플레이 화면을 함께 제공합니다. 아래 빠른 시작 명령은 원하는 폴더에서 실행할 수 있습니다. 최초 실험은 `C:\tetris\_workspace`에서 수행했으며, 뒤쪽 실험 기록과 재개 명령의 경로는 당시 작업 위치를 나타냅니다.

## GitHub에서 받아 바로 실행하기

Python 3.13과 Git이 설치된 Windows PowerShell에서 실행합니다. 비공개 저장소를 내려받을 때는 저장소 접근 권한이 있는 GitHub 계정으로 인증해야 합니다.

```powershell
git clone https://github.com/yongho158/tetris-40lines-rl.git
Set-Location '.\tetris-40lines-rl'
py -3.13 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.\.venv\Scripts\python.exe' -m pip install --no-deps -e .
& '.\.venv\Scripts\python.exe' -m tetris_rl.live --checkpoint artifacts/final_model.pt
```

브라우저에서 열린 실시간 화면의 **시작**을 누르면 AI가 새 게임을 진행합니다. 학습된 모델이 포함되어 있으므로 플레이 전에 재학습할 필요가 없습니다. 기본 주소는 `http://127.0.0.1:8766`이며 실행 중에는 터미널을 열어 둡니다. 종료는 `Ctrl+C`입니다.

```powershell
# 프로젝트 루트에서 실행
& '.\.venv\Scripts\python.exe' -m pytest
# Node.js가 설치되어 있으면 화면 제어도 검사
node tests/live_ui_check.cjs
node tests/replay_timing_check.cjs
```

새로 학습하려면 기존 실험 예산과 체크포인트를 보존하도록 **새 실험 폴더**를 사용합니다. 아래 예시는 처음 2세대만 학습·검증하는 실행입니다. 포함된 `artifacts/run/budget.json`은 완료된 원래 실험의 누적 시간·전이 기록이므로 새 실험 예산으로 사용하지 않습니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.train --method cem --run-dir artifacts/new_cem --budget-dir artifacts/new_budget --steps 200000 --population 24 --train-episodes 16 --validation-episodes 50 --validation-seeds configs/validation_seeds.json --generations 2 --seed 1729 --reward-config configs/reward.json
```

## 이번 실행 결과

**최종 테스트 200/200판 성공(100%)으로 목표 99%를 달성했습니다.** 최종 테스트 이전에 모델과 설정을 고정했으며 테스트 결과로 재튜닝하지 않았습니다.

| 에이전트 | 최종 성공률 | 성공 배치 평균 | 중앙값 | 90백분위수 |
|---|---:|---:|---:|---:|
| 무작위 유효 행동 | 0/200 | — | — | — |
| 무작위 가중치의 학습 전 정책 | 0/200 | — | — | — |
| 고정 휴리스틱 / 전문가 초기 정책 | 200/200 | 102.300 | 102 | 104 |
| CEM 학습 정책 | 200/200 | **102.085** | 102 | **103** |

전문가 초기 정책도 이미 100% 성공했습니다. 강화학습의 측정된 효과는 성공률 유지와 평균 배치 수 0.215개 감소입니다. CEM 20세대와 실제 재개 1세대를 수행했고, 같은 검증 50판에서 평균 101.94개인 CEM을 102.60개인 PPO보다 우선 선택했습니다. 학습된 14개 가중치로만 추론하며 휴리스틱 대체는 없습니다.

- 공통 예산: 학습·검증·초기 교정을 포함해 **1,101,435전이**, **610.39초(10.17분)**. 상한 5,000,000전이/5시간 이내입니다.
- 최종 모델 200판 평가 루프 실행 시간: **6.96초**. 합성 게임 시간 중앙값은 **7.47초/판**이며 실제 상용 스프린트 시간이 아닙니다.
- 테스트: 초기 학습 구현 **33개 통과**, 실시간 플레이 추가 후 전체 **48개 통과**. 실시간 화면과 기존 리플레이의 JavaScript 제어 검사도 통과했습니다. 게임 규칙, 독립 SRS 입력 재생, 정보 누출 방지, 저장/로드, 예산, 재개를 검증했습니다.
- [상세 결과](artifacts/RESULTS.md), [최종 모델](artifacts/final_model.pt), [평가 원본](artifacts/final_test.json), [리플레이](artifacts/replays/model_2000000.html), [학습 그래프](artifacts/training_and_evaluation.png).

## 설치와 테스트

Python 3.13.9와 Windows 11에서 실행했습니다. i5-11400F, RAM 32GB, RTX 3060 12GB를 확인했습니다. 이번 실행은 CPU PyTorch와 Numba를 사용합니다. 정책이 14개 가중치로 구성되어 있고 작은 후보 행렬을 연속 처리하므로 GPU 설치 대신 CPU 1스레드 추론을 사용했습니다. 패키지 버전과 장치 정보는 `requirements-lock.txt`, `artifacts/hardware.json`에 있습니다.

```powershell
Set-Location 'C:\tetris\_workspace'
py -3.13 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.\.venv\Scripts\python.exe' -m pip install --no-deps -e .
& '.\.venv\Scripts\python.exe' -m pytest
```

현재 환경에는 의존성이 이미 설치되어 있습니다. 첫 실행의 Numba JIT 컴파일에는 몇 초가 추가되며 이후에는 캐시를 사용합니다. 가상환경을 활성화할 필요 없이 위와 같이 실행할 수 있습니다.

## 학습

실제 본학습 명령입니다. CEM은 후보 정책으로 게임을 플레이해 얻은 에피소드 보상으로 탐색 분포를 갱신하는 정책 탐색 강화학습입니다. 최종 정책은 저장된 학습 가중치만 사용합니다.

```powershell
Set-Location 'C:\tetris\_workspace'
& '.\.venv\Scripts\python.exe' -m tetris_rl.train --method cem --run-dir artifacts/cem --budget-dir artifacts/run --steps 1600000 --population 24 --train-episodes 16 --validation-episodes 50 --validation-seeds configs/validation_seeds.json --generations 20 --seed 1729 --reward-config configs/reward.json
```

동일한 실험을 이어서 실행합니다. 기존 예산의 시작 시각과 누적 전이를 유지하며, 재개한다고 5시간/500만 회 제한이 초기화되지 않습니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.train --method cem --run-dir artifacts/cem --budget-dir artifacts/run --resume artifacts/cem/cem_state.json --steps 500000 --population 24 --train-episodes 16 --validation-episodes 50 --validation-seeds configs/validation_seeds.json --generations 5 --seed 1729 --reward-config configs/reward.json
```

`--steps`는 이번 호출의 학습+검증 전이 상한입니다. 모든 실험은 `artifacts/run/budget.json`을 공유하며 **5시간 또는 5,000,000전이 중 먼저 도달하는 한도**를 넘지 않습니다. 보수적으로 첫 학습부터의 경과 시간과 검증 전이까지 집계합니다. 최초 엔진 교정 실행에는 4,000전이를 보수적으로 선반영했습니다. 학습 프로세스는 하나씩 순차 실행하십시오. 예산이 끝나면 현재 상태를 저장합니다. 프로세스가 강제 종료되면 이미 예약된 최대 127개의 미사용 전이가 추가 집계될 수 있어 상한을 과소 계산하지 않습니다.

이 완료된 실험과 별개로 새 실험을 시작하려면 새로운 `--run-dir`와 `--budget-dir`를 지정하십시오. 새 실험의 결과로 기존 최종 테스트 결과를 덮어쓰거나 기존 테스트 시드에 맞춰 튜닝하지 마십시오.

MaskablePPO 경로도 실제 학습·저장·로드를 검증했습니다. 신규 PPO는 잠재함수 보상과 일치하는 `gamma=1.0`을 사용합니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.train --method ppo --run-dir artifacts/ppo --budget-dir artifacts/run --steps 30000 --ppo-chunk-steps 4096 --generations 2 --validation-episodes 50 --validation-seeds configs/validation_seeds.json --seed 1729 --reward-config configs/reward.json
```

PPO 재개 시 `--resume artifacts/ppo/latest_ppo.zip`을 지정합니다. 학습률·탐색 폭·population·보상 가중치는 CLI와 `configs/reward.json`으로 관리합니다. 실행 당시 인수는 각 실험 폴더의 `config_*.json`에 저장됩니다.

## 평가

검증 시드로 무작위·휴리스틱·초기 정책·학습 정책을 비교합니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.evaluate --agents random heuristic untrained initialized model --checkpoint untrained=artifacts/cem/untrained.pt --checkpoint initialized=artifacts/cem/initialized.pt --checkpoint model=artifacts/cem/best.pt --seeds configs/validation_seeds.json --seed-set-name validation --reward-config configs/reward.json --config configs/experiment.json --capture 1 --output artifacts/validation_comparison.json
```

모델과 설정을 고정한 뒤에만 최종 테스트를 실행합니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.evaluate --agents random heuristic untrained initialized model --checkpoint untrained=artifacts/cem/untrained.pt --checkpoint initialized=artifacts/cem/initialized.pt --checkpoint model=artifacts/final_model.pt --seeds configs/test_seeds.json --seed-set-name final_test_recheck --reward-config configs/reward.json --config configs/experiment.json --capture 1 --output artifacts/final_test_recheck.json
```

- 학습 조각 시드: `0..999999`. 정책은 시드를 관측하지 않습니다.
- 검증 시드: `1000000..1000049`. 체크포인트 선택에 사용합니다.
- 최종 테스트 시드: `2000000..2000199`. 학습과 설정 선택에 사용하지 않습니다.
- 모든 비교 에이전트는 동일한 규칙, 정보 범위, 7-bag 시퀀스로 평가합니다. Hold 선택이 다르면 같은 시퀀스를 소비하는 순서는 달라질 수 있습니다.
- 성공률과 판 수, 성공 판의 배치 수 중앙값/90백분위수, 실패 판의 줄 수/실패 원인을 별도로 기록합니다. 성공 사례만 골라 평가하지 않습니다.
- 모델·소스·설정·시드의 해시, 매 판 JSONL, 합계 JSON으로 결과를 추적합니다. 200판의 관측 성공률은 모든 가능한 조각 시퀀스에 대한 보장이 아니며 Wilson 95% 구간을 함께 저장합니다.

위 재평가 명령은 처음 저장한 `final_test.json`을 보존하도록 다른 결과 파일을 사용합니다. 테스트 시드를 이미 관찰한 뒤에는 새 후보 선택용으로 사용하지 마십시오. 현재 원본 보고서를 다시 생성하는 명령은 다음과 같습니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.report --evaluation artifacts/final_test.json --run-dir artifacts/cem --budget-dir artifacts/run
```

## AI 실시간 플레이

학습된 모델이 **새 게임의 현재 관측과 유효 행동 마스크로 매번 추론**하고, 선택한 이동·SRS 회전·홀드·하드드롭을 브라우저에 표시합니다. 리플레이 파일이나 미리 완성된 게임을 불러오지 않습니다. 한 조각의 입력 경로 표시가 끝나야 다음 배치를 요청합니다. 모델 체크포인트를 그대로 사용하며 재학습하지 않습니다.

```powershell
Set-Location 'C:\tetris\_workspace'
& '.\.venv\Scripts\python.exe' -m tetris_rl.live --checkpoint artifacts/final_model.pt
```

모델과 엔진 준비 후 기본 브라우저에서 `http://127.0.0.1:8766`을 엽니다. **시작**을 누르면 진행합니다. 터미널을 실행한 상태로 두고, 서버를 종료하려면 터미널에서 `Ctrl+C`를 누릅니다. Python 표준 라이브러리 서버를 사용하므로 추가 패키지 설치는 필요 없습니다.

```powershell
# 브라우저를 자동으로 열지 않고 실행 / 포트 변경 / 최초 시드 지정
& '.\.venv\Scripts\python.exe' -m tetris_rl.live --checkpoint artifacts/final_model.pt --no-browser --port 8767 --seed 3000000
```

- **시작·일시정지**, **한 조각 진행**, **새 게임**을 지원합니다. 시드를 비우면 새 무작위 게임, 같은 시드를 입력하면 같은 조각 순서로 시작합니다. 시드는 정책 입력에 포함되지 않습니다.
- 보드 위에 현재 미노 이름과 모양을 표시합니다. 보드에는 활성 미노와 현재 자세의 착지 위치를 표시하며, 홀드를 사용하는 순간 현재 미노·홀드·Next 5도 바뀝니다. 점선 위 4행은 원래 숨김 영역으로, 생성과 회전 동작을 볼 수 있도록 표시합니다.
- **0.25×, 0.5×, 1×, 2×, 4×** 속도를 지원합니다. 보기용 속도는 입력당 0.12초, 고정 화면 0.12초입니다. **원래 속도로 플레이**는 입력당 1/60초·1배속으로 전환하고 진행합니다. 하드드롭은 실제 규칙처럼 즉시 착지합니다.
- 원래 속도는 학습 환경의 합성 게임 시간입니다. 추론·HTTP 통신·화면 갱신 비용 때문에 실제 경과 시간은 더 길어질 수 있습니다. 배치 수, 게임 내 시간, 정지 시간을 포함한 실제 경과 시간, 모델 추론 합계를 따로 표시합니다.
- 다른 탭으로 이동하면 일시정지합니다. 진행 중인 한 조각의 서버 요청이 있다면 그 응답만 보관하고 화면 진행과 다음 추론은 정지합니다. 네트워크 오류가 나면 자동으로 멈추며 서버를 확인한 뒤 새 게임으로 복구합니다.
- 40줄 달성·top-out·500개 배치 제한에서 종료합니다. 여러 브라우저 탭은 별도의 게임 세션을 사용합니다. 서버는 이 컴퓨터의 `127.0.0.1`에서만 접속할 수 있습니다.

진입점은 `tetris_rl/live.py`, 화면은 `tetris_rl/live.html`입니다. 기존 `engine.py`, `env.py`, `policy.py`의 규칙과 정책을 재사용합니다. 학습과 동일한 **0G / 자동 고정 지연 없음**을 유지하며 키보드로 직접 조각을 조작하는 모드는 아닙니다.

## 리플레이

평가의 `--capture 1`은 각 에이전트의 첫 평가 판을 저장합니다. 첫 판이 실패해도 그대로 남깁니다. JSON의 공개 보드 상태와 실제 입력 시퀀스로 오프라인 HTML을 만듭니다.

```powershell
& '.\.venv\Scripts\python.exe' -m tetris_rl.replay artifacts/replays/model_2000000.json
Start-Process -FilePath 'artifacts/replays/model_2000000.html'
```

HTML 파일은 네트워크나 서버 없이 열 수 있습니다. 보드 위에 현재 미노의 이름과 색상·모양을 표시하고, 재생/일시정지, 이전/다음 배치, 슬라이더, hold와 next 5개를 제공합니다. 화면은 조각을 고정하고 줄을 제거한 직후의 배치 경계입니다. 각 경계까지의 입력 경로도 표시합니다.

- 배속: **0.25×, 0.5×, 1×, 2×, 4×, 8×**.
- **보기용 속도**: 기존처럼 1배속에서 조각당 0.6초 간격으로 재생합니다.
- **기록된 게임 시간**: 각 배치의 입력 틱 차이(60틱/초)를 재현합니다. 이 모드에도 배속을 적용할 수 있습니다.
- **원래 속도로 재생**: 현재 위치에서 기록된 게임 시간·1배속으로 즉시 재생합니다. 끝난 상태에서는 처음부터 시작합니다. 첫 모델 리플레이의 원래 길이는 7.5667초이며, 기록 시간 0.5배는 15.1333초, 0.25배는 30.2667초입니다.
- 일시정지는 현재 배치까지의 진행 시간을 유지합니다. 슬라이더나 이전/다음 버튼으로 이동하면 그 배치의 시작부터 재생합니다. 탭이 숨겨지면 자동 일시정지합니다.

원래 속도는 이 환경의 합성 게임 시간을 뜻하며 컴퓨터의 계산 시간은 아닙니다. 입력 중간 위치는 저장되어 있지 않으므로 조각 고정 후의 화면을 원래 간격에 맞춰 보여줍니다. Node.js가 있으면 `node tests/replay_timing_check.cjs`로 배속·원래 길이·일시정지·탐색·속도 변경을 가상 시계로 검증할 수 있습니다.

## 게임 규칙과 시간 정의

- 보드: 가시 10×20 + 위쪽 숨김 4행. 조각 생성 원점 `(x=3,y=2)`이며 좌표는 오른쪽/아래쪽이 양수입니다.
- 조각: I/J/L/O/S/T/Z의 7-bag. SRS 90도 양방향 회전과 I/JLSTZ별 wall kick을 사용합니다. 180도 회전은 없습니다.
- Hold: 한 매크로 행동은 선택적 hold 1회와 고정 1회입니다. 빈 hold에서는 next를 먼저 소비하고, 고정 후 다음 조각을 다시 소비합니다. 채워진 hold는 현재 조각과 교환합니다.
- 행동: `(hold,rotation,y,x)`를 인코딩한 2,496개 슬롯. BFS로 왼쪽/오른쪽/아래/CW/CCW 경로와 최종 hard drop이 가능한 슬롯만 허용합니다. O의 중복 회전 배치는 제거합니다.
- 성공: 누적 줄 수가 40 이상. 마지막 4줄 제거로 43줄이 되어도 성공입니다.
- 실패: 줄 제거 후 숨김 영역에 점유 셀이 남으면 partial lock-out. 새 조각 생성 충돌은 block-out이며 hold로 이미 실패한 생성을 구제하지 않습니다. 성공 판정은 같은 행동의 실패·배치 제한보다 우선합니다.
- 500개 배치 제한: `truncated=True`. 성공/top-out은 `terminated=True`이며 평가에서 제한 내 미완료는 실패입니다.
- **0G, 무제한 조작의 배치 학습 환경**입니다. 자동 중력, lock delay, DAS/ARR, 줄 삭제 대기 시간은 없습니다. 게임 내 시간은 입력 토큰 하나당 1틱, 60틱=1초로 계산한 합성 지표입니다. 상용 테트리스의 실제 40 Lines 스프린트 기록과 비교할 수 없습니다.
- 배치 수는 고정한 조각 수, 게임 내 시간은 위 입력 틱, 실행 시간은 컴퓨터의 실제 벽시계 시간으로 서로 분리합니다.

## 정책·보상·학습의 범위

관측에는 24×10 보드, 현재/hold/공개 next 5개/hold 가능 여부/줄 수/배치 수를 담습니다. 후보 특징은 모두 현재 공개 상태에서 계산한 배치 후 상태입니다. 비공개 bag, RNG 상태, seed와 미공개 조각을 정책 입력에 넣지 않습니다.

14개 특징은 착지 높이, 제거 줄에 포함된 현재 조각 셀 수×제거 줄 수, 행/열 전이, 구멍, 누적 우물 깊이, 총 높이, 표면 굴곡, 제거 줄 수, 최대 높이, 구멍 깊이, 구멍이 있는 행 수, hold 여부, 입력 수입니다. 최종 정책은 특징과 학습 가중치의 내적으로 유효 후보를 선택합니다. 보드는 신경망이 특징을 새로 학습하는 원시 픽셀 입력이 아닙니다.

고정된 Dellacherie 계수를 **명시적인 전문가 초기값**으로 사용합니다. 무작위 가중치의 학습 전 정책, 고정 휴리스틱, 전문가 초기 정책, 실제 RL 갱신 정책을 구분합니다. 행동 복제 데이터셋을 이용한 초기화는 사용하지 않습니다. CEM은 후보 정책의 실제 게임 보상으로 elite와 평균/표준편차를 갱신합니다. 검증 점수로 체크포인트를 고르고, 초기 가중치와 실제 차이가 있는 후보만 최종 RL 모델 자격을 갖습니다. 모델 로드 실패 시 휴리스틱으로 대체하는 경로는 없습니다.

기본 보상은 제거 줄당 +1(목표까지의 남은 줄로 상한), 성공 +100, top-out -100, 배치당 -0.05입니다. 보조 보상은 높이와 구멍으로 만든 범위 제한 잠재함수의 차이이며 종료 시 잠재값은 0입니다. 따라서 빈 보드에서 성공한 에피소드의 총 보상은 정확히 **140 − 0.05×배치 수**로, 초과 줄이나 보조 보상만으로 점수를 부풀릴 수 없습니다. 유효 마스크는 학습·평가·추론에 일관되게 적용합니다.

현재 선형 정책은 현재 조각 또는 hold 이후 조각의 한 번 배치만 평가합니다. Next 5개는 관측에 제공되지만 다단계 lookahead에는 사용하지 않습니다. 외부 상용 게임 클라이언트를 자동 조작하는 기능, 실시간 영상 인식, 상용 게임 기록 최적화는 포함하지 않습니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `tetris_rl/engine.py` | SRS, 충돌/줄 제거, Numba BFS, 후보 특징, 입력 재생 |
| `tetris_rl/env.py` | Gymnasium 환경, 7-bag/hold, 보상, 종료, 공개 관측 |
| `tetris_rl/policy.py` | 고정 기준 정책, 학습 가능한 PyTorch scorer, MaskablePPO 정책 |
| `tetris_rl/train.py` | CEM/PPO 학습, 검증, 체크포인트, 재개 |
| `tetris_rl/budget.py` | 영속적 누적 예산과 종료 한도 |
| `tetris_rl/evaluate.py` | 동일 seed 평가, 지표·해시·리플레이 저장 |
| `tetris_rl/replay.py` | 독립 HTML 재생기 |
| `tetris_rl/live.py`, `tetris_rl/live.html` | 실제 모델 추론 서버와 이동 경로를 표시하는 실시간 화면 |
| `tetris_rl/report.py` | 저장된 평가·학습 로그로 보고서와 그래프 생성 |
| `tests/` | 규칙, 독립 입력 경로, 누출 방지, 정책 저장/로드, 예산 검증 |

## 참고 자료

- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [MaskablePPO 공식 문서](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [Learning Tetris Using the Noisy Cross-Entropy Method, Szita & Lőrincz (2006)](https://doi.org/10.1162/neco.2006.18.12.2936)
- [The Game of Tetris in Machine Learning (2019)](https://arxiv.org/abs/1905.01652)

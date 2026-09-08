"""Convert a public-state replay JSON into a standalone offline HTML viewer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tetris 40 Lines · Replay</title>
<style>
:root{color-scheme:dark;font-family:Inter,"Segoe UI",sans-serif;background:#0b1220;color:#e8edf5}
*{box-sizing:border-box}body{margin:0;padding:28px}main{max-width:980px;margin:auto}
header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:24px}
h1{font-size:26px;margin:0 0 7px}p{margin:0;color:#98a8bc;line-height:1.5}.badge{border:1px solid #294565;border-radius:20px;padding:8px 14px;color:#78ded0;white-space:nowrap}
.layout{display:grid;grid-template-columns:360px 1fr;gap:24px}.card{background:#111d2e;border:1px solid #26364c;border-radius:14px;padding:20px}
.board-card{display:flex;flex-direction:column;align-items:center;gap:14px;padding:16px;align-self:start}canvas{display:block;max-width:100%;width:auto;height:auto;max-height:max(320px,calc(100vh - 244px))}
.current-preview{display:flex;align-items:center;justify-content:space-between;gap:14px;width:100%;padding:10px 12px;background:#0b1524;border:1px solid #2b4059;border-radius:9px;min-height:66px}.current-name{font-size:21px;font-weight:700;color:var(--mino-color,#e8edf5)}.current-preview .label{margin-bottom:3px}.mino-shape{display:grid;gap:3px;flex:none;min-height:20px;align-content:center}.mino-cell{width:18px;height:18px;visibility:hidden}.mino-cell.filled{visibility:visible;background:var(--mino-color);border-radius:2px;box-shadow:inset 0 2px 0 #ffffff45,inset 0 -2px 0 #00000025}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:20px 0}.stat{border-radius:9px;background:#0b1524;padding:13px}.label{font-size:12px;color:#97a9c0;margin-bottom:5px}.value{font-size:25px;font-variant-numeric:tabular-nums}.small{font-size:14px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#9eb0c8;margin:0 0 12px}.pieces{display:flex;flex-wrap:wrap;gap:9px}.piece{min-width:38px;text-align:center;padding:8px;border-radius:6px;background:#22324a;font-weight:700}
.hold{margin-bottom:20px}.controls{margin-top:24px}.row{display:flex;gap:12px;align-items:center;margin-top:14px;flex-wrap:wrap}
button,select{font:inherit;color:#e8edf5;border:1px solid #3a4c66;border-radius:7px;background:#24364e;padding:9px 14px}button{cursor:pointer}button:hover{background:#345373}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #66dfcd;outline-offset:3px}
input[type=range]{width:100%;accent-color:#5cd8c2}#position{margin-left:auto;font-variant-numeric:tabular-nums;color:#a8bbd3}
.path{font-family:Consolas,monospace;font-size:13px;line-height:1.6;word-break:break-word;min-height:44px;color:#c2d1e5}.note{font-size:12px;margin-top:20px;line-height:1.6}footer{color:#72849e;font-size:12px;margin-top:20px}
@media(max-width:760px){body{padding:16px}.layout{grid-template-columns:1fr}.board-card{max-width:390px;width:100%;margin:auto}header{align-items:flex-start}h1{font-size:22px}.badge{font-size:12px}}
</style></head><body><main>
<header><div><h1>Tetris · 40 Lines</h1><p id="subtitle"></p></div><span class="badge" id="status"></span></header>
<div class="layout"><section class="card board-card"><div class="current-preview" id="current-preview"><div><div class="label" id="current-label">현재 미노</div><div class="current-name" id="current-name"></div></div><div class="mino-shape" id="current-shape" role="img" aria-label="현재 미노 모양"></div></div><canvas id="board" width="320" height="768" aria-label="테트리스 보드와 숨김 행"></canvas></section>
<section class="card"><div class="hold"><h2>Hold / Current</h2><div class="pieces"><span class="piece" id="hold"></span><span class="piece" id="current"></span></div></div>
<h2>Next 5</h2><div class="pieces" id="next"></div><div class="stats">
<div class="stat"><div class="label">제거한 줄</div><div class="value" id="lines"></div></div>
<div class="stat"><div class="label">고정한 조각</div><div class="value" id="pieces"></div></div>
<div class="stat"><div class="label">게임 내 시간 (입력 / 60)</div><div class="value" id="game-time"></div></div>
<div class="stat"><div class="label">누적 입력 틱</div><div class="value" id="ticks"></div></div></div>
<h2>이 배치까지 실행한 입력</h2><div class="path" id="path"></div>
<div class="controls"><label for="slider" class="label">배치 시점 탐색</label><input id="slider" type="range" min="0" value="0" step="1"><div class="row">
<button id="play" type="button">재생</button><button id="back" aria-label="이전 배치">←</button><button id="forward" aria-label="다음 배치">→</button>
<span id="position"></span></div><div class="row"><label for="timing-mode" class="label">재생 기준</label><select id="timing-mode"><option value="view">보기용 속도</option><option value="recorded" id="recorded-mode">기록된 게임 시간</option></select>
<label for="speed" class="label">배속</label><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></div>
<div class="row"><button id="original-speed" type="button">원래 속도로 재생</button></div><p class="note" id="timing-note"></p></div>
<p class="note">점선 위 4행은 숨김 영역입니다. 각 화면은 조각을 고정하고 줄을 삭제한 직후입니다. 원래 속도는 저장된 게임 시간(입력 1개당 1/60초)을 따릅니다. 재생 기준과 배속을 바꿔도 게임 기록은 바뀌지 않습니다.</p>
<footer id="performance"></footer></section></div></main>
<script>
const data=__REPLAY_DATA__;
const $=id=>document.getElementById(id), canvas=$('board'), ctx=canvas.getContext('2d');
const palette=['#101b2c','#52d1e5','#5d85ee','#f1a259','#eed36b','#77d8a7','#b58af0','#ec7c8f'];
const names=['I','J','L','O','S','T','Z'];let index=0,playing=false,last=0,elapsed=0,timingMode='view',playbackRate=1;
const minoShapes={I:['1111'],J:['100','111'],L:['001','111'],O:['11','11'],S:['011','110'],T:['010','111'],Z:['110','011']};
const frames=data.frames, max=frames.length-1;
if(!frames.length)throw Error('Replay contains no frames');
const recordedDelays=data.actions.map((action,i)=>{const a=frames[i],b=frames[i+1];
if(Number.isFinite(a.input_ticks)&&Number.isFinite(b.input_ticks)&&b.input_ticks>a.input_ticks)return (b.input_ticks-a.input_ticks)*1000/60;
if(Number.isFinite(a.game_time_seconds)&&Number.isFinite(b.game_time_seconds)&&b.game_time_seconds>a.game_time_seconds)return (b.game_time_seconds-a.game_time_seconds)*1000;
return Array.isArray(action.inputs)&&action.inputs.length?action.inputs.length*1000/60:null;});
const hasRecordedTiming=max>0&&recordedDelays.length===max&&recordedDelays.every(v=>Number.isFinite(v)&&v>0);
$('recorded-mode').disabled=!hasRecordedTiming;$('original-speed').disabled=!hasRecordedTiming;
$('slider').max=max;$('subtitle').textContent=`${data.agent||'agent'} · seed ${data.seed} · ${data.seed_set_name||'recorded episode'}`;
$('status').textContent=data.episode.success?'최종 결과: 40 Lines 성공':`최종 결과: 미완료 · ${data.episode.reason}`;
$('performance').textContent=`이 판 실행 ${data.episode.wall_seconds.toFixed(3)}초 · 정책 추론 합계 ${data.episode.inference_seconds.toFixed(3)}초`;
function piece(value){return value===null||value===undefined||value===-1?'없음':(typeof value==='number'?names[value]:String(value));}
function drawCurrent(value){const name=piece(value),shape=minoShapes[name],preview=$('current-preview'),icon=$('current-shape');
$('current-name').textContent=shape?`${name} 미노`:'없음';preview.style.setProperty('--mino-color',palette[names.indexOf(name)+1]||'#e8edf5');
icon.setAttribute('aria-label',shape?`현재 ${name} 미노 모양`:'현재 미노 없음');icon.style.gridTemplateColumns=shape?`repeat(${shape[0].length},18px)`:'none';
icon.replaceChildren(...(shape?shape.join('').split('').map(cell=>{const block=document.createElement('span');block.className=cell==='1'?'mino-cell filled':'mino-cell';return block;}):[]));}
function draw(){const f=frames[index], board=f.board,cell=32;ctx.clearRect(0,0,canvas.width,canvas.height);
for(let y=0;y<board.length;y++)for(let x=0;x<board[y].length;x++){const v=board[y][x];ctx.fillStyle=v?palette[(Math.abs(v)-1)%7+1]:(y<4?'#172238':'#0d1828');ctx.fillRect(x*cell+1,y*cell+1,cell-2,cell-2);if(v){ctx.fillStyle='#ffffff25';ctx.fillRect(x*cell+3,y*cell+3,cell-6,3);}}
ctx.save();ctx.strokeStyle='#758bac';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(0,4*cell);ctx.lineTo(320,4*cell);ctx.stroke();ctx.restore();
$('lines').textContent=`${f.lines} / ${data.target_lines}`;$('pieces').textContent=f.pieces;$('ticks').textContent=f.input_ticks||0;
$('game-time').textContent=`${Number(f.game_time_seconds??(f.input_ticks||0)/60).toFixed(2)}s`;
$('hold').textContent=`Hold: ${piece(f.hold)}`;$('current').textContent=`현재: ${piece(index===max?null:f.current)}`;$('current-label').textContent=index===max?'플레이 종료':'현재 미노';drawCurrent(index===max?null:f.current);
$('next').replaceChildren(...(f.next||f.next_queue||[]).map(v=>{const el=document.createElement('span');el.className='piece';el.textContent=piece(v);return el;}));
$('path').textContent=index?data.actions[index-1].inputs.join(' → '):'시작 보드';$('slider').value=index;$('position').textContent=`${index} / ${max}`;
$('play').textContent=playing?'일시 정지':'재생';
$('timing-note').textContent=timingMode==='recorded'?`기록된 게임 시간의 ${playbackRate}배 속도 · 조각마다 원래 입력 간격을 유지합니다.`:`보기용 ${playbackRate}배 속도 · 조각당 ${(600/playbackRate/1000).toFixed(2)}초${hasRecordedTiming?'':' · 원래 시간 기록이 없습니다.'}`;}
function stepDuration(position=index){return timingMode==='recorded'?recordedDelays[position]:600;}
function advance(now){if(!playing){last=now;return false;}elapsed+=Math.max(0,now-last)*playbackRate;last=now;let changed=false;
while(index<max&&elapsed+1e-7>=stepDuration()){elapsed=Math.max(0,elapsed-stepDuration());index++;changed=true;}
if(index===max){playing=false;elapsed=0;}return changed;}
function seek(value){index=Math.max(0,Math.min(max,Math.round(Number(value))));elapsed=0;last=performance.now();if(index===max)playing=false;draw();}
$('slider').addEventListener('input',e=>seek(e.target.value));$('back').onclick=()=>seek(index-1);$('forward').onclick=()=>seek(index+1);
$('play').onclick=()=>{const now=performance.now();if(playing){advance(now);playing=false;}else{if(index===max){index=0;elapsed=0;}playing=max>0;}last=now;draw();};
$('speed').addEventListener('change',()=>{const now=performance.now();advance(now);playbackRate=Number($('speed').value);last=now;draw();});
$('timing-mode').addEventListener('change',()=>{const now=performance.now();advance(now);const fraction=index<max?elapsed/stepDuration():0;
timingMode=$('timing-mode').value==='recorded'&&hasRecordedTiming?'recorded':'view';$('timing-mode').value=timingMode;elapsed=index<max?fraction*stepDuration():0;last=now;draw();});
$('original-speed').onclick=()=>{if(!hasRecordedTiming)return;if(index===max)index=0;timingMode='recorded';playbackRate=1;elapsed=0;last=performance.now();playing=true;$('timing-mode').value='recorded';$('speed').value='1';draw();};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.key==='ArrowRight')seek(index+1);else if(e.key==='ArrowLeft')seek(index-1);else if(e.code==='Space'){e.preventDefault();$('play').click();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){advance(performance.now());playing=false;draw();}});
function animate(now){if(advance(now))draw();requestAnimationFrame(animate);}draw();requestAnimationFrame(animate);
</script></body></html>'''


def build_html(replay_path: str | Path, output_path: str | Path | None = None) -> Path:
    replay_path = Path(replay_path)
    output_path = Path(output_path) if output_path else replay_path.with_suffix(".html")
    data = json.loads(replay_path.read_text(encoding="utf-8-sig"))
    if not data.get("frames") or len(data.get("actions", [])) != len(data["frames"]) - 1:
        raise ValueError("Replay must have one more public frame than input actions")
    # Escaping '<' prevents embedded user-controlled text from closing the script tag.
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(TEMPLATE.replace("__REPLAY_DATA__", encoded), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_html(args.replay, args.output).resolve())


if __name__ == "__main__":
    main()

# Capture Omok

따내기 있는 오목. 창에서 직접 두거나 봇끼리 붙여 놓고 구경할 수 있음.

## 규칙

- 19x19, 흑 선
- 딱 5개만 승리. 6목 이상은 안 쳐줌
- 내 돌 / 상대 돌 2개 / 내 돌 모양이 되면 가운데 2개를 따냄. 8방향 다 됨
- 따내서 상대 6목이 5목이 되면 상대 승. 한 수에 양쪽 다 5목이 생기면 무승부
- 시간은 한 판 전체 기준이고, 다 쓰면 짐

## 실행

Python 3.8 이상이면 됨. 창은 tkinter라서 따로 깔 건 없음.
윈도우는 `play.bat` 더블클릭하면 메뉴 뜸.

```
python game/omok.py              # 메뉴
python game/omok.py 0 5 1        # 나(흑) vs Smart(백), 1분씩
python game/omok.py 5 2 1        # Smart vs AlphaBeta 구경
```

| 번호 | 플레이어 |
|---|---|
| 0 | 사람 |
| 1 | Random |
| 2 | AlphaBeta |
| 3 | Threat |
| 4 | MCTS |
| 5 | Smart (제일 셈) |

## Smart 봇

`agent/smartplayer.py`, `agent/fastcore.py` 두 파일. numba가 깔려 있어야 제대로 돌아감.
없어도 돌긴 하는데 훨씬 느려서 많이 약해짐.

```
pip install -r requirements.txt     # 윈도우는 install-numba.bat 더블클릭
```

- 처음 켤 때 컴파일하느라 1분 정도 걸림. 그다음부터는 바로 뜸
- 스레드 수는 코어 수 보고 알아서 정함. 바꾸려면 `OMOK_THREADS`
- 같은 상대랑 연달아 두면 앞 판을 기억함. 이긴 판은 그대로 다시 두고, 진 판은 갈라진 자리에서 다른 수를 찾음.
  기록은 `agent/omok_memory.json`에 남고 3시간 지나면 잊음. 끄려면 `OMOK_MEMORY=off`

## 자기 봇 붙여 보기

`game/player.py`의 `Player`를 상속해서 `take_turn(board, time)`에서 `(행, 열)`을 리턴하면 됨.
`board`는 19x19 리스트(-1 빈칸, 0 흑, 1 백), `time`은 남은 시간(ms).

창에서 한 판:

```
python game/omok.py 경로/mybot.py:MyBot 5 1
```

창 없이 8판(흑백 번갈아 가면서):

```
python tools/arena.py 경로/mybot.py:MyBot smart --games 8 --minutes 1
```

흑이 엄청 유리한 게임이라 한두 판으로는 아무것도 모름. 최소 8판은 돌려 봐야 함.

arena는 기본으로 Smart의 기억을 끄고 돌림. 실제 대결처럼 기억 켜고 하려면 `agent/omok_memory.json`을 지우고:

```
set OMOK_MEMORY=on
python tools/arena.py 경로/mybot.py:MyBot smart --games 8 --minutes 1
```

맥/리눅스는 `OMOK_MEMORY=on python tools/arena.py ...`

## 대결 전에 체크

- numba 깔기 (`install-numba.bat`)
- 컴파일 때문에 그 컴퓨터에서 미리 한 판 돌려 두기
- 첫 판 전에 `agent/omok_memory.json` 지우기. 미리 돌린 판 기록이 남아 있음
- 8판은 같은 컴퓨터에서 이어서 두기. 경기 중에는 기록 파일 건드리지 않기

## 테스트

```
python tests/test_rules.py
python tests/test_game.py
python tests/test_agent.py
```

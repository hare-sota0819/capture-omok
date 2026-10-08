# Capture Omok

따내기가 있는 오목입니다. 창에서 직접 두거나, 봇끼리 붙여 놓고 구경할 수 있습니다.

## 규칙

- 19x19, 흑이 먼저 둡니다.
- 정확히 다섯 개가 이어져야 이깁니다. 여섯 개 이상은 오목이 아닙니다.
- 내 돌 - 상대 돌 두 개 - 내 돌 모양이 되면 가운데 두 개를 따냅니다(8방향).
- 따내기 때문에 상대의 여섯 줄이 다섯 줄이 되면 상대가 이깁니다. 한 수로 양쪽에 오목이 생기면 무승부입니다.
- 시간은 한 판 전체에 대해 각자 주어지고, 다 쓰면 집니다.

## 실행

Python 3.8 이상이면 됩니다. 창은 tkinter로 그리기 때문에 따로 설치할 것은 없습니다.
Windows에서는 `play.bat`을 더블클릭하면 메뉴가 열립니다.

```
python game/omok.py                # 메뉴
python game/omok.py 0 5 1          # 사람(흑) 대 Smart(백), 각자 1분
python game/omok.py 5 2 1          # Smart 대 AlphaBeta 구경
```

| 코드 | 상대 |
|---|---|
| 0 | 사람 |
| 1 | Random: 돌 근처 아무 데나 |
| 2 | AlphaBeta: 알파베타 탐색 |
| 3 | Threat: 위협 수순 위주 |
| 4 | MCTS: 몬테카를로 트리 탐색 |
| 5 | Smart: 제일 센 것 (`agent/`) |

## Smart

`agent/smartplayer.py`와 `agent/fastcore.py` 두 파일입니다. numba가 있으면 컴파일된 엔진으로 돌고,
없으면 같은 방식의 순수 Python 엔진으로 돕니다(훨씬 느려서 그만큼 약합니다).

```
pip install -r requirements.txt    # Windows는 install-numba.bat
```

- 처음 한 번은 컴파일하느라 1분쯤 걸립니다. 그 뒤로는 캐시를 씁니다.
- 탐색 스레드 수는 코어 수에 맞춰 정해지고, 환경 변수 `OMOK_THREADS`로 바꿀 수 있습니다.
- 같은 상대와 여러 판을 이어서 둘 때는 앞 판을 기억합니다. 이긴 수순은 다시 두고, 진 수순은 갈라진 자리에서 다른 수를 찾습니다.
  기록은 `agent/omok_memory.json`에 남고 3시간이 지나면 잊습니다. 끄려면 환경 변수 `OMOK_MEMORY=off`.

## 내 봇 붙이기

`player.py`의 `Player`를 상속해서 `take_turn(board, time)`이 `(행, 열)`을 돌려주면 됩니다.
`board`는 19x19 리스트(-1 빈칸, 0 흑, 1 백), `time`은 남은 시간(ms)입니다.

```
python game/omok.py path/to/mybot.py:MyBot 5 1
python tools/arena.py path/to/mybot.py:MyBot smart --games 8 --minutes 1
```

`tools/arena.py`는 창 없이 여러 판을 돌리고 승패, 시간패, 반칙 수를 알려 줍니다.
흑백은 판마다 바뀝니다. 흑이 많이 유리한 게임이라 한 판만 보고 판단하면 안 됩니다.

## 테스트

```
python tests/test_rules.py
python tests/test_game.py
python tests/test_agent.py
```

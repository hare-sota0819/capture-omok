Capture Omok

처음에 install-numba.bat 한 번 실행해야 함. 안 하면 Smart 봇이 엄청 느려짐.

게임은 play.bat 더블클릭하면 켜짐. 메뉴에서 흑이랑 백 고르고 시작하면 됨.

명령어로 바로 켜려면 python game/omok.py 다음에 흑, 백, 시간 순서로 쓰면 됨. 번호는 0 사람, 1 Random, 2 AlphaBeta, 3 Threat, 4 MCTS, 5 Smart이고 시간은 분 단위.

python game/omok.py 0 5 1

이렇게 하면 내가 흑, Smart가 백이고 1분씩임.

Smart는 처음 켤 때 1분 정도 멈춰 있는데 컴파일하는 거라 그냥 기다리면 됨. 두 번째부터는 바로 뜸.

자기 봇이랑 붙여 보려면 game/player.py에 있는 Player를 상속해서 take_turn(board, time)이 (행, 열)을 리턴하게 만들면 됨. board는 19x19 리스트고 빈칸 -1, 흑 0, 백 1. time은 남은 시간(ms).

창에서 붙이려면 번호 자리에 파일 경로랑 클래스 이름을 쓰면 됨.

python game/omok.py mybot.py:MyBot 5 1

창 없이 여러 판 돌리려면 이렇게 하면 됨. 흑백은 판마다 바뀌고 끝나면 승패 알려줌. 흑이 많이 유리해서 최소 8판은 돌려봐야 의미 있음.

python tools/arena.py mybot.py:MyBot smart --games 8 --minutes 1

Smart는 같은 상대랑 연달아 두면 앞 판을 기억하는데 arena에서는 기본으로 꺼져 있음. 켜고 하려면 agent 폴더에 있는 omok_memory.json을 지우고 set OMOK_MEMORY=on 입력한 다음 arena 돌리면 됨. 제대로 붙을 때도 첫 판 전에 이 파일은 지우고 시작하는 게 좋음.

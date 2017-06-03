MATRIX = 5
PITCH_LOCATION = "| " + "{:^6s} | " * MATRIX  # 투구 영역
PITCH_LOCATION = (PITCH_LOCATION + '\n') * MATRIX
PITCH_LOCATION = "---------" * MATRIX + "\n" + PITCH_LOCATION + "---------" * MATRIX
print(PITCH_LOCATION.format(*[str(idx) for idx in range(26)]))  # 투구 영역 5 * 5 출력


print(*[str(idx) for idx in range(26)])
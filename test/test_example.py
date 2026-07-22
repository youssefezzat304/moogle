def test_eueler_formula():
    N = 10
    compute = 0
    for i in range(N):
        compute += i
    answer = N * (N - 1) // 2

    assert compute == answer

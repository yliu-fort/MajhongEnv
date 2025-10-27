import sys, os, random, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "ext"))

from _shanten_dp import compute_ukeire_advanced as _f0
from _shanten_dp import compute_all_discards_ukeire_fast as _f1
try:
    from shanten_dp_cy import compute_ukeire_advanced, compute_all_discards_ukeire_fast
    CYTHON_AVAILABLE = True
except Exception:
    compute_ukeire_advanced = _f0
    compute_all_discards_ukeire_fast = _f1
    CYTHON_AVAILABLE = False

# --------- 工具函数 ---------
def random_hand14():
    """随机生成 14 张牌的 34 进制计数（每张<=4）。"""
    counts = [0]*34
    for _ in range(14):
        while True:
            t = random.randrange(34)
            if counts[t] < 4:
                counts[t] += 1
                break
    return counts

def random_last_draw(counts):
    """按张数加权在手牌中随机选一个 last_draw 索引。"""
    pool = []
    for t, c in enumerate(counts):
        if c > 0:
            pool.extend([t]*c)
    return random.choice(pool)

def remaining_from_counts(counts):
    """剩余牌山：简单设为 4 - 手牌数（裁剪到>=0）。"""
    return [max(0, 4 - c) for c in counts]

# --------- 测试主体 ---------
def run_tests(n=1000, seed=42):
    random.seed(seed)
    # 预生成测试用例，避免把生成时间算进两边实现的计时
    tests = []
    for _ in range(n):
        hand = random_hand14()
        last_draw = random_last_draw(hand)
        remaining = remaining_from_counts(hand)
        tests.append((hand, last_draw, remaining))

    # 两套函数句柄：cy（可能就是 py）和 py（基准）
    cy_adv = compute_ukeire_advanced
    cy_dis = compute_all_discards_ukeire_fast
    py_adv = _f0
    py_dis = _f1

    # --- 测试 compute_ukeire_advanced ---
    # 计时：Cython/当前实现
    t0 = time.perf_counter()
    cy_adv_out = []
    for hand, last_draw, remaining in tests:
        cy_adv_out.append(cy_adv(hand[:], last_draw, remaining[:]))
    t1 = time.perf_counter()

    # 计时：Python 基准
    t2 = time.perf_counter()
    py_adv_out = []
    for hand, last_draw, remaining in tests:
        py_adv_out.append(py_adv(hand[:], last_draw, remaining[:]))
    t3 = time.perf_counter()

    # 对比：按 (shanten, ukeire) 判断是否一致
    mismatches_adv = 0
    for a, b in zip(cy_adv_out, py_adv_out):
        if (a.get("shanten"), a.get("ukeire")) != (b.get("shanten"), b.get("ukeire")):
            mismatches_adv += 1

    # --- 测试 compute_all_discards_ukeire_fast ---
    t4 = time.perf_counter()
    cy_dis_out = []
    for hand, _, remaining in tests:
        cy_dis_out.append(cy_dis(hand[:], remaining[:]))
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    py_dis_out = []
    for hand, _, remaining in tests:
        py_dis_out.append(py_dis(hand[:], remaining[:]))
    t7 = time.perf_counter()

    # 对比：整个 34 位数组完全一致才算匹配
    mismatches_dis_hands = 0          # 有任意位置不一致的手数
    mismatches_dis_positions = 0      # 全部手牌中逐位不一致的总数
    for (cy_sh, cy_uk), (py_sh, py_uk) in zip(cy_dis_out, py_dis_out):
        hand_has_mismatch = False
        for i in range(34):
            if cy_sh[i] != py_sh[i] or cy_uk[i] != py_uk[i]:
                mismatches_dis_positions += 1
                hand_has_mismatch = True
        if hand_has_mismatch:
            mismatches_dis_hands += 1

    # 输出
    print("====== 随机测试汇总 ======")
    print(f"样本数: {n}, 随机种子: {seed}")
    print(f"Cython 可用: {CYTHON_AVAILABLE}")
    print("--- compute_ukeire_advanced ---")
    print(f"Cython/当前实现总耗时: {(t1 - t0)*1000:.2f} ms")
    print(f"Python 基准总耗时   : {(t3 - t2)*1000:.2f} ms")
    print(f"不匹配手数（按 shanten+ukeire）: {mismatches_adv}/{n}")
    print("--- compute_all_discards_ukeire_fast ---")
    print(f"Cython/当前实现总耗时: {(t5 - t4)*1000:.2f} ms")
    print(f"Python 基准总耗时   : {(t7 - t6)*1000:.2f} ms")
    print(f"不匹配手数（任一位置不同）: {mismatches_dis_hands}/{n}")
    print(f"不匹配逐位总计（34*N）   : {mismatches_dis_positions}/{34*n}")

if __name__ == "__main__":
    run_tests(n=1000, seed=42)
    run_tests(n=10000, seed=42)
    run_tests(n=100000, seed=42)

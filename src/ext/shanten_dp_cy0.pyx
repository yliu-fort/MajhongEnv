# cython: boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
"""
Mahjong 向听/受入 计算（Cython 版）
移植自 shanten_dp.py，接口保持一致：
- compute_ukeire_advanced(hand, last_draw34, remaining) -> dict
- compute_all_discards_ukeire_fast(counts, remaining) -> (shantens, ukeires)

修复：
- 移除了 cpdef 函数体内的闭包/生成器表达式（Cython 不支持）。
- 将内部小函数提为模块级 cdef 函数；将生成器表达式改为显式 for 循环。
"""

from functools import lru_cache
cimport cython

# =========================
# 工具 & 常量
# =========================

SUIT_CACHE_MAX: int = 500_000
HONOR_CACHE_MAX: int = 100_000

@cython.cfunc
@cython.inline
cdef int _is_same_suit_c(int a, int b):
    cdef int la, ha, lb, hb
    la, ha = (0, 8) if a < 9 else ((9, 17) if a < 18 else ((18, 26) if a < 27 else (27, 33)))
    lb, hb = (0, 8) if b < 9 else ((9, 17) if b < 18 else ((18, 26) if b < 27 else (27, 33)))
    return 1 if la == lb else 0

@cython.cfunc
@cython.inline
cdef tuple _suit_range(int tile):
    """返回该 tile 所在花色的 [lo, hi] 以及是否字牌"""
    if tile < 9:
        return 0, 8, False
    elif tile < 18:
        return 9, 17, False
    elif tile < 27:
        return 18, 26, False
    else:
        return 27, 33, True

@cython.cfunc
@cython.inline
cdef int _calc_shanten_from_mtp(int m_total, int t_total, bint pair_used):
    cdef int t_eff = t_total
    if 4 - m_total < t_total:
        t_eff = 4 - m_total
    return 8 - 2*m_total - t_eff - (1 if pair_used else 0)

# =========================
# 花色DP：返回帕累托前沿的 (m,t,p) 状态集合
# =========================

@cython.cfunc
cdef list _pareto_prune(list states):
    """去掉被支配的状态：m、t、p三个维度全不劣且至少一维更优的，保留前沿"""
    cdef list kept = []
    cdef int i, j
    # 粗排减少比较
    states = list(states)
    states.sort(reverse=True)
    for i in range(len(states)):
        m, t, p = states[i]
        dominated = False
        for j in range(len(kept)):
            m2, t2, p2 = kept[j]
            if m2 >= m and t2 >= t and p2 >= p and (m2 > m or t2 > t or p2 > p):
                dominated = True
                break
        if not dominated:
            kept.append((m, t, p))
    return kept

@lru_cache(maxsize=SUIT_CACHE_MAX)
def _enumerate_suit_states(counts_tuple):
    """
    对一个花色(9格)计数tuple，枚举所有 (m,t,p)：
      m: 面子数（刻/顺）
      t: 搭子数（12、23、13 这些两张搭子；不含对子）
      p: 对子数（用于后续合并时决定是否拿一个作雀头，其余对子可视为搭子）
    返回帕累托前沿集合 list[(m,t,p)]
    """
    counts = list(counts_tuple)

    @lru_cache(maxsize=None)
    def dfs(state_tuple):
        c = list(state_tuple)
        # 找到第一个非0位置
        i = -1
        for k in range(9):
            if c[k] > 0:
                i = k
                break
        if i == -1:
            return {(0,0,0)}  # 空
        res = set()

        # 1) 刻子
        if c[i] >= 3:
            c[i] -= 3
            for m,t,p in dfs(tuple(c)):
                res.add((m+1, t, p))
            c[i] += 3

        # 2) 顺子
        if i <= 6 and c[i+1] > 0 and c[i+2] > 0:
            c[i] -= 1; c[i+1] -= 1; c[i+2] -= 1
            for m,t,p in dfs(tuple(c)):
                res.add((m+1, t, p))
            c[i] += 1; c[i+1] += 1; c[i+2] += 1

        # 3) 两面/边 & 嵌张搭子
        if i <= 7 and c[i+1] > 0:
            c[i] -= 1; c[i+1] -= 1
            for m,t,p in dfs(tuple(c)):
                res.add((m, t+1, p))
            c[i] += 1; c[i+1] += 1
        if i <= 6 and c[i+2] > 0:
            c[i] -= 1; c[i+2] -= 1
            for m,t,p in dfs(tuple(c)):
                res.add((m, t+1, p))
            c[i] += 1; c[i+2] += 1

        # 4) 对子
        if c[i] >= 2:
            c[i] -= 2
            for m,t,p in dfs(tuple(c)):
                res.add((m, t, p+1))
            c[i] += 2

        # 5) 丢单张
        c[i] -= 1
        for m,t,p in dfs(tuple(c)):
            res.add((m, t, p))
        c[i] += 1

        return frozenset(_pareto_prune(list(res)))

    return list(dfs(tuple(counts)))

@lru_cache(maxsize=HONOR_CACHE_MAX)
def _enumerate_honor_states(counts_tuple):
    """
    字牌(7格)版本：没有顺子，只有刻子/对子/单张
    返回帕累托前沿 list[(m,t,p)]，t来源于“未作为雀头的对子”可当搭子
    """
    counts = list(counts_tuple)

    @lru_cache(maxsize=None)
    def dfs(state_tuple, idx):
        if idx == 7:
            return {(0,0,0)}
        c = list(state_tuple)
        x = c[idx]
        res = set()

        # 刻子
        if x >= 3:
            c[idx] -= 3
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m+1, t, p))
            c[idx] += 3

        # 对子
        if x >= 2:
            c[idx] -= 2
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m, t, p+1))
            c[idx] += 2

        # 用掉1张 / 0张跳过
        if x >= 1:
            c[idx] -= 1
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m, t, p))
            c[idx] += 1
        else:
            for m,t,p in dfs(tuple(c), idx+1):
                res.add((m, t, p))

        return frozenset(_pareto_prune(list(res)))

    return list(dfs(tuple(counts), 0))

cdef tuple _combine_four_groups(list ms_states, list ps_states, list ss_states, list z_states):
    """
    合并4类的帕累托，返回 (best_sh, cache)
    cache 包含：除去某一花色后的合并前缀，便于只重算单花色时快速拼回
    """
    def merge(A, B):
        tmp = {}
        for m1,t1,p1 in A:
            for m2,t2,p2 in B:
                m = m1+m2; t = t1+t2; p = p1+p2
                pair_used = 1 if p>0 else 0
                t_eff = t + (p - pair_used if p - pair_used > 0 else 0)
                sh = _calc_shanten_from_mtp(m if m < 4 else 4, t_eff, pair_used)
                key = (m,t,p)
                if key not in tmp or sh < tmp[key]:
                    tmp[key] = sh
        return list(tmp.keys())

    mp = merge(ms_states, ps_states)
    mps = merge(mp, ss_states)
    mpsz = merge(mps, z_states)

    # 计算全局最小普通手向听
    cdef int best_sh = 10**9
    for m,t,p in mpsz:
        pair_used = 1 if p>0 else 0
        t_eff = t + (p - pair_used if p - pair_used > 0 else 0)
        sh = _calc_shanten_from_mtp(m if m < 4 else 4, t_eff, pair_used)
        if sh < best_sh:
            best_sh = sh

    def merge_pair(A, B):
        res = set()
        for m1,t1,p1 in A:
            for m2,t2,p2 in B:
                res.add((m1+m2, t1+t2, p1+p2))
        return list(res)

    mp_cache = merge(ms_states, ps_states)
    ps_cache = merge(ps_states, ss_states)
    ms_cache = merge(ms_states, ss_states)

    mps_cache = merge_pair(mp_cache, ss_states)
    psm_cache = merge_pair(ps_cache, ms_states)
    msp_cache = merge_pair(ms_cache, ps_states)

    other_than_m = merge_pair(ps_cache, z_states)
    other_than_p = merge_pair(ms_cache, z_states)
    other_than_s = merge_pair(mp_cache, z_states)

    return best_sh, {
        "mpsz": mpsz,
        "other_than_m": other_than_m,
        "other_than_p": other_than_p,
        "other_than_s": other_than_s,
        "z_only": z_states,
        "mp_cache": mp_cache,
        "ps_cache": ps_cache,
        "ms_cache": ms_cache
    }

@cython.cfunc
@cython.inline
cdef int _best_sh_from_states(object states):
    cdef int best = 10**9
    cdef int m, t, p
    cdef int pair_used, t_eff, sh
    for m,t,p in states:
        pair_used = 1 if p>0 else 0
        t_eff = t + (p - pair_used if p - pair_used > 0 else 0)
        sh = _calc_shanten_from_mtp(m if m < 4 else 4, t_eff, pair_used)
        if sh < best:
            best = sh
    return best

# =========================
# 七对 & 国士：向听与改良集合
# =========================

@cython.cfunc
def _chiitoi_shanten_and_improves(list hand, list remaining):
    cdef int pairs = 0
    cdef int singles = 0
    cdef int t
    for t in range(34):
        if hand[t] >= 2:
            pairs += 1
        elif hand[t] == 1:
            singles += 1
    cdef int distinct = pairs + singles
    cdef int sh = 6 - pairs + (7 - distinct if 7 - distinct > 0 else 0)

    improve = set()
    if sh <= 6:
        for t in range(34):
            if hand[t] == 1 and remaining[t] > 0 and hand[t] < 2:
                improve.add(t)
        if distinct < 7:
            for t in range(34):
                if hand[t] == 0 and remaining[t] > 0:
                    improve.add(t)
    return sh, improve

cdef set _TERMINALS_AND_HONORS = set([0,8,9,17,18,26] + list(range(27,34)))

@cython.cfunc
def _kokushi_shanten_and_improves(list hand, list remaining):
    cdef int have = 0
    cdef int pair = 0
    cdef int t
    need_set = []
    for t in _TERMINALS_AND_HONORS:
        if hand[t] > 0:
            have += 1
            if hand[t] >= 2:
                pair = 1
        else:
            need_set.append(t)
    cdef int sh = 13 - have - pair

    improve = set()
    if sh >= 0:
        if have < 13:
            for t in need_set:
                if remaining[t] > 0:
                    improve.add(t)
        if pair == 0:
            for t in _TERMINALS_AND_HONORS:
                if hand[t] > 0 and remaining[t] > 0 and hand[t] < 4:
                    improve.add(t)
    return sh, improve

# =========================
# 普通手：一次性DP→base，增量只改单花色→是否降向听
# =========================

@cython.cfunc
def _normal_base_and_cache(list hand):
    m_cnts = tuple(hand[0:9])
    p_cnts = tuple(hand[9:18])
    s_cnts = tuple(hand[18:27])
    z_cnts = tuple(hand[27:34])

    ms = _enumerate_suit_states(m_cnts)
    ps = _enumerate_suit_states(p_cnts)
    ss = _enumerate_suit_states(s_cnts)
    zs = _enumerate_honor_states(z_cnts)

    base_sh, cache = _combine_four_groups(ms, ps, ss, zs)
    cache["ms"] = ms; cache["ps"] = ps; cache["ss"] = ss; cache["zs"] = zs
    return base_sh, cache

@cython.cfunc
def _neighbors_for_suit_index(int x):
    cdef set cand = {x}
    if x-1 >= 0: cand.add(x-1)
    if x-2 >= 0: cand.add(x-2)
    if x+1 <= 8: cand.add(x+1)
    if x+2 <= 8: cand.add(x+2)
    return cand

@cython.cfunc
def _normal_candidate_tiles(list hand, list remaining):
    cdef set cand = set()
    cdef int base, i, c, j, t
    for base in (0,9,18):
        for i in range(9):
            c = hand[base+i]
            if c == 0:
                continue
            for j in _neighbors_for_suit_index(i):
                t = base + j
                if hand[t] < 4 and remaining[t] > 0:
                    cand.add(t)
    for t in range(27,34):
        if hand[t] > 0 and hand[t] < 4 and remaining[t] > 0:
            cand.add(t)
    return cand

@cython.cfunc
def _recompute_normal_with_one_tile_added(list hand, dict cache, int add_tile):
    lo, hi, is_honor = _suit_range(add_tile)
    if is_honor:
        z_cnts = list(hand[27:34])
        z_cnts[add_tile-27] += 1
        zs2 = _enumerate_honor_states(tuple(z_cnts))
        mps_cache = cache.get("other_than_z", None)
        if mps_cache is None:
            mps_cache = []
            for a in _combine_four_groups(cache["ms"], cache["ps"], cache["ss"], [(0,0,0)])[1]["mpsz"]:
                mps_cache.append(a)
            cache["other_than_z"] = mps_cache
        merged = []
        for m1,t1,p1 in mps_cache:
            for m2,t2,p2 in zs2:
                merged.append((m1+m2, t1+t2, p1+p2))
        return _best_sh_from_states(merged)

    base = lo
    arr = list(hand[lo:hi+1])
    arr[add_tile-lo] += 1
    states2 = _enumerate_suit_states(tuple(arr))

    if base == 0:
        other = cache["other_than_m"]
    elif base == 9:
        other = cache["other_than_p"]
    else:
        other = cache["other_than_s"]

    merged = []
    for m1,t1,p1 in states2:
        for m2,t2,p2 in other:
            merged.append((m1+m2, t1+t2, p1+p2))
    return _best_sh_from_states(merged)

# 新增：将原先 cpdef 内部的辅助函数提到模块级，避免闭包
cdef tuple _enum_after_minus_one(int tile, list counts, list ms0, list ps0, list ss0, list zs0, dict minus1_cache):
    lo, hi, is_honor = _suit_range(tile)
    if is_honor:
        arr = list(counts[27:34]); arr[tile-27] -= 1
        key = ('z', tile-27, tuple(arr))
        if key not in minus1_cache:
            minus1_cache[key] = _enumerate_honor_states(tuple(arr))
        zs_minus1 = minus1_cache[key]
        return ms0, ps0, ss0, zs_minus1
    base = lo
    arr2 = list(counts[lo:hi+1]); arr2[tile-lo] -= 1
    key2 = ({0:'m',9:'p',18:'s'}[base], tile-lo, tuple(arr2))
    if key2 not in minus1_cache:
        minus1_cache[key2] = _enumerate_suit_states(tuple(arr2))
    states_minus1 = minus1_cache[key2]
    if base == 0:
        return states_minus1, ps0, ss0, zs0
    elif base == 9:
        return ms0, states_minus1, ss0, zs0
    else:
        return ms0, ps0, states_minus1, zs0

# =========================
# 主入口
# =========================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict compute_ukeire_advanced(list hand, object last_draw34, list remaining):
    """
    进阶版受入计算（普通/七对/国士三合一），只对普通手做“邻域候选+单花色重算”。
    """
    hand = hand[:]  # 复制以免原地污染
    if last_draw34 is not None:
        hand[<int>last_draw34] -= 1

    normal_sh, cache = _normal_base_and_cache(hand)

    if "other_than_z" not in cache:
        mps_cache = []
        for a in _combine_four_groups(cache["ms"], cache["ps"], cache["ss"], [(0,0,0)])[1]["mpsz"]:
            mps_cache.append(a)
        cache["other_than_z"] = mps_cache

    chiitoi_sh, chiitoi_improve = _chiitoi_shanten_and_improves(hand, remaining)
    kokushi_sh, kokushi_improve = _kokushi_shanten_and_improves(hand, remaining)

    cdef int base_sh_global = normal_sh if normal_sh <= chiitoi_sh and normal_sh <= kokushi_sh else (chiitoi_sh if chiitoi_sh <= kokushi_sh else kokushi_sh)

    normal_cands = _normal_candidate_tiles(hand, remaining)

    improve_tiles = set()
    if chiitoi_sh == base_sh_global:
        improve_tiles |= chiitoi_improve
    if kokushi_sh == base_sh_global:
        improve_tiles |= kokushi_improve
    if normal_sh == base_sh_global:
        for t in normal_cands:
            if hand[t] >= 4 or remaining[t] <= 0:
                continue
            if _recompute_normal_with_one_tile_added(hand, cache, t) < normal_sh:
                improve_tiles.add(t)

    # 替换生成器表达式为显式循环，避免在 cpdef 中创建闭包
    cdef list tiles_list = []
    cdef int _t
    for _t in range(34):
        if _t in improve_tiles and remaining[_t] > 0:
            tiles_list.append((_t, remaining[_t]))
    tiles_list.sort()  # 仅按牌序排序即可

    cdef int ukeire = 0
    for _, cnt in tiles_list:
        ukeire += cnt
    mode = "normal" if base_sh_global == normal_sh else ("chiitoi" if base_sh_global == chiitoi_sh else "kokushi")

    return {
        "shanten": int(base_sh_global - (((14 - sum(hand))//3)*2)),
        "ukeire": int(ukeire),
        "tiles": tiles_list,
        "explain": {"best_mode": mode,
                    "shanten_regular": int(normal_sh),
                    "shanten_chiitoi": int(chiitoi_sh),
                    "shanten_kokushi": int(kokushi_sh)}
    }

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple compute_all_discards_ukeire_fast(list counts, list remaining):
    """
    对每个可弃的 tile，计算“弃掉该张后”的整手最小向听与受入。
    返回 (shantens, ukeires)，均为长度34的数组。
    """
    cdef int NUM_TILES = 34
    cdef list shantens = [8] * NUM_TILES
    cdef list ukeires = [0] * NUM_TILES
    cdef int ukeire = 0

    ms0 = _enumerate_suit_states(tuple(counts[0:9]))
    ps0 = _enumerate_suit_states(tuple(counts[9:18]))
    ss0 = _enumerate_suit_states(tuple(counts[18:27]))
    zs0 = _enumerate_honor_states(tuple(counts[27:34]))

    minus1_cache = {}

    cdef int tile, cnt
    for tile in range(NUM_TILES):
        cnt = counts[tile]
        if cnt <= 0:
            continue

        h13 = counts[:]
        h13[tile] -= 1

        # 使用模块级辅助函数，避免 cpdef 内部闭包
        ms, ps, ss, zs = _enum_after_minus_one(tile, counts, ms0, ps0, ss0, zs0, minus1_cache)

        normal_sh, cache = _combine_four_groups(ms, ps, ss, zs)
        cache["ms"], cache["ps"], cache["ss"], cache["zs"] = ms, ps, ss, zs

        if "other_than_z" not in cache:
            cache["other_than_z"] = list(_combine_four_groups(ms, ps, ss, [(0,0,0)])[1]["mpsz"])  

        chiitoi_sh, chiitoi_imp = _chiitoi_shanten_and_improves(h13, remaining)
        kokushi_sh, kokushi_imp = _kokushi_shanten_and_improves(h13, remaining)
        base_sh = normal_sh if normal_sh <= chiitoi_sh and normal_sh <= kokushi_sh else (chiitoi_sh if chiitoi_sh <= kokushi_sh else kokushi_sh)

        improves = set()
        if chiitoi_sh == base_sh:
            improves |= chiitoi_imp
        if kokushi_sh == base_sh:
            improves |= kokushi_imp
        if normal_sh == base_sh:
            for t2 in _normal_candidate_tiles(h13, remaining):
                if h13[t2] >= 4 or remaining[t2] <= 0:
                    continue
                if _recompute_normal_with_one_tile_added(h13, cache, t2) < normal_sh:
                    improves.add(t2)

        ukeire = 0
        for t2 in improves:
            if remaining[t2] > 0:
                ukeire += remaining[t2]
        correction = ((14 - sum(h13)) // 3) * 2
        shantens[tile] = int(base_sh - correction)
        ukeires[tile] = int(ukeire)

    return shantens, ukeires
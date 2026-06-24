"""C2K (Cargo 2000) 延误尾审计 —— service-level CVaR 的真实数据锚。

数据来源（不随本仓分发，CC BY 4.0）:
  UCI ML Repository id 382 "Cargo 2000 Freight Tracking and Tracing" (Metzger et al.).
  本地副本: data/raw/c2k/c2k_data_comma.csv
  3943 实例 × 98 列；每条 leg 的 RCS->DEP->RCF->DLV 各有 planned(_p)/effective(_e) 分钟时长。

这个脚本证明什么 / 不证明什么:
  - 证明 (motivation): 真实货运履约延误极右偏——均值/中位是"提前"，
    但重延误上尾巨大；mean 掩盖尾，CVaR 才抓得住。=> "为什么用 CVaR 而非 mean"的真实锚。
  - 不证明 (validation): C2K 无容量/航班映射、无到达流、无收入，
    故无法验证在线接拒策略 (ONLINE* vs FLOOR) 或 revenue-CVaR。仅服务级、仅事后履约时长。

用法:
  python experiments/c2k_delay_tail.py [path/to/c2k_data_comma.csv]
"""
import csv, sys, statistics as st
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "data/raw/c2k/c2k_data_comma.csv"


def cvar_upper(xs, a):
    """最差（最大）a 分位尾的平均 —— 延误的上尾 CVaR。"""
    k = max(1, int(a * len(xs)))
    return sum(sorted(xs, reverse=True)[:k]) / k


def main(path):
    rows = list(csv.reader(open(path)))
    h, data = rows[0], rows[1:]
    ip, ie = h.index("i1_dlv_p"), h.index("i1_dlv_e")
    delays = []
    for r in data:
        try:
            p, e = float(r[ip]), float(r[ie])
        except ValueError:
            continue  # '?' = 未用腿 / 缺失
        delays.append(e - p)

    n = len(delays)
    late = [d for d in delays if d > 0]
    print(f"数据: {path}")
    print(f"有效 i1_dlv 实例: {n} / {len(data)}")
    print(f"延误 (effective>planned): {len(late)} ({100*len(late)/n:.1f}%)")
    print(f"延误分钟  mean={st.mean(delays):.0f}  median={st.median(delays):.0f}  max={max(delays):.0f}")
    print("service-level delay CVaR (最差尾平均延误, 分钟 / 天):")
    for a in (0.1, 0.2, 0.3):
        c = cvar_upper(delays, a)
        print(f"  CVaR_{a:.1f} = {c:.0f} min  (~{c/1440:.1f} 天)")
    print(f"迟到子集: mean={st.mean(late):.0f} min, CVaR_0.2={cvar_upper(late, 0.2):.0f} min")
    print("=> mean/median 提前, 但重延误尾巨大: CVaR 抓得住, mean 掩盖. (motivation, 非 validation)")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not p.exists():
        sys.exit(f"找不到 C2K 数据: {p}\n（C2K 数据见 data/raw/c2k/）")
    main(p)

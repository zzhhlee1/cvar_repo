#!/usr/bin/env python3
"""从公开真数据提取半合成 CVaR 订舱仿真的校准参数。

把 theory 篇结论落到货运应用,第一步 = 校准。只用本仓 data/raw 的**真公开数据**抽取
仿真器需要的**真参数**;明确标出哪些是 **SYNTHETIC**(公开数据拿不到、须由文献/假设设定)。

scope 红线(见 data/README Modeling Use):货运容量只在**货机**上才真咬——按
AIRCRAFT_CONFIG=2(Freighter)过滤;客机腹舱装载率≈0、风险厌恶 vacuous,不入校准。

真参数(本脚本从 T-100 + FRED 抽):
  - 单航班容量(lbs):货机段 PAYLOAD/DEPARTURES_PERFORMED 分布;
  - contention ρ:货机 route-month 的 freight/payload(=需求/容量利用率)分布;
  - 候选 lane:高货量货运走廊 + 各自 ρ(给相图选 高/中/低 contention 点);
  - regime(P1):FRED 航空货运 revenue-ton-miles(2000–2026)去季节后分 低/正常/高,
    给 π_z + 各 regime 需求乘子 + 分歧比;另出 12 个月季节乘子。
SYNTHETIC 待定(本脚本不臆造,只占位):每单 shipment 重量分布、报价 value/rate、show-up/offload。

运行:uv run python scripts/calibrate_sim_params.py
"""
from __future__ import annotations
import csv, json, statistics as st, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "sim_calibration.json"
FREIGHTER_CONFIG = "2"   # AIRCRAFT_CONFIG: 1=Passenger 2=Freighter 3=Combi ...


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def iter_t100():
    for zp in sorted(RAW.glob("bts_t100_segment_all_carriers_*.zip")):
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            if not names:
                continue
            with zf.open(names[0]) as fh:
                for row in csv.DictReader(line.decode("utf-8-sig") for line in fh):
                    yield row


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


def calibrate_capacity_contention():
    """货机 route-month 聚合 → 单航班容量分布 + contention ρ 分布 + 候选 lane。"""
    rm = defaultdict(lambda: {"dep": 0.0, "payload": 0.0, "freight": 0.0, "mail": 0.0,
                              "distw": 0.0, "city": ("", "")})
    rows_seen = freighter_rows = 0
    for r in iter_t100():
        rows_seen += 1
        if r.get("AIRCRAFT_CONFIG") != FREIGHTER_CONFIG:
            continue
        freighter_rows += 1
        key = (r["ORIGIN"], r["DEST"], int(fnum(r["YEAR"])), int(fnum(r["MONTH"])))
        dep = fnum(r["DEPARTURES_PERFORMED"])
        d = rm[key]
        d["dep"] += dep
        d["payload"] += fnum(r["PAYLOAD"])
        d["freight"] += fnum(r["FREIGHT"])
        d["mail"] += fnum(r["MAIL"])
        d["distw"] += fnum(r["DISTANCE"]) * max(dep, 1.0)
        d["city"] = (r.get("ORIGIN_CITY_NAME", ""), r.get("DEST_CITY_NAME", ""))

    cap_per_flight, rho = [], []           # 真参数分布
    lane = defaultdict(lambda: {"dep": 0.0, "payload": 0.0, "freight": 0.0, "mail": 0.0,
                                "distw": 0.0, "months": 0, "city": ("", "")})
    for (o, dst, y, m), d in rm.items():
        if d["payload"] <= 0 or d["dep"] <= 0:
            continue
        fm = d["freight"] + d["mail"]
        cap_per_flight.append(d["payload"] / d["dep"])
        rho.append(min(fm / d["payload"], 2.0))
        L = lane[(o, dst)]
        for k in ("dep", "payload", "freight", "mail", "distw"):
            L[k] += d[k]
        L["months"] += 1
        L["city"] = d["city"]

    # 候选 lane:按总货量排序,取头部,记 ρ/容量(给相图选 高/中/低 contention)
    lanes = []
    for (o, dst), L in lane.items():
        if L["payload"] <= 0 or L["dep"] <= 0:
            continue
        fm = L["freight"] + L["mail"]
        lanes.append({
            "origin": o, "dest": dst,
            "origin_city": L["city"][0], "dest_city": L["city"][1],
            "rho_contention": round(min(fm / L["payload"], 2.0), 4),
            "capacity_lbs_per_flight": round(L["payload"] / L["dep"], 1),
            "departures_per_month": round(L["dep"] / max(L["months"], 1), 2),
            "avg_freight_mail_per_flight_lbs": round(fm / L["dep"], 1),
            "distance_miles": round(L["distw"] / max(L["dep"], 1.0), 1),
            "total_freight_mail_lbs": round(fm, 1),
            "months_observed": L["months"],
        })
    lanes.sort(key=lambda x: -x["total_freight_mail_lbs"])
    return dict(rows_seen=rows_seen, freighter_rows=freighter_rows,
                route_months=len(cap_per_flight),
                cap_per_flight=cap_per_flight, rho=rho, lanes=lanes)


def calibrate_regime():
    """FRED 航空货运 revenue-ton-miles(SA)→ 去季节后分 低/正常/高 regime + 季节乘子。"""
    f = RAW / "fred_air_revenue_ton_miles_sa.csv"
    series = []   # (year, month, value)
    with f.open(newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        for row in rdr:
            if len(row) < 2 or not row[1] or row[1] == ".":
                continue
            dt = row[0]
            try:
                y, m = int(dt[:4]), int(dt[5:7]); v = float(row[1])
            except ValueError:
                continue
            series.append((y, m, v))
    vals = [v for _, _, v in series]
    overall = st.mean(vals)
    # 季节乘子:month-of-year 均值 / 总均值
    bymonth = defaultdict(list)
    for _, m, v in series:
        bymonth[m].append(v)
    seasonal = {m: round(st.mean(bymonth[m]) / overall, 4) for m in range(1, 13)}
    # 去季节,按 terciles 分 regime(low/normal/high),给 π_z + 需求乘子
    deseason = [v / (st.mean(bymonth[m])) for _, m, v in series]   # 相对各自月份均值
    ds_sorted = sorted(deseason)
    lo_cut, hi_cut = pct(ds_sorted, 1 / 3), pct(ds_sorted, 2 / 3)
    buckets = {"low": [], "normal": [], "high": []}
    for x in deseason:
        z = "low" if x <= lo_cut else ("high" if x > hi_cut else "normal")
        buckets[z].append(x)
    n = len(deseason)
    pi = {z: round(len(b) / n, 4) for z, b in buckets.items()}
    mult = {z: round(st.mean(b), 4) for z, b in buckets.items() if b}
    diverg = round(mult["high"] / mult["low"], 4) if mult.get("low") else None
    return dict(source="FRED AIRRTMFM (air freight+mail revenue ton-miles, SA, 2000-2026)",
                n_months=n, seasonal_multiplier=seasonal,
                regime_pi=pi, regime_demand_multiplier=mult,
                divergence_high_over_low=diverg,
                note="去季节后按 terciles 分；regime 乘子缩放需求强度→contention(P1 隐 regime)")


def main():
    cc = calibrate_capacity_contention()
    reg = calibrate_regime()
    cap, rho = cc["cap_per_flight"], cc["rho"]

    real = {
        "scope": {"filter": "AIRCRAFT_CONFIG=2 (Freighter)", "route_months": cc["route_months"],
                  "freighter_seg_rows": cc["freighter_rows"], "total_seg_rows": cc["rows_seen"],
                  "n_lanes": len(cc["lanes"])},
        "capacity_lbs_per_flight": {"p10": round(pct(cap, .1), 1), "p50": round(pct(cap, .5), 1),
                                     "p90": round(pct(cap, .9), 1), "mean": round(st.mean(cap), 1)},
        "contention_rho_freight_over_payload": {
            "p10": round(pct(rho, .1), 4), "p50": round(pct(rho, .5), 4),
            "p90": round(pct(rho, .9), 4),
            "frac_ge_0.6": round(sum(x >= .6 for x in rho) / len(rho), 4),
            "frac_ge_0.8": round(sum(x >= .8 for x in rho) / len(rho), 4),
            "note": "ρ=已实现 freight/payload=需求对容量利用率;>=0.6 即明显 contention(风险厌恶非 vacuous)"},
        "regime_P1": reg,
        "candidate_lanes_top20": cc["lanes"][:20],
        "contention_lane_picks": {
            "high": [l for l in cc["lanes"][:60] if l["rho_contention"] >= 0.6][:5],
            "mid":  [l for l in cc["lanes"][:60] if 0.3 <= l["rho_contention"] < 0.6][:5],
            "low":  [l for l in cc["lanes"][:60] if l["rho_contention"] < 0.3][:5]},
    }
    synthetic_todo = {
        "_warning": "以下公开数据拿不到,须由文献/IATA/假设设定;仿真时所有此类必须显式标 SYNTHETIC,不得冒充真实。",
        "shipment_size_lbs_dist": "每单 chargeable weight 分布——T-100 是聚合,无每单;占位:对数正态/经验分布,均值由 avg_freight_per_flight/每航班单数 反推。",
        "value_rate_dist": "每单报价 $/kg 或总 value——无免费现货 rate feed;占位:rate tier 或借 DB1B 票价离散度(Gini≈0.42)作 value-分布形状代理(半合成)。",
        "arrival_process": "订舱到达时刻——占位:订舱窗内 Poisson,强度由 lane 月需求×regime 乘子定。",
        "showup_noshow": "show-up/under-tender 率——占位:文献/行业假设(货运 no-show 显著)。",
        "offload_penalty": "甩货罚金 d 相对收益 r* 的倍率——占位:沿用前期 scope(d/r*≈3.5 阈值)。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "repo": "cvar-online-cargo-booking-replication", "purpose": "semi-synthetic CVaR booking sim calibration",
                 "real_source": "BTS T-100 Segment freighter scope + FRED AIRRTMFM"},
        "real_params": real, "synthetic_todo": synthetic_todo,
    }, ensure_ascii=False, indent=2))

    # ---- 控制台摘要 ----
    print("=== 校准:货机 scope(AIRCRAFT_CONFIG=2)===")
    print(f"  段行 {cc['rows_seen']:,} → 货机段 {cc['freighter_rows']:,};货机 route-month {cc['route_months']:,};lane {len(cc['lanes']):,}")
    c = real["capacity_lbs_per_flight"]; print(f"  单航班容量(lbs): p10={c['p10']:,} p50={c['p50']:,} p90={c['p90']:,} mean={c['mean']:,}")
    rr = real["contention_rho_freight_over_payload"]
    print(f"  contention ρ: p50={rr['p50']} p90={rr['p90']} | ρ≥0.6 占{rr['frac_ge_0.6']:.1%} ρ≥0.8 占{rr['frac_ge_0.8']:.1%}")
    print(f"  regime(FRED 去季节): π={reg['regime_pi']} 乘子={reg['regime_demand_multiplier']} 分歧 high/low={reg['divergence_high_over_low']}")
    print(f"  季节乘子(部分): 1月={reg['seasonal_multiplier'][1]} 7月={reg['seasonal_multiplier'][7]} 12月={reg['seasonal_multiplier'][12]}")
    print("  候选 contention lane(高 ρ,前5):")
    for l in real["contention_lane_picks"]["high"]:
        print(f"    {l['origin']}->{l['dest']:<4} ρ={l['rho_contention']} cap/航班={l['capacity_lbs_per_flight']:,}lbs dep/月={l['departures_per_month']} 距离={l['distance_miles']}mi")
    print(f"  SYNTHETIC 待定(占位,不臆造): {list(synthetic_todo.keys())[1:]}")
    print(f"\n写出 → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

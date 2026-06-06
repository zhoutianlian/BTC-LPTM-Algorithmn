from __future__ import annotations

from pathlib import Path
import html
import json
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from .constants import STATES, STATE_DEFINITIONS
from .segment_quality import compute_segment_diagnostics

STATE_COLORS = {
    "ST": "#00D084",
    "VT": "#FFB000",
    "RC": "#22A7F0",
    "RHA": "#B36BFF",
    "HPEM": "#FF3B5C",
    "AMB": "#9AA4B2",
}
DARK_BG = "#070B14"
PANEL_BG = "#0E1626"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
GRID = "rgba(148,163,184,0.18)"


def _safe_num(x: Any, digits: int = 4) -> Any:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    return str(x)


def _fmt_ts(x: Any) -> str:
    try:
        return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "NA"


def _fmt_input(x: Any) -> str:
    try:
        return pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M")
    except Exception:
        return ""


def _compact_text(value: object, max_chars: int = 300) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    text = str(value).replace("; ", "\n").replace(" | ", "\n")
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def _plie_direction_label(v: object) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if x > 0:
        return "+1 short-liq forced buying ↑"
    if x < 0:
        return "-1 long-liq forced selling ↓"
    return "0 neutral"


def _hmm_state_label(v: object) -> str:
    try:
        x = int(float(v))
    except Exception:
        return "NA"
    return {
        1: "1 strong short-liq / upward pressure",
        2: "2 mild short-liq / upward pressure",
        3: "3 balanced / neutral",
        4: "4 mild long-liq / downward pressure",
        5: "5 strong long-liq / downward pressure",
    }.get(x, str(x))


def _series(df: pd.DataFrame, col: str, default: Any = None) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _num_list(df: pd.DataFrame, col: str) -> list[Any]:
    return [None if pd.isna(v) or not np.isfinite(v) else round(float(v), 6) for v in pd.to_numeric(_series(df, col, np.nan), errors="coerce")]


def _str_list(df: pd.DataFrame, col: str, default: str = "NA") -> list[str]:
    return [_safe_str(v) for v in _series(df, col, default).fillna(default)]


def _state_segments(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    times = pd.to_datetime(df["time"])
    step = times.diff().dropna().median() if len(times) > 1 else pd.Timedelta(hours=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=1)
    gids = df["stable_state"].astype(str).ne(df["stable_state"].astype(str).shift()).cumsum()
    starts = []
    groups = []
    for _, g in df.assign(_gid=gids).groupby("_gid", sort=True):
        groups.append(g)
        starts.append(pd.Timestamp(g["time"].iloc[0]))
    segs = []
    for i, g in enumerate(groups):
        x0 = starts[i]
        x1 = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp(g["time"].iloc[-1]) + step
        segs.append({"x0": _fmt_ts(x0), "x1": _fmt_ts(x1), "state": str(g["stable_state"].iloc[0]), "n": int(len(g))})
    return segs


def _top_problem_segments(df: pd.DataFrame, limit: int = 80) -> list[dict[str, Any]]:
    """Fast dashboard segment table.

    The full segment audit is written by the pipeline to segment_diagnostics.csv.
    To keep the self-contained dashboard responsive on 40k+ rows, this panel only
    builds a lightweight table from row-level quality flags when available and does
    not recompute full segment diagnostics inside the browser artifact generation.
    """
    if df.empty or "segment_id" not in df.columns:
        return []
    issue_cols = [
        c for c in [
            "price_expectation_exit_flag", "price_expectation_raw_issue_flag",
            "directional_exit_trigger_flag", "weak_state_flag",
            "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag",
            "vt_low_move_flag", "vt_mixed_direction_flag", "vt_weak_trend_flag",
            "st_flat_flag", "st_weak_trend_flag", "st_too_extreme_flag",
            "hpem_price_expectation_fail_flag", "rha_price_expectation_fail_flag",
        ] if c in df.columns
    ]
    if not issue_cols:
        return []
    d = df.copy()
    d["_issue_count"] = pd.to_numeric(d[issue_cols].sum(axis=1), errors="coerce").fillna(0)
    g = d.groupby("segment_id", sort=False).agg(
        state=("stable_state", "first"),
        start_time=("time", "first"),
        end_time=("time", "last"),
        bars=("time", "size"),
        price0=("price", "first"),
        price1=("price", "last"),
        high=("price", "max"),
        low=("price", "min"),
        quality=("trading_trigger_quality", "mean"),
        issues=("_issue_count", "sum"),
    ).reset_index()
    g["abs_return_bps"] = (10000.0 * np.log(pd.to_numeric(g["price1"], errors="coerce") / pd.to_numeric(g["price0"], errors="coerce"))).abs()
    g["range_bps"] = 10000.0 * np.log(pd.to_numeric(g["high"], errors="coerce") / pd.to_numeric(g["low"], errors="coerce"))
    g = g[g["issues"] > 0].sort_values(["issues", "quality"], ascending=[False, True]).head(limit)
    rows = []
    for _, r in g.iterrows():
        rows.append({
            "segment_id": int(r.get("segment_id", 0)),
            "state": _safe_str(r.get("state")),
            "start_time": _fmt_ts(r.get("start_time")),
            "end_time": _fmt_ts(r.get("end_time")),
            "bars": _safe_num(r.get("bars"), 0),
            "abs_return_bps": _safe_num(r.get("abs_return_bps"), 2),
            "range_bps": _safe_num(r.get("range_bps"), 2),
            "quality": _safe_num(r.get("quality"), 3),
            "issues": f"row issue count={int(r.get('issues',0))}",
        })
    return rows


def _hover_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for i, r in df.iterrows():
        records.append({
            "idx": int(i),
            "date": _fmt_ts(r.get("time")),
            "price": _safe_num(r.get("price"), 2),
            "stable_state": _safe_str(r.get("stable_state")),
            "candidate_state": _safe_str(r.get("candidate_state")),
            "confirmation_status": _safe_str(r.get("state_confirmation_status", r.get("trading_trigger_quality_label", "NA"))),
            "trading_trigger_state": _safe_str(r.get("trading_trigger_state", r.get("confirmed_state", "NA"))),
            "trading_trigger_quality": _safe_num(r.get("trading_trigger_quality"), 3),
            "state_confirmation_score": _safe_num(r.get("state_confirmation_score"), 3),
            "weak_state_flag": _safe_num(r.get("weak_state_flag", 0), 0),
            "quality_issue": _safe_str(r.get("state_quality_issue", r.get("state_confirmation_failed_rules", ""))),
            "segment_id": _safe_num(r.get("segment_id", r.get("current_segment_id", 0)), 0),
            "bars_since_entry": _safe_num(r.get("bars_since_state_entry", r.get("bars_since_entry", None)), 0),
            "entry_return_bps": _safe_num(r.get("entry_to_current_return_bps", r.get("entry_return_bps", None)), 2),
            "entry_range_bps": _safe_num(r.get("entry_to_current_range_bps", r.get("entry_range_bps", None)), 2),
            "plie_direction": _plie_direction_label(r.get("plie_direction")),
            "plie_main_bps": _safe_num(r.get("plie_main_bps"), 2),
            "plie_intensity": _safe_num(r.get("plie_intensity"), 3),
            "plie_reliability": _safe_num(r.get("plie_reliability"), 3),
            "plie_phase": _safe_str(r.get("plie_phase")),
            "plie_strong_entry": _safe_str(r.get("plie_strong_entry")),
            "hmm_state": _hmm_state_label(r.get("hmm_state")),
            "hmm_conf": _safe_num(r.get("hmm_conf"), 3),
            "liq_entropy": _safe_num(r.get("liq_entropy"), 3),
            "path6": f"{_safe_str(r.get('path_context_6h'))} / {_safe_str(r.get('path_label_6h'))}",
            "path24": f"{_safe_str(r.get('path_context_24h'))} / {_safe_str(r.get('path_label_24h'))}",
            "rv1h": _safe_num(r.get("realized_vol_1h_bps"), 2),
            "rv6h": _safe_num(r.get("realized_vol_6h_bps"), 2),
            "trend6h": _safe_num(r.get("trend_strength_6h"), 3),
            "consistency6h": _safe_num(r.get("trend_consistency_6h"), 3),
            "range_comp6h": _safe_num(r.get("range_compression_6h"), 3),
            "jump1h": _safe_num(r.get("jump_proxy_1h"), 3),
            "shock1h": _safe_num(r.get("e_price_shock_1h"), 3),
            "impulse1h": _safe_num(r.get("e_short_impulse_1h"), 3),
            "exit_pressure": _safe_num(r.get("exit_pressure_current_state"), 3),
            "invalidity": _safe_num(r.get("state_invalidity_score"), 3),
            "beliefs": {s: _safe_num(r.get(f"b_{s}"), 3) for s in STATES},
            "stable_beliefs": {s: _safe_num(r.get(f"stable_b_{s}"), 3) for s in STATES},
            "top_evidence": _compact_text(r.get("top_evidence_features", ""), 360),
            "state_price_expectation_score": _safe_num(r.get("state_price_expectation_score"), 3),
            "state_price_expectation_issue": _safe_str(r.get("state_price_expectation_issue")),
            "price_expectation_exit_flag": _safe_num(r.get("price_expectation_exit_flag"), 0),
            "price_expectation_exit_target": _safe_str(r.get("price_expectation_exit_target")),
            "price_expectation_exit_reason": _safe_str(r.get("price_expectation_exit_reason")),
            "price_expectation_exit_pressure": _safe_num(r.get("price_expectation_exit_pressure"), 3),
            "execution_state": _safe_str(r.get("execution_state")),
            "execution_trade_direction": _safe_str(r.get("execution_trade_direction")),
            "execution_quality": _safe_num(r.get("execution_quality"), 3),
            "execution_no_trade_reason": _safe_str(r.get("execution_no_trade_reason")),
            "top_evidence": _compact_text(r.get("top_evidence_features", ""), 360),
            "transition_reason": _compact_text(r.get("transition_reason", ""), 420),
        })
    return records


def _summary_cards(df: pd.DataFrame) -> str:
    n = len(df)
    switches = int(pd.to_numeric(_series(df, "stable_switch_flag", 0), errors="coerce").fillna(0).sum())
    weak_ratio = float(pd.to_numeric(_series(df, "weak_state_flag", 0), errors="coerce").fillna(0).mean())
    mean_tq = float(pd.to_numeric(_series(df, "trading_trigger_quality", np.nan), errors="coerce").mean())
    cards = [
        ("Rows", f"{n:,}"),
        ("Start", _fmt_ts(df["time"].min())),
        ("End", _fmt_ts(df["time"].max())),
        ("Switches", f"{switches:,}"),
        ("Weak-state ratio", f"{weak_ratio:.2%}"),
        ("Mean trading quality", "NA" if not np.isfinite(mean_tq) else f"{mean_tq:.3f}"),
    ]
    return "<section class='cards'>" + "".join(f"<div class='card'><div class='card-label'>{html.escape(k)}</div><div class='card-value'>{html.escape(v)}</div></div>" for k, v in cards) + "</section>"


def _state_definitions() -> str:
    rows = []
    for s in STATES:
        d = STATE_DEFINITIONS[s]
        rows.append(f"<tr><td><span class='state-pill' style='background:{STATE_COLORS[s]}'></span><b>{s}</b></td><td>{html.escape(d['name'])}</td><td>{html.escape(d['cn'])}</td><td>{html.escape(d['meaning'])}</td></tr>")
    return "<section class='panel'><h2>State Definitions</h2><table class='defs-table'><thead><tr><th>State</th><th>Name</th><th>中文</th><th>Meaning</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></section>"


def _css() -> str:
    return f"""
<style>
:root {{ --bg:{DARK_BG}; --panel:{PANEL_BG}; --text:{TEXT}; --muted:{MUTED}; --border:#243044; --accent:#38BDF8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at top left,#0F1B33 0%,var(--bg) 34%,#020617 100%); color:var(--text); font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ padding:24px 30px 14px; border-bottom:1px solid var(--border); background:rgba(7,11,20,.9); position:sticky; top:0; z-index:30; backdrop-filter:blur(12px); }}
h1 {{ margin:0 0 8px; font-size:25px; }} h2 {{ margin:0 0 14px; font-size:18px; }} .subtitle {{ color:var(--muted); line-height:1.5; }}
.control-bar {{ margin-top:15px; display:flex; flex-wrap:wrap; gap:10px 12px; align-items:center; }} label {{ color:var(--muted); font-size:13px; }}
input[type=datetime-local] {{ margin-left:6px; background:#0B1220; color:#E2E8F0; border:1px solid #334155; border-radius:8px; padding:8px 10px; }}
button {{ background:linear-gradient(135deg,#0EA5E9,#2563EB); color:white; border:0; border-radius:999px; padding:8px 13px; cursor:pointer; font-weight:700; }} button:hover {{ filter:brightness(1.12); }} .hint {{ color:#94A3B8; font-size:12px; }}
main {{ padding:22px 28px 48px; }} .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:18px; }}
.card,.panel,.plot-wrap {{ background:rgba(14,22,38,.78); border:1px solid var(--border); border-radius:16px; box-shadow:0 12px 35px rgba(0,0,0,.18); }} .card {{ padding:13px 14px; }} .card-label {{ color:var(--muted); font-size:12px; margin-bottom:5px; }} .card-value {{ font-size:17px; font-weight:800; }} .panel {{ padding:16px; margin:16px 0; }} .plot-wrap {{ padding:8px; margin:16px 0; overflow:hidden; }}
.price-inspect-grid {{ display:grid; grid-template-columns:minmax(0,1fr) 430px; gap:16px; align-items:start; }} .grid-2 {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }} @media(max-width:1250px){{.price-inspect-grid,.grid-2{{grid-template-columns:1fr;}}}}
.inspect-panel {{ position:sticky; top:146px; max-height:calc(100vh - 170px); overflow:auto; background:rgba(11,18,32,.98); border:1px solid #334155; border-radius:16px; padding:14px; box-shadow:0 12px 35px rgba(0,0,0,.25); }} .inspect-title {{ font-size:17px; font-weight:900; }} .inspect-subtitle {{ color:#94A3B8; font-size:12px; margin-bottom:12px; line-height:1.45; }} .inspect-content.empty {{ color:#64748B; }} .inspect-section {{ border-top:1px solid #243044; padding-top:9px; margin-top:9px; }} .inspect-row {{ display:grid; grid-template-columns:145px minmax(0,1fr); gap:8px; margin:4px 0; font-size:12px; line-height:1.36; }} .inspect-key {{ color:#94A3B8; }} .inspect-value {{ color:#E5E7EB; overflow-wrap:anywhere; white-space:pre-wrap; }} .state-badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:800; color:#020617; }} .warn {{ color:#FCA5A5; }} .good {{ color:#86EFAC; }} .muted {{ color:#94A3B8; }}
.defs-table,.quality-table {{ width:100%; border-collapse:collapse; font-size:13px; }} .defs-table th,.defs-table td,.quality-table th,.quality-table td {{ border-bottom:1px solid #243044; padding:9px 10px; text-align:left; vertical-align:top; }} .defs-table th,.quality-table th {{ color:#CBD5E1; background:#0B1220; position:sticky; top:0; }} .quality-scroll {{ max-height:460px; overflow:auto; }} .state-pill {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; vertical-align:-1px; box-shadow:0 0 12px currentColor; }}
</style>
"""


def _js(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    colors_json = json.dumps(STATE_COLORS, separators=(",", ":"))
    states_json = json.dumps(STATES)
    js = r"""
<script>
const D = __DATA_JSON__;
const COLORS = __COLORS_JSON__;
const STATES = __STATES_JSON__;
let PINNED = false, SYNCING = false;
function val(x,d=3){ if(x===null||x===undefined||Number.isNaN(x)) return 'NA'; if(typeof x==='number') return x.toFixed(d); return String(x); }
function parseT(x){ return Date.parse(x); } function pad(n){ return String(n).padStart(2,'0'); } function inp(d){ return d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+'T'+pad(d.getHours())+':'+pad(d.getMinutes()); }
function yRange(start,end){ const s=parseT(start), e=parseT(end); let mn=Infinity,mx=-Infinity,c=0; for(let i=0;i<D.time.length;i++){ const t=parseT(D.time[i]), p=D.price[i]; if(p===null) continue; if(t>=s&&t<=e){mn=Math.min(mn,p); mx=Math.max(mx,p); c++;} } if(!c) return null; const padv=Math.max((mx-mn)*.06, Math.abs(mx)*.001, 10); return [mn-padv,mx+padv]; }
function relayout(id,upd){ const el=document.getElementById(id); if(!el) return Promise.resolve(); return Plotly.relayout(id,upd).catch(()=>{}); }
const TIME_CHARTS=['price-chart','state-band-chart','belief-chart','stable-belief-chart','evidence-chart','responsiveness-chart','confirmation-chart','directional-chart'];
function applyRange(s,e,upd=true){ if(!s||!e||parseT(s)>=parseT(e)) return; if(upd){startTime.value=s; endTime.value=e;} SYNCING=true; const common={'xaxis.range':[s,e]}; const yr=yRange(s,e); const ops=[relayout('price-chart', yr?{'xaxis.range':[s,e],'yaxis.range':yr}:common)]; for(const id of TIME_CHARTS) if(id!=='price-chart') ops.push(relayout(id,common)); Promise.all(ops).finally(()=>setTimeout(()=>{SYNCING=false;},80)); }
function applySelectedRange(){ applyRange(startTime.value,endTime.value,false); } function resetFullRange(){ applyRange(D.start,D.end,true); } function setLastDays(days){ const end=parseT(endTime.value||D.end); applyRange(inp(new Date(end-days*86400000)), inp(new Date(end)), true); }
function bindSync(id){ const el=document.getElementById(id); if(!el) return; el.on('plotly_relayout', ev=>{ if(SYNCING) return; let s=null,e=null; if(ev['xaxis.range[0]']&&ev['xaxis.range[1]']){s=String(ev['xaxis.range[0]']).slice(0,16); e=String(ev['xaxis.range[1]']).slice(0,16);} else if(Array.isArray(ev['xaxis.range'])){s=String(ev['xaxis.range'][0]).slice(0,16); e=String(ev['xaxis.range'][1]).slice(0,16);} else if(ev['xaxis.autorange']){resetFullRange(); return;} if(s&&e) applyRange(s,e,true); }); }
function section(t,rows){ let h=`<div class="inspect-section"><b>${t}</b>`; for(const r of rows) h += `<div class="inspect-row"><div class="inspect-key">${r[0]}</div><div class="inspect-value ${r[2]||''}">${r[1]}</div></div>`; return h+'</div>'; }
function H(name,i){ return D.h && D.h[name] ? D.h[name][i] : null; }
function B(prefix,s,i){ const k=prefix+s; return D[k] ? D[k][i] : null; }
function updateInspect(i,pin=false){
 const r={
  date:D.time[i], price:D.price[i], stable_state:D.state[i], candidate_state:H('candidate_state',i),
  confirmation_status:H('confirmation_status',i), trading_trigger_state:H('trading_trigger_state',i),
  trading_trigger_quality:H('trading_trigger_quality',i), directional_state:H('directional_state',i),
  trade_direction:H('trade_direction',i), direction_confidence:H('direction_confidence',i),
  direction_source:H('direction_source',i), directional_trading_quality:H('directional_trading_quality',i),
  directional_no_trade_reason:H('directional_no_trade_reason',i), directional_exit_trigger_flag:H('directional_exit_trigger_flag',i),
  directional_exit_reason:H('directional_exit_reason',i), directional_exit_target:H('directional_exit_target',i),
  price_expectation_score:H('price_expectation_score',i), price_expectation_issue:H('price_expectation_issue',i),
  price_expectation_exit_flag:H('price_expectation_exit_flag',i), price_expectation_exit_target:H('price_expectation_exit_target',i),
  price_expectation_exit_reason:H('price_expectation_exit_reason',i), price_expectation_exit_pressure:H('price_expectation_exit_pressure',i),
  execution_state:H('execution_state',i), execution_trade_direction:H('execution_trade_direction',i),
  execution_quality:H('execution_quality',i), execution_no_trade_reason:H('execution_no_trade_reason',i),
  rc_price_drift_flag:H('rc_price_drift_flag',i), rc_range_break_flag:H('rc_range_break_flag',i), rc_trend_exclusion_flag:H('rc_trend_exclusion_flag',i),
  vt_low_move_flag:H('vt_low_move_flag',i), vt_mixed_direction_flag:H('vt_mixed_direction_flag',i), vt_weak_trend_flag:H('vt_weak_trend_flag',i),
  pe_live_score:H('price_expectation_live_score',i), pe_live_issue:H('price_expectation_live_issue',i), pe_live_invalidity:H('price_expectation_invalidity_score',i), pe_live_reason:H('price_expectation_live_reason',i), pe_live_exit_flag:H('price_expectation_live_exit_flag',i), pe_live_exit_target:H('price_expectation_live_exit_target',i), pe_live_override:H('price_expectation_override_flag',i),
  st_flat_flag:H('st_flat_flag',i), st_weak_trend_flag:H('st_weak_trend_flag',i), st_too_extreme_flag:H('st_too_extreme_flag',i),
  state_confirmation_score:H('state_confirmation_score',i),
  weak_state_flag:H('weak_state_flag',i), quality_issue:H('quality_issue',i), segment_id:H('segment_id',i), bars_since_entry:H('bars_since_entry',i),
  entry_return_bps:H('entry_return_bps',i), entry_range_bps:H('entry_range_bps',i), plie_direction:H('plie_direction',i),
  plie_main_bps:H('plie_main_bps',i), plie_intensity:H('plie_intensity',i), plie_reliability:H('plie_reliability',i), plie_phase:H('plie_phase',i), plie_strong_entry:H('plie_strong_entry',i),
  hmm_state:H('hmm_state',i), hmm_conf:H('hmm_conf',i), liq_entropy:H('liq_entropy',i), path6:H('path6',i), path24:H('path24',i), rv1h:H('rv1h',i), rv6h:H('rv6h',i), trend6h:H('trend6h',i), consistency6h:H('consistency6h',i), range_comp6h:H('range_comp6h',i), jump1h:H('jump1h',i),
  shock1h:H('shock1h',i), impulse1h:H('impulse1h',i), exit_pressure:H('exit_pressure',i), invalidity:H('invalidity',i),
  hpem_aligned_move:H('hpem_aligned_move',i), hpem_aligned_mfe:H('hpem_aligned_mfe',i),
  hpem_giveback:H('hpem_giveback',i), hpem_giveback_ratio:H('hpem_giveback_ratio',i), hpem_decay_flag:H('hpem_decay_flag',i), hpem_no_response_flag:H('hpem_no_response_flag',i),
  rha_subtype:H('rha_subtype',i), rha_reversal_move:H('rha_reversal_move',i), rha_reversal_mfe:H('rha_reversal_mfe',i), rha_stall_flag:H('rha_stall_flag',i), rha_invalid_follow_plie_flag:H('rha_invalid_follow_plie_flag',i),
  vt_direction_score:H('vt_direction_score',i), vt_direction_state:H('vt_direction_state',i), vt_direction_confidence:H('vt_direction_confidence',i), vt_direction_flip_flag:H('vt_direction_flip_flag',i),
  top_evidence:H('top_evidence',i), transition_reason:H('transition_reason',i)
 };
 const color=COLORS[r.stable_state]||'#64748B'; const qcls=(r.trading_trigger_quality!==null&&r.trading_trigger_quality<.45)?'warn':'good'; let h='';
 h += `<div class="inspect-row"><div class="inspect-key">Date</div><div class="inspect-value"><b>${r.date}</b> ${pin?'<span class="muted">(pinned)</span>':''}</div></div>`;
 h += `<div class="inspect-row"><div class="inspect-key">Price</div><div class="inspect-value"><b>${val(r.price,2)}</b></div></div>`;
 h += `<div class="inspect-row"><div class="inspect-key">State</div><div class="inspect-value"><span class="state-badge" style="background:${color}">${r.stable_state}</span> / ${r.confirmation_status}</div></div>`;
 h += section('Confirmation / trading quality', [['Trading state',r.trading_trigger_state],['Trading quality',val(r.trading_trigger_quality,3),qcls],['Confirmation score',val(r.state_confirmation_score,3)],['Weak / issue',`${r.weak_state_flag} / ${r.quality_issue||'none'}`,r.weak_state_flag?'warn':'good'],['Segment / bars',`${r.segment_id} / ${r.bars_since_entry}`],['Entry return / range',`${val(r.entry_return_bps,2)} bps / ${val(r.entry_range_bps,2)} bps`]]);
 h += section('Directional lifecycle', [['Directional state',r.directional_state],['Trade direction',r.trade_direction],['Direction confidence',val(r.direction_confidence,3)],['Directional quality',val(r.directional_trading_quality,3),(r.directional_trading_quality!==null&&r.directional_trading_quality<.45)?'warn':'good'],['Source',r.direction_source],['No-trade reason',r.directional_no_trade_reason||'none'],['Exit trigger / target',`${r.directional_exit_trigger_flag} / ${r.directional_exit_target||''}`],['Exit reason',r.directional_exit_reason||'none']]);
 h += section('Price expectation compliance', [['Expectation score',val(r.price_expectation_score,3),(r.price_expectation_score!==null&&r.price_expectation_score<.45)?'warn':'good'],['Issue',r.price_expectation_issue||'OK'],['Exit flag / target',`${r.price_expectation_exit_flag} / ${r.price_expectation_exit_target||''}`],['Exit reason',r.price_expectation_exit_reason||'none'],['Exit pressure',val(r.price_expectation_exit_pressure,3)],['Execution state',r.execution_state||'NA'],['Execution quality',val(r.execution_quality,3),(r.execution_quality!==null&&r.execution_quality<.45)?'warn':'good'],['Execution reason',r.execution_no_trade_reason||'none'],['RC drift/range/trend',`${r.rc_price_drift_flag} / ${r.rc_range_break_flag} / ${r.rc_trend_exclusion_flag}`],['VT low/mixed/weak-vol',`${r.vt_low_move_flag} / ${r.vt_mixed_direction_flag} / ${r.vt_weak_trend_flag}`],['Live PE score/invalidity',`${val(r.pe_live_score,3)} / ${val(r.pe_live_invalidity,3)}`],['Live PE exit/target',`${r.pe_live_exit_flag} / ${r.pe_live_exit_target||''}`],['Live PE issue/reason',`${r.pe_live_issue||''} / ${r.pe_live_reason||'none'}`],['ST flat/weak/extreme',`${r.st_flat_flag} / ${r.st_weak_trend_flag} / ${r.st_too_extreme_flag}`]]);
 h += section('HPEM / RHA / VT direction metrics', [['HPEM aligned / MFE',`${val(r.hpem_aligned_move,2)} / ${val(r.hpem_aligned_mfe,2)} bps`],['HPEM giveback / ratio',`${val(r.hpem_giveback,2)} bps / ${val(r.hpem_giveback_ratio,3)}`],['HPEM decay / no-response',`${r.hpem_decay_flag} / ${r.hpem_no_response_flag}`],['RHA subtype',r.rha_subtype],['RHA reversal / MFE',`${val(r.rha_reversal_move,2)} / ${val(r.rha_reversal_mfe,2)} bps`],['RHA stall / invalid',`${r.rha_stall_flag} / ${r.rha_invalid_follow_plie_flag}`],['VT score / state',`${val(r.vt_direction_score,3)} / ${r.vt_direction_state}`],['VT confidence / flip',`${val(r.vt_direction_confidence,3)} / ${r.vt_direction_flip_flag}`]]);
 h += section('PLIE / HMM', [['PLIE direction',r.plie_direction],['PLIE main bps',val(r.plie_main_bps,2)],['Intensity / reliability',`${val(r.plie_intensity,3)} / ${val(r.plie_reliability,3)}`],['Phase / strong entry',`${r.plie_phase} / ${r.plie_strong_entry}`],['HMM state',r.hmm_state],['HMM conf / entropy',`${val(r.hmm_conf,3)} / ${val(r.liq_entropy,3)}`]]);
 h += section('Path / price context', [['6h path',r.path6],['24h path',r.path24],['RV 1h / 6h',`${val(r.rv1h,2)} / ${val(r.rv6h,2)} bps`],['Trend strength / consistency 6h',`${val(r.trend6h,3)} / ${val(r.consistency6h,3)}`],['Range comp 6h / Jump 1h',`${val(r.range_comp6h,3)} / ${val(r.jump1h,3)}`]]);
 h += section('Responsiveness', [['Shock / impulse 1h',`${val(r.shock1h,3)} / ${val(r.impulse1h,3)}`],['Exit pressure / invalidity',`${val(r.exit_pressure,3)} / ${val(r.invalidity,3)}`]]);
 h += section('Belief vector', [['raw',`ST ${val(B('b_','ST',i))} | VT ${val(B('b_','VT',i))} | RC ${val(B('b_','RC',i))}<br>RHA ${val(B('b_','RHA',i))} | HPEM ${val(B('b_','HPEM',i))} | AMB ${val(B('b_','AMB',i))}`],['stable',`ST ${val(B('stable_b_','ST',i))} | VT ${val(B('stable_b_','VT',i))} | RC ${val(B('stable_b_','RC',i))}<br>RHA ${val(B('stable_b_','RHA',i))} | HPEM ${val(B('stable_b_','HPEM',i))} | AMB ${val(B('stable_b_','AMB',i))}`]]);
 h += section('Evidence / reason', [['Top evidence',r.top_evidence],['Transition reason',r.transition_reason]]);
 inspectContent.innerHTML=h;
}
function idx(ev){ if(!ev||!ev.points||!ev.points.length) return null; const cd=ev.points[0].customdata; if(cd===undefined||cd===null) return null; return Array.isArray(cd)?Number(cd[0]):Number(cd); }
function bindInspect(id){ const el=document.getElementById(id); if(!el) return; el.on('plotly_hover', ev=>{ if(PINNED) return; const i=idx(ev); if(Number.isFinite(i)) updateInspect(i,false); }); el.on('plotly_click', ev=>{ const i=idx(ev); if(Number.isFinite(i)){PINNED=true; updateInspect(i,true);} }); el.on('plotly_doubleclick',()=>{PINNED=false;}); }
function baseLayout(title,h){ return {title:{text:title,x:.01,font:{color:'#E5E7EB',size:17}},height:h,paper_bgcolor:'__DARK_BG__',plot_bgcolor:'__PANEL_BG__',font:{color:'#E5E7EB'},margin:{l:58,r:28,t:58,b:52},legend:{orientation:'h',y:1.02,x:1,xanchor:'right'},hovermode:'closest',hoverdistance:90,spikedistance:-1,xaxis:{type:'date',gridcolor:'__GRID__',showspikes:true},yaxis:{gridcolor:'__GRID__'}}; }
function stateShapes(){ return D.segments.map(s=>{ const c=COLORS[s.state]||'#64748B'; const r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16); return {type:'rect',xref:'x',yref:'paper',x0:s.x0,x1:s.x1,y0:0,y1:1,fillcolor:`rgba(${r},${g},${b},.13)`,line:{width:0},layer:'below'}; }); }
function drawCharts(){
 const ids=D.idx; const stateText=D.state.map((s,i)=>s+' / '+(D.confirmation_status[i]||'')); const colors=D.state.map(s=>COLORS[s]||'#64748B');
 let priceData=[{x:D.time,y:D.price,type:'scattergl',mode:'lines',name:'BTC price',line:{color:'#C7D2FE',width:1.35},hoverinfo:'skip'},{x:D.time,y:D.price,type:'scattergl',mode:'markers',name:'state dots',marker:{size:3,color:colors,opacity:.82},hoverinfo:'skip'},{x:D.time,y:D.price,type:'scattergl',mode:'markers',name:'inspect anchor',customdata:ids,text:stateText,marker:{size:8,color:'rgba(255,255,255,.01)'},hovertemplate:'<b>%{x|%Y-%m-%d %H:%M}</b><br>Price: %{y:,.2f}<br>State: %{text}<br><span style="color:#94A3B8">Details update in the inspection panel.</span><extra></extra>'}];
 let pl=baseLayout('Price + stable-state background + fast inspection anchor',670); pl.shapes=stateShapes(); pl.xaxis.rangeslider={visible:true,bgcolor:'#0B1220',bordercolor:'#334155',borderwidth:1}; Plotly.newPlot('price-chart',priceData,pl,{responsive:true});
 const stateId=Object.fromEntries(STATES.map((s,i)=>[s,i])); Plotly.newPlot('state-band-chart',[{x:D.time,y:D.state.map(s=>stateId[s]),type:'scattergl',mode:'markers',customdata:ids,text:D.state,marker:{size:4,color:colors,symbol:'square'},hovertemplate:'<b>%{x|%Y-%m-%d %H:%M}</b><br>State: %{text}<extra></extra>'}],Object.assign(baseLayout('Stable-state timeline band',310),{yaxis:{tickmode:'array',tickvals:STATES.map((s,i)=>i),ticktext:STATES,range:[-.7,STATES.length-.3],gridcolor:'__GRID__'}}),{responsive:true});
 const beliefData=STATES.map(s=>({x:D.time,y:D['b_'+s],type:'scattergl',mode:'lines',name:s,line:{color:COLORS[s],width:1.2},hovertemplate:`<b>%{x|%Y-%m-%d %H:%M}</b><br>b_${s}: %{y:.4f}<extra></extra>`})); Plotly.newPlot('belief-chart',beliefData,Object.assign(baseLayout('Filtered raw belief vector',430),{yaxis:{range:[0,1],gridcolor:'__GRID__'}}),{responsive:true});
 const sbData=STATES.map(s=>({x:D.time,y:D['stable_b_'+s],type:'scattergl',mode:'lines',name:s,line:{color:COLORS[s],width:1.2},hovertemplate:`<b>%{x|%Y-%m-%d %H:%M}</b><br>stable_b_${s}: %{y:.4f}<extra></extra>`})); Plotly.newPlot('stable-belief-chart',sbData,Object.assign(baseLayout('EWMA stable belief vector',430),{yaxis:{range:[0,1],gridcolor:'__GRID__'}}),{responsive:true});
 function lineChart(id,title,cols,h){ const pal=['#38BDF8','#F59E0B','#EF4444','#10B981','#A78BFA','#F97316','#22C55E','#E879F9','#60A5FA','#FACC15','#FB7185','#22D3EE']; const data=cols.filter(c=>D[c]).map((c,i)=>({x:D.time,y:D[c],type:'scattergl',mode:(c.endsWith('flag')?'markers':'lines'),name:c,marker:{size:4,color:pal[i%pal.length]},line:{color:pal[i%pal.length],width:1.15},hovertemplate:`<b>%{x|%Y-%m-%d %H:%M}</b><br>${c}: %{y:.4f}<extra></extra>`})); Plotly.newPlot(id,data,Object.assign(baseLayout(title,h),{yaxis:{range:[0,1],gridcolor:'__GRID__'}}),{responsive:true}); }
 lineChart('evidence-chart','Semantic evidence primitives',['e_directional_pressure','e_pressure_strength','e_cascade','e_absorption','e_rejection','e_active','e_quiet','e_realized_vol','e_range_compression','e_trend_strength','e_jump','e_conflict'],480);
 lineChart('responsiveness-chart','Responsiveness: shock, invalidity, exit pressure',['e_price_shock_1h','e_short_impulse_1h','e_shock_aligned_plie','e_shock_against_plie','exit_pressure_current_state','state_invalidity_score','delayed_hold_flag','fast_transition_flag','exit_pressure_trigger_flag'],440);
 lineChart('confirmation-chart','Causal state confirmation + trading-trigger quality',['state_confirmation_score','trading_trigger_quality','weak_state_flag','state_pending_confirmation_flag','state_confirmed_flag'],440);
 lineChart('directional-chart','Directional & price-expectation lifecycle',['directional_trading_quality','direction_confidence','directional_exit_trigger_flag','state_price_expectation_score','price_expectation_exit_pressure','price_expectation_exit_flag','price_expectation_exit_pressure','price_expectation_exit_flag','hpem_directional_compliance_score','hpem_giveback_ratio','hpem_decay_flag','rha_directional_compliance_score','rha_invalid_follow_plie_flag','vt_direction_score','vt_direction_confidence','vt_direction_flip_flag','rc_price_drift_flag','vt_low_move_flag','vt_weak_trend_flag','st_flat_flag'],500);
 for(const id of TIME_CHARTS) bindSync(id); bindInspect('price-chart'); bindInspect('state-band-chart'); resetFullRange(); if(D.time.length) updateInspect(D.time.length-1,false);
}
window.addEventListener('load',drawCharts);
</script>
"""
    return js.replace("__DATA_JSON__", data_json).replace("__COLORS_JSON__", colors_json).replace("__STATES_JSON__", states_json).replace("__DARK_BG__", DARK_BG).replace("__PANEL_BG__", PANEL_BG).replace("__GRID__", GRID)

def _quality_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<section class='panel'><h2>Segment Quality Audit</h2><p class='muted'>No segment quality issues found.</p></section>"
    keys = ["segment_id", "state", "start_time", "bars", "abs_return_bps", "range_bps", "quality", "issues"]
    thead = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    trs = []
    for r in rows:
        trs.append("<tr>" + "".join(f"<td>{html.escape(str(r.get(k,'')))}</td>" for k in keys) + "</tr>")
    return "<section class='panel'><h2>Segment Quality Audit: weakest/problem segments</h2><div class='quality-scroll'><table class='quality-table'><thead><tr>" + thead + "</tr></thead><tbody>" + "\n".join(trs) + "</tbody></table></div></section>"


def build_dashboard(df: pd.DataFrame, output_html: str | Path, title: str = "Six-State CD-IOHSMM Dashboard") -> None:
    if df.empty:
        raise ValueError("Cannot build dashboard from an empty dataframe")
    # Keep all rows, but copy only columns used by the dashboard.
    # This is critical for hover responsiveness and memory use when the caller passes
    # state_timeseries_full_debug.csv with hundreds of columns.
    required = {
        "time", "price", "stable_state", "candidate_state", "state_confirmation_status",
        "trading_trigger_state", "trading_trigger_quality", "state_confirmation_score",
        "weak_state_flag", "state_quality_issue", "segment_id", "bars_since_state_entry",
        "entry_to_current_return_bps", "entry_to_current_range_bps", "plie_direction",
        "plie_main_bps", "plie_intensity", "plie_reliability", "plie_phase",
        "plie_strong_entry", "hmm_state", "hmm_conf", "liq_entropy",
        "path_context_6h", "path_label_6h", "path_context_24h", "path_label_24h",
        "realized_vol_1h_bps", "realized_vol_6h_bps", "trend_strength_6h",
        "trend_consistency_6h", "range_compression_6h", "jump_proxy_1h",
        "e_price_shock_1h", "e_short_impulse_1h", "exit_pressure_current_state",
        "state_invalidity_score", "top_evidence_features", "transition_reason",
        "e_directional_pressure", "e_pressure_strength", "e_cascade", "e_absorption",
        "e_rejection", "e_active", "e_quiet", "e_realized_vol", "e_range_compression",
        "e_trend_strength", "e_jump", "e_conflict", "e_shock_aligned_plie",
        "e_shock_against_plie", "delayed_hold_flag", "fast_transition_flag",
        "exit_pressure_trigger_flag", "state_pending_confirmation_flag", "state_confirmed_flag",
        "directional_state", "trade_direction", "direction_confidence", "direction_source",
        "directional_trading_quality", "directional_no_trade_reason",
        "directional_exit_trigger_flag", "directional_exit_reason", "directional_exit_target",
        "hpem_aligned_move_bps", "hpem_aligned_mfe_bps", "hpem_giveback_bps",
        "hpem_giveback_ratio", "hpem_no_response_flag", "hpem_decay_flag",
        "hpem_directional_compliance_score", "rha_subtype", "rha_reversal_move_bps",
        "rha_reversal_mfe_bps", "rha_stall_flag", "rha_invalid_follow_plie_flag",
        "rha_directional_compliance_score", "vt_direction_score", "vt_direction_state",
        "vt_direction_confidence", "vt_direction_flip_flag",
        "stability_price_expectation_score", "stability_price_expectation_issue",
        "stability_price_expectation_exit_flag", "stability_price_expectation_exit_target",
        "stability_price_expectation_exit_reason", "price_expectation_invalidity_score",
        "price_expectation_override_flag",
        "state_price_expectation_score", "state_price_expectation_issue",
        "price_expectation_exit_pressure", "price_expectation_live_reason", "price_expectation_exit_flag", "price_expectation_live_exit_target", "price_expectation_live_issue", "price_expectation_live_score", "price_expectation_exit_flag",
        "price_expectation_exit_flag", "price_expectation_exit_target", "price_expectation_exit_reason",
        "price_expectation_exit_pressure", "price_expectation_trading_quality", "price_expectation_no_trade_reason",
        "execution_state", "execution_trade_direction", "execution_quality", "execution_no_trade_reason",
        "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag",
        "vt_low_move_flag", "vt_mixed_direction_flag", "vt_weak_trend_flag",
        "st_flat_flag", "st_weak_trend_flag", "st_too_extreme_flag",
        "rha_weak_no_reversal_flag", "rha_stall_not_trend_flag", "rha_invalid_follow_plie_exit_flag",
        "hpem_price_no_response_exit_flag", "hpem_price_decay_exit_flag", "hpem_price_adverse_exit_flag",
    }
    for st in STATES:
        required.add(f"b_{st}")
        required.add(f"stable_b_{st}")
    keep = [c for c in df.columns if c in required]
    d = df[keep].copy()
    if "time" not in d.columns or "price" not in d.columns or "stable_state" not in d.columns:
        raise ValueError("Dashboard requires at least: time, price, stable_state")
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    # Backward-compatible aliases for V2.5 price-expectation fields.
    alias_map = {
        "price_expectation_exit_pressure": ["price_expectation_invalidity_score", "stability_price_expectation_score"],
        "price_expectation_live_reason": ["stability_price_expectation_issue"],
        "price_expectation_live_exit_target": ["stability_price_expectation_exit_target"],
        "price_expectation_live_score": ["stability_price_expectation_score"],
        "price_expectation_live_issue": ["stability_price_expectation_issue"],
    }
    for dst, srcs in alias_map.items():
        if dst not in d.columns:
            for src in srcs:
                if src in d.columns:
                    d[dst] = d[src]
                    break
        if dst not in d.columns:
            d[dst] = np.nan
    start, end = _fmt_input(d["time"].min()), _fmt_input(d["time"].max())
    state = _str_list(d, "stable_state")
    payload: dict[str, Any] = {
        "start": start, "end": end, "idx": list(range(len(d))),
        "time": [_fmt_input(t) for t in d["time"]],
        "price": _num_list(d, "price"),
        "state": state,
        "confirmation_status": _str_list(d, "state_confirmation_status", "NA"),
        "segments": _state_segments(d),
        "h": {
            "candidate_state": _str_list(d, "candidate_state"),
            "confirmation_status": _str_list(d, "state_confirmation_status", "NA"),
            "trading_trigger_state": _str_list(d, "trading_trigger_state", "NA"),
            "trading_trigger_quality": _num_list(d, "trading_trigger_quality"),
            "directional_state": _str_list(d, "directional_state", "NA"),
            "trade_direction": _str_list(d, "trade_direction", "NA"),
            "direction_confidence": _num_list(d, "direction_confidence"),
            "direction_source": _str_list(d, "direction_source", ""),
            "directional_trading_quality": _num_list(d, "directional_trading_quality"),
            "directional_no_trade_reason": _str_list(d, "directional_no_trade_reason", ""),
            "directional_exit_trigger_flag": _num_list(d, "directional_exit_trigger_flag"),
            "directional_exit_reason": _str_list(d, "directional_exit_reason", ""),
            "directional_exit_target": _str_list(d, "directional_exit_target", ""),
            "price_expectation_score": _num_list(d, "state_price_expectation_score"),
            "price_expectation_issue": _str_list(d, "state_price_expectation_issue", ""),
            "price_expectation_exit_flag": _num_list(d, "price_expectation_exit_flag"),
            "price_expectation_exit_target": _str_list(d, "price_expectation_exit_target", ""),
            "price_expectation_exit_reason": _str_list(d, "price_expectation_exit_reason", ""),
            "price_expectation_exit_pressure": _num_list(d, "price_expectation_exit_pressure"),
            "price_expectation_exit_pressure": _num_list(d, "price_expectation_exit_pressure"),
            "price_expectation_live_reason": _str_list(d, "price_expectation_live_reason", ""),
            "price_expectation_exit_flag": _num_list(d, "price_expectation_exit_flag"),
            "price_expectation_live_exit_target": _str_list(d, "price_expectation_live_exit_target", ""),
            "price_expectation_live_issue": _str_list(d, "price_expectation_live_issue", ""),
            "price_expectation_exit_flag": _num_list(d, "price_expectation_exit_flag"),
            "execution_state": _str_list(d, "execution_state", "NA"),
            "execution_trade_direction": _str_list(d, "execution_trade_direction", "NA"),
            "execution_quality": _num_list(d, "execution_quality"),
            "execution_no_trade_reason": _str_list(d, "execution_no_trade_reason", ""),
            "rc_price_drift_flag": _num_list(d, "rc_price_drift_flag"),
            "rc_range_break_flag": _num_list(d, "rc_range_break_flag"),
            "rc_trend_exclusion_flag": _num_list(d, "rc_trend_exclusion_flag"),
            "vt_low_move_flag": _num_list(d, "vt_low_move_flag"),
            "vt_mixed_direction_flag": _num_list(d, "vt_mixed_direction_flag"),
            "st_flat_flag": _num_list(d, "st_flat_flag"),
            "st_weak_trend_flag": _num_list(d, "st_weak_trend_flag"),
            "st_too_extreme_flag": _num_list(d, "st_too_extreme_flag"),
            "state_confirmation_score": _num_list(d, "state_confirmation_score"),
            "weak_state_flag": _num_list(d, "weak_state_flag"),
            "quality_issue": _str_list(d, "state_quality_issue", ""),
            "segment_id": _num_list(d, "segment_id"),
            "bars_since_entry": _num_list(d, "bars_since_state_entry"),
            "entry_return_bps": _num_list(d, "entry_to_current_return_bps"),
            "entry_range_bps": _num_list(d, "entry_to_current_range_bps"),
            "plie_direction": [_plie_direction_label(v) for v in _series(d, "plie_direction", np.nan)],
            "plie_main_bps": _num_list(d, "plie_main_bps"),
            "plie_intensity": _num_list(d, "plie_intensity"),
            "plie_reliability": _num_list(d, "plie_reliability"),
            "plie_phase": _str_list(d, "plie_phase", "NA"),
            "plie_strong_entry": _str_list(d, "plie_strong_entry", "NA"),
            "hmm_state": [_hmm_state_label(v) for v in _series(d, "hmm_state", np.nan)],
            "hmm_conf": _num_list(d, "hmm_conf"),
            "liq_entropy": _num_list(d, "liq_entropy"),
            "path6": [f"{a} / {b}" for a, b in zip(_str_list(d, "path_context_6h", "NA"), _str_list(d, "path_label_6h", "NA"))],
            "path24": [f"{a} / {b}" for a, b in zip(_str_list(d, "path_context_24h", "NA"), _str_list(d, "path_label_24h", "NA"))],
            "rv1h": _num_list(d, "realized_vol_1h_bps"),
            "rv6h": _num_list(d, "realized_vol_6h_bps"),
            "trend6h": _num_list(d, "trend_strength_6h"),
            "consistency6h": _num_list(d, "trend_consistency_6h"),
            "range_comp6h": _num_list(d, "range_compression_6h"),
            "jump1h": _num_list(d, "jump_proxy_1h"),
            "shock1h": _num_list(d, "e_price_shock_1h"),
            "impulse1h": _num_list(d, "e_short_impulse_1h"),
            "exit_pressure": _num_list(d, "exit_pressure_current_state"),
            "invalidity": _num_list(d, "state_invalidity_score"),
            "hpem_aligned_move": _num_list(d, "hpem_aligned_move_bps"),
            "hpem_aligned_mfe": _num_list(d, "hpem_aligned_mfe_bps"),
            "hpem_giveback": _num_list(d, "hpem_giveback_bps"),
            "hpem_giveback_ratio": _num_list(d, "hpem_giveback_ratio"),
            "hpem_decay_flag": _num_list(d, "hpem_decay_flag"),
            "hpem_no_response_flag": _num_list(d, "hpem_no_response_flag"),
            "rha_subtype": _str_list(d, "rha_subtype", "NA"),
            "rha_reversal_move": _num_list(d, "rha_reversal_move_bps"),
            "rha_reversal_mfe": _num_list(d, "rha_reversal_mfe_bps"),
            "rha_stall_flag": _num_list(d, "rha_stall_flag"),
            "rha_invalid_follow_plie_flag": _num_list(d, "rha_invalid_follow_plie_flag"),
            "vt_direction_score": _num_list(d, "vt_direction_score"),
            "vt_direction_state": _str_list(d, "vt_direction_state", "NA"),
            "vt_direction_confidence": _num_list(d, "vt_direction_confidence"),
            "vt_direction_flip_flag": _num_list(d, "vt_direction_flip_flag"),
            "price_expectation_score": _num_list(d, "state_price_expectation_score"),
            "price_expectation_issue": _str_list(d, "state_price_expectation_issue", "NA"),
            "price_expectation_exit_flag": _num_list(d, "price_expectation_exit_flag"),
            "price_expectation_exit_target": _str_list(d, "price_expectation_exit_target", ""),
            "price_expectation_exit_reason": _str_list(d, "price_expectation_exit_reason", ""),
            "price_expectation_exit_pressure": _num_list(d, "price_expectation_exit_pressure"),
            "price_expectation_exit_pressure": _num_list(d, "price_expectation_exit_pressure"),
            "price_expectation_live_reason": _str_list(d, "price_expectation_live_reason", ""),
            "price_expectation_exit_flag": _num_list(d, "price_expectation_exit_flag"),
            "price_expectation_live_exit_target": _str_list(d, "price_expectation_live_exit_target", ""),
            "price_expectation_live_issue": _str_list(d, "price_expectation_live_issue", ""),
            "price_expectation_exit_flag": _num_list(d, "price_expectation_exit_flag"),
            "execution_state": _str_list(d, "execution_state", "NA"),
            "execution_trade_direction": _str_list(d, "execution_trade_direction", "NA"),
            "execution_quality": _num_list(d, "execution_quality"),
            "execution_no_trade_reason": _str_list(d, "execution_no_trade_reason", ""),
            "top_evidence": [_compact_text(v, 360) for v in _series(d, "top_evidence_features", "")],
            "transition_reason": [_compact_text(v, 420) for v in _series(d, "transition_reason", "")],
        },
    }
    for s in STATES:
        payload[f"b_{s}"] = _num_list(d, f"b_{s}")
        payload[f"stable_b_{s}"] = _num_list(d, f"stable_b_{s}")
    extra_cols = [
        "e_directional_pressure","e_pressure_strength","e_cascade","e_absorption","e_rejection","e_active","e_quiet","e_realized_vol","e_range_compression","e_trend_strength","e_jump","e_conflict",
        "e_price_shock_1h","e_short_impulse_1h","e_shock_aligned_plie","e_shock_against_plie","exit_pressure_current_state","state_invalidity_score","delayed_hold_flag","fast_transition_flag","exit_pressure_trigger_flag",
        "state_confirmation_score","trading_trigger_quality","weak_state_flag","state_pending_confirmation_flag","state_confirmed_flag",
        "directional_trading_quality","direction_confidence","directional_exit_trigger_flag",
        "hpem_directional_compliance_score","hpem_giveback_ratio","hpem_decay_flag",
        "rha_directional_compliance_score","rha_invalid_follow_plie_flag",
        "vt_direction_score","vt_direction_confidence","vt_direction_flip_flag",
        "state_price_expectation_score","price_expectation_exit_flag","price_expectation_exit_pressure",
        "price_expectation_exit_pressure","price_expectation_exit_flag","price_expectation_live_score","price_expectation_exit_flag",
        "price_expectation_trading_quality","price_expectation_raw_issue_flag","rc_price_drift_flag","rc_range_break_flag",
        "rc_trend_exclusion_flag","vt_low_move_flag","vt_mixed_direction_flag","vt_weak_trend_flag",
        "st_flat_flag","st_weak_trend_flag","st_too_extreme_flag","hpem_price_expectation_fail_flag","rha_price_expectation_fail_flag"
    ]
    for c in extra_cols:
        payload[c] = _num_list(d, c)
    controls = f"""
<section class='control-bar'>
  <b>Time Window</b>
  <label>Start <input id='startTime' type='datetime-local' value='{start}' /></label>
  <label>End <input id='endTime' type='datetime-local' value='{end}' /></label>
  <button onclick='applySelectedRange()'>Apply Zoom</button><button onclick='resetFullRange()'>Full Range</button>
  <button onclick='setLastDays(1)'>24H</button><button onclick='setLastDays(7)'>7D</button><button onclick='setLastDays(30)'>30D</button><button onclick='setLastDays(90)'>90D</button><button onclick='setLastDays(365)'>1Y</button>
  <span class='hint'>Hover/click price points to update the right inspection panel. Double click unpins.</span>
</section>
"""
    inspect = """
<aside class='inspect-panel'><div class='inspect-title'>Inspection Panel</div><div class='inspect-subtitle'>Hover or click a price point. Date is included first; click pins the row.</div><div id='inspectContent' class='inspect-content empty'>No point selected yet.</div></aside>
"""
    plot_shell = """
<div class='price-inspect-grid'><div class='plot-wrap'><div id='price-chart'></div></div>__INSPECT__</div>
<div class='plot-wrap'><div id='state-band-chart'></div></div>
<div class='grid-2'><div class='plot-wrap'><div id='belief-chart'></div></div><div class='plot-wrap'><div id='stable-belief-chart'></div></div></div>
<div class='plot-wrap'><div id='evidence-chart'></div></div>
<div class='grid-2'><div class='plot-wrap'><div id='responsiveness-chart'></div></div><div class='plot-wrap'><div id='confirmation-chart'></div></div></div>
<div class='plot-wrap'><div id='directional-chart'></div></div>
""".replace("__INSPECT__", inspect)
    html_doc = "\n".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title>", _css(), f"<script>{get_plotlyjs()}</script>", "</head><body>",
        "<header>", f"<h1>{html.escape(title)}</h1>",
        "<div class='subtitle'>增强版人工检视 dashboard：右侧固定 inspection panel 承载完整 hover 信息，价格图使用 invisible anchor 提升历史数据 hover 灵敏度；所有图保留全量数据，不做抽样。</div>",
        controls, "</header><main>", _summary_cards(d), _state_definitions(), plot_shell, _quality_table(_top_problem_segments(d)), "</main>", _js(payload), "</body></html>"
    ])
    Path(output_html).write_text(html_doc, encoding="utf-8")

# -----------------------------------------------------------------------------
# V2.5 lightweight dashboard override
# -----------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame, output_html: Path, title: str = "Six-State Market Regime Dashboard") -> None:  # type: ignore[override]
    """Build a full-data, hover-friendly dashboard.

    This V2.5 override keeps all rows but avoids embedding overly long hover text in
    Plotly traces. Detailed inspection data is stored once in a compact payload and
    rendered in a fixed right-side panel on hover/click. This materially improves
    hover reliability on historical data versus heavy per-point hover templates.
    """
    d = df.copy()
    if "time" not in d.columns or "price" not in d.columns or "stable_state" not in d.columns:
        raise ValueError("Dashboard requires at least: time, price, stable_state")
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    d["price"] = pd.to_numeric(d["price"], errors="coerce")

    def nlist(col: str, digits: int = 5) -> list[Any]:
        if col not in d.columns:
            return [None] * len(d)
        vals = pd.to_numeric(d[col], errors="coerce")
        out: list[Any] = []
        for v in vals:
            if pd.isna(v) or not np.isfinite(float(v)):
                out.append(None)
            else:
                out.append(round(float(v), digits))
        return out

    def slist(col: str, default: str = "") -> list[str]:
        if col not in d.columns:
            return [default] * len(d)
        return [default if pd.isna(v) else str(v) for v in d[col]]

    def colors(states: list[str]) -> list[str]:
        return [STATE_COLORS.get(str(s), "#9AA4B2") for s in states]

    time = [pd.Timestamp(t).isoformat() for t in d["time"]]
    state = slist("stable_state", "NA")
    start, end = time[0][:16], time[-1][:16]

    hover_cols_num = [
        "price", "candidate_confidence", "belief_gap", "belief_entropy",
        "plie_main_bps", "plie_intensity", "plie_reliability", "hmm_conf", "liq_entropy",
        "realized_vol_1h_bps", "realized_vol_6h_bps", "range_compression_6h",
        "trend_strength_1h", "trend_strength_6h", "trend_consistency_6h", "jump_proxy_1h", "jump_proxy_6h",
        "state_invalidity_score", "exit_pressure_current_state", "effective_tau_in", "effective_belief_gap_min", "effective_ewma_lambda",
        "state_confirmation_score", "trading_trigger_quality", "direction_confidence", "directional_trading_quality",
        "bars_since_state_entry", "entry_to_current_return_bps", "entry_to_current_range_bps", "entry_to_current_mfe_bps", "entry_to_current_mae_bps",
        "hpem_aligned_move_bps", "hpem_aligned_mfe_bps", "hpem_giveback_bps", "hpem_giveback_ratio", "hpem_decay_flag", "hpem_no_response_flag", "hpem_adverse_flag",
        "rha_reversal_move_bps", "rha_reversal_mfe_bps", "rha_follow_plie_move_bps", "rha_stall_flag", "rha_invalid_follow_plie_flag",
        "vt_direction_score", "vt_direction_confidence", "vt_direction_flip_flag",
        "state_price_expectation_score", "price_expectation_exit_flag", "price_expectation_exit_pressure", "price_expectation_trading_quality",
        "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag", "vt_low_move_flag", "vt_mixed_direction_flag", "vt_weak_trend_flag", "st_flat_flag", "st_weak_trend_flag", "st_too_extreme_flag",
        "execution_quality",
    ]
    hover_cols_str = [
        "stable_state", "candidate_state", "from_state", "transition_reason", "top_evidence_features",
        "plie_direction", "plie_phase", "plie_strong_entry", "plie_transition_type", "plie_transition_severity", "hmm_state",
        "path_context_6h", "path_label_6h", "path_context_24h", "path_label_24h",
        "state_confirmation_status", "trading_trigger_state", "state_quality_issue",
        "directional_state", "trade_direction", "direction_source", "directional_no_trade_reason", "directional_exit_reason", "directional_exit_target",
        "rha_subtype", "state_price_expectation_issue", "price_expectation_exit_target", "price_expectation_exit_reason",
        "execution_state", "execution_trade_direction", "execution_no_trade_reason",
    ]
    h: dict[str, Any] = {c: nlist(c) for c in hover_cols_num}
    h.update({c: slist(c, "") for c in hover_cols_str})
    for st in STATES:
        h[f"b_{st}"] = nlist(f"b_{st}")
        h[f"stable_b_{st}"] = nlist(f"stable_b_{st}")

    line_cols = [
        "state_confirmation_score", "trading_trigger_quality", "directional_trading_quality", "direction_confidence",
        "state_price_expectation_score", "price_expectation_exit_flag", "price_expectation_exit_pressure", "price_expectation_trading_quality",
        "hpem_giveback_ratio", "hpem_decay_flag", "rha_invalid_follow_plie_flag", "vt_direction_score", "vt_direction_flip_flag",
        "rc_price_drift_flag", "rc_range_break_flag", "rc_trend_exclusion_flag", "vt_low_move_flag", "st_flat_flag",
        "e_price_shock_1h", "e_short_impulse_1h", "exit_pressure_current_state", "state_invalidity_score",
    ]
    payload = {
        "time": time,
        "price": nlist("price", 4),
        "state": state,
        "stateColor": colors(state),
        "idx": list(range(len(d))),
        "h": h,
        "lines": {c: nlist(c) for c in line_cols},
        "belief": {st: nlist(f"b_{st}") for st in STATES},
        "stableBelief": {st: nlist(f"stable_b_{st}") for st in STATES},
        "states": STATES,
        "colors": STATE_COLORS,
        "start": start,
        "end": end,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    css = f"""
    <style>
    :root {{ --bg:{DARK_BG}; --panel:{PANEL_BG}; --text:{TEXT}; --muted:{MUTED}; --grid:{GRID}; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top left,#172033 0,#070B14 38%,#030712 100%); color:var(--text); font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    header {{ padding:18px 22px 8px; position:sticky; top:0; z-index:20; background:rgba(7,11,20,.94); backdrop-filter:blur(10px); border-bottom:1px solid rgba(148,163,184,.18); }}
    h1 {{ margin:0 0 6px; font-size:22px; letter-spacing:.02em; }} .subtitle {{ color:var(--muted); font-size:13px; }}
    .control-bar {{ margin-top:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:12px; }}
    input,button {{ background:#111827; color:var(--text); border:1px solid rgba(148,163,184,.28); border-radius:8px; padding:6px 9px; }}
    button {{ cursor:pointer; }} button:hover {{ border-color:#38BDF8; color:#BAE6FD; }}
    main {{ padding:14px 18px 32px; }} .grid {{ display:grid; grid-template-columns:minmax(0,1fr) 390px; gap:14px; align-items:start; }}
    .plot-card {{ background:rgba(14,22,38,.82); border:1px solid rgba(148,163,184,.14); border-radius:14px; padding:10px; box-shadow:0 14px 40px rgba(0,0,0,.22); margin-bottom:14px; }}
    .inspect {{ position:sticky; top:122px; max-height:calc(100vh - 140px); overflow:auto; background:rgba(14,22,38,.95); border:1px solid rgba(56,189,248,.22); border-radius:14px; padding:12px; box-shadow:0 16px 42px rgba(0,0,0,.35); }}
    .inspect h3 {{ margin:0 0 6px; font-size:15px; }} .inspect .hint {{ color:var(--muted); font-size:12px; margin-bottom:8px; }}
    .row {{ display:grid; grid-template-columns:148px minmax(0,1fr); gap:8px; border-top:1px solid rgba(148,163,184,.10); padding:6px 0; font-size:12px; }}
    .k {{ color:#A5B4FC; }} .v {{ word-break:break-word; }} .warn {{ color:#FBBF24; }} .bad {{ color:#FB7185; }} .good {{ color:#34D399; }}
    .section {{ margin-top:10px; padding-top:8px; border-top:1px solid rgba(56,189,248,.18); font-weight:700; color:#7DD3FC; }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    @media(max-width:1100px) {{ .grid,.two {{ grid-template-columns:1fr; }} .inspect {{ position:relative; top:0; max-height:none; }} }}
    </style>
    """
    js = r"""
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script>
    const P = __PAYLOAD__;
    const layoutBase = {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(3,7,18,.78)', font:{color:'#E5E7EB'}, margin:{l:58,r:20,t:30,b:42}, hovermode:'closest', xaxis:{gridcolor:'rgba(148,163,184,.14)', rangeslider:{visible:true, thickness:.06}, showspikes:true, spikemode:'across', spikedash:'dot'}, yaxis:{gridcolor:'rgba(148,163,184,.14)', zeroline:false}, hoverdistance:80, spikedistance:80, legend:{orientation:'h', y:1.08}};
    function val(x,d=3){return (x===null||x===undefined||Number.isNaN(x))?'NA':Number(x).toFixed(d)}
    function row(label, value, cls=''){return `<div class='row'><div class='k'>${label}</div><div class='v ${cls}'>${value??'NA'}</div></div>`}
    function sec(title){return `<div class='section'>${title}</div>`}
    function H(c,i){return (P.h[c]||[])[i]}
    function render(i){
      const date=P.time[i]; let h=`<h3>${H('stable_state',i)||P.state[i]} · ${date}</h3><div class='hint'>Index ${i}; click point to pin, double-click chart to unpin.</div>`;
      h+=sec('State & execution');
      h+=row('Price', val(P.price[i],2)); h+=row('Candidate', H('candidate_state',i)); h+=row('Stable', H('stable_state',i)||P.state[i]);
      h+=row('Directional state', H('directional_state',i)); h+=row('Trade direction', H('trade_direction',i)); h+=row('Directional quality', val(H('directional_trading_quality',i),3), H('directional_trading_quality',i)<.45?'warn':'good');
      h+=row('Execution state', H('execution_state',i)); h+=row('Execution quality', val(H('execution_quality',i),3), H('execution_quality',i)<.45?'warn':'good'); h+=row('No-trade reason', H('execution_no_trade_reason',i));
      h+=sec('Price expectation compliance'); h+=row('Expectation score', val(H('state_price_expectation_score',i),3), H('state_price_expectation_score',i)<.45?'bad':'good'); h+=row('Issue', H('state_price_expectation_issue',i)); h+=row('Exit flag / pressure', `${H('price_expectation_exit_flag',i)} / ${val(H('price_expectation_exit_pressure',i),3)}`); h+=row('Exit target/reason', `${H('price_expectation_exit_target',i)||''} · ${H('price_expectation_exit_reason',i)||''}`);
      h+=sec('Entry-to-current path'); h+=row('Bars since entry', H('bars_since_state_entry',i)); h+=row('Entry return / range', `${val(H('entry_to_current_return_bps',i),2)} / ${val(H('entry_to_current_range_bps',i),2)} bps`); h+=row('MFE / MAE', `${val(H('entry_to_current_mfe_bps',i),2)} / ${val(H('entry_to_current_mae_bps',i),2)} bps`);
      h+=sec('HPEM/RHA/VT lifecycle'); h+=row('HPEM move/MFE/giveback', `${val(H('hpem_aligned_move_bps',i),2)} / ${val(H('hpem_aligned_mfe_bps',i),2)} / ${val(H('hpem_giveback_bps',i),2)}`); h+=row('HPEM decay/no-response/adverse', `${H('hpem_decay_flag',i)} / ${H('hpem_no_response_flag',i)} / ${H('hpem_adverse_flag',i)}`); h+=row('RHA subtype', H('rha_subtype',i)); h+=row('RHA reversal/follow PLIE', `${val(H('rha_reversal_move_bps',i),2)} / ${val(H('rha_follow_plie_move_bps',i),2)}`); h+=row('VT direction score/state', `${val(H('vt_direction_score',i),3)} / ${H('vt_direction_state',i)}`);
      h+=sec('PLIE / HMM / Path'); h+=row('PLIE dir/main', `${H('plie_direction',i)} / ${val(H('plie_main_bps',i),2)}`); h+=row('PLIE intensity/reliability', `${val(H('plie_intensity',i),3)} / ${val(H('plie_reliability',i),3)}`); h+=row('HMM state/conf/entropy', `${H('hmm_state',i)} / ${val(H('hmm_conf',i),3)} / ${val(H('liq_entropy',i),3)}`); h+=row('Path 6h', `${H('path_context_6h',i)} / ${H('path_label_6h',i)}`); h+=row('Path 24h', `${H('path_context_24h',i)} / ${H('path_label_24h',i)}`);
      h+=sec('Price context'); h+=row('RV 1h / 6h', `${val(H('realized_vol_1h_bps',i),2)} / ${val(H('realized_vol_6h_bps',i),2)}`); h+=row('Trend 1h / 6h / consistency', `${val(H('trend_strength_1h',i),3)} / ${val(H('trend_strength_6h',i),3)} / ${val(H('trend_consistency_6h',i),3)}`); h+=row('Range comp / Jump 1h', `${val(H('range_compression_6h',i),3)} / ${val(H('jump_proxy_1h',i),3)}`);
      h+=sec('Belief vector'); h+=row('Raw belief', P.states.map(s=>`${s}:${val(H('b_'+s,i),2)}`).join(' · ')); h+=row('Stable belief', P.states.map(s=>`${s}:${val(H('stable_b_'+s,i),2)}`).join(' · '));
      h+=sec('Reason'); h+=row('Top evidence', H('top_evidence_features',i)); h+=row('Transition', H('transition_reason',i));
      document.getElementById('inspect').innerHTML=h;
    }
    function shapesFor(range){
      const shapes=[]; let s=0; for(let i=1;i<=P.state.length;i++){ if(i===P.state.length || P.state[i]!==P.state[s]){ shapes.push({type:'rect', xref:'x', yref:'paper', x0:P.time[s], x1:P.time[Math.max(i-1,s)], y0:0, y1:1, fillcolor:P.colors[P.state[s]]||'#999', opacity:.13, line:{width:0}, layer:'below'}); s=i; }} return shapes;
    }
    function priceRange(x0,x1){ let lo=Infinity,hi=-Infinity; for(let i=0;i<P.time.length;i++){ if((!x0||P.time[i]>=x0)&&(!x1||P.time[i]<=x1)){ const v=P.price[i]; if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}} if(!isFinite(lo)) return null; const pad=Math.max((hi-lo)*.08, 10); return [lo-pad,hi+pad]; }
    function relayoutAll(x0,x1){ const xr=[x0,x1]; const yr=priceRange(x0,x1); Plotly.relayout('price-chart', {'xaxis.range':xr, 'yaxis.range':yr}); ['belief-chart','stable-belief-chart','lifecycle-chart','state-band-chart'].forEach(id=>Plotly.relayout(id, {'xaxis.range':xr})); }
    function applySelectedRange(){ relayoutAll(document.getElementById('startTime').value, document.getElementById('endTime').value); }
    function resetFullRange(){ document.getElementById('startTime').value=P.start; document.getElementById('endTime').value=P.end; relayoutAll(P.start,P.end); }
    function setLastDays(days){ const end=new Date(P.end); const start=new Date(end.getTime()-days*86400000); document.getElementById('startTime').value=start.toISOString().slice(0,16); document.getElementById('endTime').value=P.end; applySelectedRange(); }
    window.applySelectedRange=applySelectedRange; window.resetFullRange=resetFullRange; window.setLastDays=setLastDays;
    const priceTrace={x:P.time,y:P.price,type:'scattergl',mode:'lines',name:'Price',line:{color:'#E5E7EB',width:1.4},hoverinfo:'skip'};
    const pointTrace={x:P.time,y:P.price,type:'scattergl',mode:'markers',name:'State points',marker:{size:4,color:P.stateColor,opacity:.70},customdata:P.idx,hovertemplate:'%{x}<br>Price %{y:.2f}<extra></extra>'};
    const anchorTrace={x:P.time,y:P.price,type:'scattergl',mode:'markers',name:'hover anchor',marker:{size:13,color:'rgba(255,255,255,0.01)'},customdata:P.idx,hovertemplate:'%{x}<br>open inspection panel<extra></extra>',showlegend:false};
    Plotly.newPlot('price-chart',[priceTrace,pointTrace,anchorTrace], {...layoutBase, title:'Price with stable-state background', shapes:shapesFor(), yaxis:{...layoutBase.yaxis, range:priceRange(P.start,P.end)}}, {responsive:true});
    const band={x:P.time,y:P.state.map(s=>P.states.indexOf(s)),type:'scattergl',mode:'markers',name:'stable_state',marker:{size:5,color:P.stateColor},customdata:P.idx,hovertemplate:'%{x}<br>%{text}<extra></extra>',text:P.state};
    Plotly.newPlot('state-band-chart',[band], {...layoutBase,title:'Stable state timeline',yaxis:{tickvals:P.states.map((s,i)=>i),ticktext:P.states,gridcolor:'rgba(148,163,184,.12)'}}, {responsive:true});
    function beliefPlot(id,obj,title){ const traces=P.states.map(s=>({x:P.time,y:obj[s],type:'scattergl',mode:'lines',name:s,line:{color:P.colors[s],width:1.2}})); Plotly.newPlot(id,traces,{...layoutBase,title:title,yaxis:{...layoutBase.yaxis,range:[0,1]}},{responsive:true}); }
    beliefPlot('belief-chart',P.belief,'Raw filtered belief'); beliefPlot('stable-belief-chart',P.stableBelief,'Stable belief');
    const lc=['state_confirmation_score','trading_trigger_quality','directional_trading_quality','direction_confidence','state_price_expectation_score','price_expectation_exit_pressure','price_expectation_exit_flag','hpem_giveback_ratio','hpem_decay_flag','rha_invalid_follow_plie_flag','vt_direction_score','rc_price_drift_flag','vt_low_move_flag','st_flat_flag','e_short_impulse_1h','exit_pressure_current_state'];
    const traces=lc.filter(c=>P.lines[c]).map(c=>({x:P.time,y:P.lines[c],type:'scattergl',mode:'lines',name:c,line:{width:1.05}}));
    Plotly.newPlot('lifecycle-chart',traces,{...layoutBase,title:'Confirmation, directional quality, and price-expectation lifecycle',yaxis:{...layoutBase.yaxis,range:[0,1]}},{responsive:true});
    ['price-chart','state-band-chart'].forEach(id=>{document.getElementById(id).on('plotly_hover',ev=>{const i=ev.points?.[0]?.customdata; if(i!==undefined) render(i)}); document.getElementById(id).on('plotly_click',ev=>{const i=ev.points?.[0]?.customdata; if(i!==undefined) render(i)});});
    document.getElementById('price-chart').on('plotly_relayout',ev=>{ if(ev['xaxis.range[0]']) relayoutAll(ev['xaxis.range[0]'],ev['xaxis.range[1]']); });
    render(P.idx[P.idx.length-1]);
    </script>
    """.replace('__PAYLOAD__', data_json)
    html_doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{css}</head><body>
    <header><h1>{html.escape(title)}</h1><div class='subtitle'>V2.5: price expectation compliance, directional lifecycle, and execution-quality inspection. Full data, no sampling.</div>
    <div class='control-bar'><b>Time Window</b><label>Start <input id='startTime' type='datetime-local' value='{start}'></label><label>End <input id='endTime' type='datetime-local' value='{end}'></label><button onclick='applySelectedRange()'>Apply Zoom</button><button onclick='resetFullRange()'>Full Range</button><button onclick='setLastDays(1)'>24H</button><button onclick='setLastDays(7)'>7D</button><button onclick='setLastDays(30)'>30D</button><button onclick='setLastDays(90)'>90D</button><button onclick='setLastDays(365)'>1Y</button></div></header>
    <main><div class='grid'><div><div class='plot-card'><div id='price-chart' style='height:560px'></div></div><div class='plot-card'><div id='state-band-chart' style='height:210px'></div></div></div><aside class='inspect' id='inspect'>Loading…</aside></div>
    <div class='two'><div class='plot-card'><div id='belief-chart' style='height:330px'></div></div><div class='plot-card'><div id='stable-belief-chart' style='height:330px'></div></div></div>
    <div class='plot-card'><div id='lifecycle-chart' style='height:430px'></div></div></main>{js}</body></html>"""
    Path(output_html).write_text(html_doc, encoding='utf-8')

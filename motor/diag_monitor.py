"""Type02 수신 품질/지연/누락을 진단하는 모니터."""

import argparse
import math
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from lib.common import unpack_ext_id
from lib.config import DEFAULT_CH
from lib.limits import feedback_position_violation
from lib.parse import Feedback, parse_feedback_like_type2
from lib.runtime import run_with_bus
from lib.stream_state import load_expected_hz_map

# CAN 2.0B(extended), DLC=8 기준 대략치.
# - nominal: bit stuffing 제외
# - worst: stuffing 최대치 가정
CAN_EXT_DLC8_BITS_NOMINAL = 131
CAN_EXT_DLC8_BITS_WORST = 160
CAN_BITRATE_1MBPS = 1_000_000.0


@dataclass
class RunningStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def add(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.min_v = min(self.min_v, value)
        self.max_v = max(self.max_v, value)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass
class MotorDiag:
    feedback_total: int = 0
    est_missing: int = 0
    gap_stat: RunningStat = field(default_factory=RunningStat)
    jitter_stat: RunningStat = field(default_factory=RunningStat)
    lag_stat: RunningStat = field(default_factory=RunningStat)
    pos_jump_stat: RunningStat = field(default_factory=RunningStat)
    vel_jump_stat: RunningStat = field(default_factory=RunningStat)
    tor_jump_stat: RunningStat = field(default_factory=RunningStat)
    last_rx_ts: float | None = None
    prev_fb: Feedback | None = None


def _dedupe(values: list[int] | None) -> list[int] | None:
    if values is None:
        return None
    return list(dict.fromkeys(values))


def _fmt_triplet_ms(stat: RunningStat) -> str:
    if stat.n == 0:
        return "n/a"
    return f"{stat.mean * 1000:.3f}/{stat.std * 1000:.3f}/{stat.max_v * 1000:.3f}"


def _fmt_pair_ms(stat: RunningStat) -> str:
    if stat.n == 0:
        return "n/a"
    return f"{stat.mean * 1000:.3f}/{stat.max_v * 1000:.3f}"


def _fmt_pair(stat: RunningStat) -> str:
    if stat.n == 0:
        return "n/a"
    return f"{stat.mean:.5f}/{stat.max_v:.5f}"


def _to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000.0


def _stat_mean(stat: RunningStat) -> float | None:
    return stat.mean if stat.n > 0 else None


def _stat_std(stat: RunningStat) -> float | None:
    return stat.std if stat.n > 0 else None


def _stat_max(stat: RunningStat) -> float | None:
    return stat.max_v if stat.n > 0 else None


def _bus_load_pct(rate_hz: float, bits_per_frame: int) -> float:
    return (rate_hz * float(bits_per_frame) / CAN_BITRATE_1MBPS) * 100.0


def _col_name(col_idx_1based: int) -> str:
    out = []
    x = col_idx_1based
    while x > 0:
        x, rem = divmod(x - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


def _xlsx_cell_xml(row_idx_1based: int, col_idx_1based: int, value) -> str:
    ref = f"{_col_name(col_idx_1based)}{row_idx_1based}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="n"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}" t="n"><v>{value}</v></c>'
    text = xml_escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(rows: list[list]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        lines.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            lines.append(_xlsx_cell_xml(r_idx, c_idx, value))
        lines.append("</row>")
    lines.extend(["</sheetData>", "</worksheet>"])
    return "".join(lines)


def _dict_rows_to_table(rows: list[dict]) -> list[list]:
    if not rows:
        return [["empty"]]
    headers = list(rows[0].keys())
    table = [headers]
    for row in rows:
        table.append([row.get(h) for h in headers])
    return table


def _warn_feedback_position_if_needed(
    fb: Feedback,
    last_warn_by_motor: dict[int, float],
    warn_period_sec: float = 2.0,
):
    violation = feedback_position_violation(fb)
    if violation is None:
        return

    now = time.monotonic()
    last_warn = last_warn_by_motor.get(fb.motor_id)
    if last_warn is not None and now - last_warn < warn_period_sec:
        return
    last_warn_by_motor[fb.motor_id] = now

    q_limited, q_min, q_max = violation
    print(
        f"[WARN id={fb.motor_id}] feedback pos outside limit: "
        f"{fb.pos:+.4f} -> {q_limited:+.4f} "
        f"(limit=[{q_min:+.4f},{q_max:+.4f}])"
    )


def _write_xlsx(path: Path, sheets: list[tuple[str, list[list]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_defs = []
    for idx, (name, rows) in enumerate(sheets, start=1):
        sheet_defs.append((idx, name, _sheet_xml(rows)))

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx, _, _ in sheet_defs:
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    wb_sheets = []
    wb_rels = []
    app_titles = []
    for idx, name, _ in sheet_defs:
        safe_name = xml_escape(name[:31])
        wb_sheets.append(f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>')
        wb_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        app_titles.append(f"<vt:lpstr>{safe_name}</vt:lpstr>")

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(wb_sheets)}</sheets>"
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(wb_rels)}"
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>diag_monitor report</dc:title>"
        "<dc:creator>diag_monitor.py</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:modified>'
        "</cp:coreProperties>"
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>diag_monitor.py</Application>"
        f"<TitlesOfParts><vt:vector size=\"{len(sheet_defs)}\" baseType=\"lpstr\">{''.join(app_titles)}</vt:vector></TitlesOfParts>"
        f"<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(sheet_defs)}</vt:i4></vt:variant></vt:vector></HeadingPairs>"
        "</Properties>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for idx, _, sheet_xml in sheet_defs:
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stats_sec", type=float, default=1.0, help="진단 출력 주기(초)")
    ap.add_argument("--expect_hz", type=float, default=None, help="모든 can_id에 동일 기대 주파수 사용")
    ap.add_argument(
        "--stream_hz_max_age",
        type=float,
        default=3.0,
        help="mit_stream 상태 파일에서 기대 주파수 읽을 때 허용하는 최대 age(초)",
    )
    ap.add_argument("--no_stream_hz", action="store_true", help="mit_stream 상태 파일 자동 참조 비활성화")
    ap.add_argument("--quiet", action="store_true", help="프레임별 출력 비활성화")
    ap.add_argument("--report_dir", default=None, help="엑셀 저장 경로(기본: 이 파일과 같은 폴더)")
    ap.add_argument("--report_prefix", default="diag_report", help="엑셀 파일명 prefix")
    ap.add_argument("--no_report", action="store_true", help="종료 시 엑셀 저장 비활성화")
    args = ap.parse_args()

    if args.every <= 0:
        raise SystemExit("--every must be > 0")
    if args.stats_sec <= 0:
        raise SystemExit("--stats_sec must be > 0")
    if args.expect_hz is not None and args.expect_hz <= 0:
        raise SystemExit("--expect_hz must be > 0")
    if args.stream_hz_max_age <= 0:
        raise SystemExit("--stream_hz_max_age must be > 0")

    can_ids = _dedupe(args.can_id)
    can_filter = set(can_ids) if can_ids else None

    def _run(bus):
        cnt = 0
        raw_total = 0
        err_frames = 0
        dropped_non_feedback = 0
        dropped_other_id = 0
        motor_stats: dict[int, MotorDiag] = {}
        expected_hz_map: dict[int, float] = {}
        overall_rows: list[dict] = []
        motor_rows: list[dict] = []
        last_limit_warn_by_motor: dict[int, float] = {}

        started_wall = time.time()
        started_mono = time.perf_counter()
        next_report_mono = started_mono + args.stats_sec

        def refresh_expected_hz():
            nonlocal expected_hz_map
            if args.expect_hz is not None or args.no_stream_hz:
                expected_hz_map = {}
                return
            expected_hz_map = load_expected_hz_map(
                can_ids=can_filter,
                channel=args.ch,
                max_age_sec=args.stream_hz_max_age,
            )

        def get_expected_hz(can_id: int) -> float | None:
            if args.expect_hz is not None:
                return args.expect_hz
            return expected_hz_map.get(can_id)

        def snapshot_rows(elapsed: float):
            total_fb = sum(st.feedback_total for st in motor_stats.values())
            raw_rate_hz = raw_total / elapsed
            fb_rate_hz = total_fb / elapsed
            bus_nom_pct = _bus_load_pct(raw_rate_hz, CAN_EXT_DLC8_BITS_NOMINAL)
            bus_worst_pct = _bus_load_pct(raw_rate_hz, CAN_EXT_DLC8_BITS_WORST)
            overall_rows.append(
                {
                    "t_sec": elapsed,
                    "raw_total": raw_total,
                    "fb_total": total_fb,
                    "err_frames": err_frames,
                    "drop_nonfb": dropped_non_feedback,
                    "drop_id": dropped_other_id,
                    "raw_rate_hz": raw_rate_hz,
                    "fb_rate_hz": fb_rate_hz,
                    "bus_nom_pct_1mbps": bus_nom_pct,
                    "bus_worst_pct_1mbps": bus_worst_pct,
                }
            )

            for can_id in sorted(motor_stats.keys()):
                st = motor_stats[can_id]
                exp_hz = get_expected_hz(can_id)
                rate_hz = st.feedback_total / elapsed
                lag_max = _stat_max(st.lag_stat)
                lag_buf_max = (lag_max * exp_hz) if (lag_max is not None and exp_hz is not None) else None
                motor_rows.append(
                    {
                        "t_sec": elapsed,
                        "can_id": can_id,
                        "fb_total": st.feedback_total,
                        "rate_hz": rate_hz,
                        "exp_hz": exp_hz,
                        "gap_ms_mean": _to_ms(_stat_mean(st.gap_stat)),
                        "gap_ms_std": _to_ms(_stat_std(st.gap_stat)),
                        "gap_ms_max": _to_ms(_stat_max(st.gap_stat)),
                        "jitter_ms_mean": _to_ms(_stat_mean(st.jitter_stat)),
                        "jitter_ms_max": _to_ms(_stat_max(st.jitter_stat)),
                        "lag_ms_mean": _to_ms(_stat_mean(st.lag_stat)),
                        "lag_ms_max": _to_ms(lag_max),
                        "lag_buf_max_frames": lag_buf_max,
                        "est_miss": st.est_missing,
                        "pos_jump_mean": _stat_mean(st.pos_jump_stat),
                        "pos_jump_max": _stat_max(st.pos_jump_stat),
                        "vel_jump_mean": _stat_mean(st.vel_jump_stat),
                        "vel_jump_max": _stat_max(st.vel_jump_stat),
                        "tor_jump_mean": _stat_mean(st.tor_jump_stat),
                        "tor_jump_max": _stat_max(st.tor_jump_stat),
                        "fb_bus_nom_pct_1mbps": _bus_load_pct(rate_hz, CAN_EXT_DLC8_BITS_NOMINAL),
                        "fb_bus_worst_pct_1mbps": _bus_load_pct(rate_hz, CAN_EXT_DLC8_BITS_WORST),
                    }
                )

            return total_fb, raw_rate_hz, bus_nom_pct, bus_worst_pct

        def maybe_print_diag(now_mono: float, force: bool = False):
            nonlocal next_report_mono
            if not force and now_mono < next_report_mono:
                return

            refresh_expected_hz()
            elapsed = max(now_mono - started_mono, 1e-9)
            total_fb, _, bus_nom_pct, bus_worst_pct = snapshot_rows(elapsed)
            print(
                f"[diag] t={elapsed:.1f}s raw={raw_total} fb={total_fb} err={err_frames} "
                f"drop_nonfb={dropped_non_feedback} drop_id={dropped_other_id} "
                f"bus1M%(nom/worst)={bus_nom_pct:.3f}/{bus_worst_pct:.3f}"
            )

            for can_id in sorted(motor_stats.keys()):
                st = motor_stats[can_id]
                exp_hz = get_expected_hz(can_id)
                exp_text = f"{exp_hz:.3f}" if exp_hz is not None else "n/a"
                rate_hz = st.feedback_total / elapsed
                lag_buf = "n/a"
                if exp_hz is not None and st.lag_stat.n > 0:
                    lag_buf = f"{st.lag_stat.max_v * exp_hz:.2f}"
                fb_bus_nom = _bus_load_pct(rate_hz, CAN_EXT_DLC8_BITS_NOMINAL)
                fb_bus_worst = _bus_load_pct(rate_hz, CAN_EXT_DLC8_BITS_WORST)
                print(
                    f"[diag id={can_id}] rate={rate_hz:.2f}Hz exp_hz={exp_text} "
                    f"gap_ms(mean/std/max)={_fmt_triplet_ms(st.gap_stat)} "
                    f"jitter_ms(mean/max)={_fmt_pair_ms(st.jitter_stat)} "
                    f"lag_ms(mean/max)={_fmt_pair_ms(st.lag_stat)} lag_buf(max)={lag_buf}f "
                    f"est_miss={st.est_missing} fb_bus1M%(nom/worst)={fb_bus_nom:.3f}/{fb_bus_worst:.3f} "
                    f"jump_abs(pos/vel/tor mean/max)="
                    f"{_fmt_pair(st.pos_jump_stat)}/{_fmt_pair(st.vel_jump_stat)}/{_fmt_pair(st.tor_jump_stat)}"
                )

            if not force:
                next_report_mono = now_mono + args.stats_sec

        def save_report(now_mono: float):
            if args.no_report:
                return None

            refresh_expected_hz()
            elapsed = max(now_mono - started_mono, 1e-9)
            ended_wall = time.time()
            ended_iso = datetime.fromtimestamp(ended_wall, tz=timezone.utc).isoformat()
            started_iso = datetime.fromtimestamp(started_wall, tz=timezone.utc).isoformat()

            final_rows = []
            for can_id in sorted(motor_stats.keys()):
                st = motor_stats[can_id]
                exp_hz = get_expected_hz(can_id)
                rate_hz = st.feedback_total / elapsed
                lag_max = _stat_max(st.lag_stat)
                lag_buf_max = (lag_max * exp_hz) if (lag_max is not None and exp_hz is not None) else None
                final_rows.append(
                    {
                        "can_id": can_id,
                        "fb_total": st.feedback_total,
                        "rate_hz": rate_hz,
                        "exp_hz": exp_hz,
                        "est_miss": st.est_missing,
                        "gap_ms_mean": _to_ms(_stat_mean(st.gap_stat)),
                        "gap_ms_std": _to_ms(_stat_std(st.gap_stat)),
                        "gap_ms_max": _to_ms(_stat_max(st.gap_stat)),
                        "jitter_ms_mean": _to_ms(_stat_mean(st.jitter_stat)),
                        "jitter_ms_max": _to_ms(_stat_max(st.jitter_stat)),
                        "lag_ms_mean": _to_ms(_stat_mean(st.lag_stat)),
                        "lag_ms_max": _to_ms(lag_max),
                        "lag_buf_max_frames": lag_buf_max,
                        "pos_jump_mean": _stat_mean(st.pos_jump_stat),
                        "pos_jump_max": _stat_max(st.pos_jump_stat),
                        "vel_jump_mean": _stat_mean(st.vel_jump_stat),
                        "vel_jump_max": _stat_max(st.vel_jump_stat),
                        "tor_jump_mean": _stat_mean(st.tor_jump_stat),
                        "tor_jump_max": _stat_max(st.tor_jump_stat),
                        "fb_bus_nom_pct_1mbps": _bus_load_pct(rate_hz, CAN_EXT_DLC8_BITS_NOMINAL),
                        "fb_bus_worst_pct_1mbps": _bus_load_pct(rate_hz, CAN_EXT_DLC8_BITS_WORST),
                    }
                )

            report_dir = Path(args.report_dir) if args.report_dir else Path(__file__).resolve().parent
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(ended_wall))
            report_path = report_dir / f"{args.report_prefix}_{ts}.xlsx"

            meta_rows = [
                ["key", "value"],
                ["saved_at_utc", ended_iso],
                ["started_at_utc", started_iso],
                ["elapsed_sec", elapsed],
                ["channel", args.ch],
                ["can_filter", ",".join(str(x) for x in sorted(can_filter)) if can_filter else "all"],
                ["stats_sec", args.stats_sec],
                ["every", args.every],
                ["expect_hz_arg", args.expect_hz if args.expect_hz is not None else ""],
                ["no_stream_hz", args.no_stream_hz],
                ["stream_hz_max_age", args.stream_hz_max_age],
                ["can_bits_nominal", CAN_EXT_DLC8_BITS_NOMINAL],
                ["can_bits_worst", CAN_EXT_DLC8_BITS_WORST],
                ["can_bitrate_bps", CAN_BITRATE_1MBPS],
            ]

            sheets = [
                ("meta", meta_rows),
                ("overall_snapshots", _dict_rows_to_table(overall_rows)),
                ("motor_snapshots", _dict_rows_to_table(motor_rows)),
                ("motor_final", _dict_rows_to_table(final_rows)),
            ]
            _write_xlsx(report_path, sheets)
            return report_path

        refresh_expected_hz()
        report_path = None
        try:
            while True:
                msg = bus.recv(timeout=1.0)
                now_mono = time.perf_counter()
                now_wall = time.time()
                maybe_print_diag(now_mono)

                if msg is None:
                    continue
                raw_total += 1

                if getattr(msg, "is_error_frame", False):
                    err_frames += 1
                    continue

                fb = parse_feedback_like_type2(msg.arbitration_id, msg.data)
                if fb is None:
                    dropped_non_feedback += 1
                    continue
                if can_filter is not None and fb.motor_id not in can_filter:
                    dropped_other_id += 1
                    continue

                _warn_feedback_position_if_needed(fb, last_limit_warn_by_motor)

                st = motor_stats.setdefault(fb.motor_id, MotorDiag())
                st.feedback_total += 1
                cnt += 1

                msg_ts = getattr(msg, "timestamp", None)
                if isinstance(msg_ts, (int, float)) and msg_ts > 0:
                    lag = now_wall - float(msg_ts)
                    if lag >= 0:
                        st.lag_stat.add(lag)
                    rx_ts = float(msg_ts)
                else:
                    rx_ts = now_wall

                if st.last_rx_ts is not None:
                    dt = rx_ts - st.last_rx_ts
                    if dt >= 0:
                        st.gap_stat.add(dt)
                        exp_hz = get_expected_hz(fb.motor_id)
                        if exp_hz is not None:
                            expected_period = 1.0 / exp_hz
                            st.jitter_stat.add(abs(dt - expected_period))
                            st.est_missing += max(int(round(dt / expected_period)) - 1, 0)
                st.last_rx_ts = rx_ts

                if st.prev_fb is not None:
                    st.pos_jump_stat.add(abs(fb.pos - st.prev_fb.pos))
                    st.vel_jump_stat.add(abs(fb.vel - st.prev_fb.vel))
                    st.tor_jump_stat.add(abs(fb.tor - st.prev_fb.tor))
                st.prev_fb = fb

                if args.quiet or cnt % args.every != 0:
                    continue

                comm_type, _, _ = unpack_ext_id(msg.arbitration_id)
                print(
                    f"[type=0x{comm_type:02X} id={fb.motor_id}] "
                    f"pos={fb.pos:+.4f} vel={fb.vel:+.4f} tor={fb.tor:+.4f} "
                    f"temp={fb.temp_c:.1f}C mode={fb.mode_status} fault=0x{fb.fault_bits:02X}"
                )
        finally:
            now_mono = time.perf_counter()
            maybe_print_diag(now_mono, force=True)
            report_path = save_report(now_mono)
            if report_path is not None:
                print(f"[diag] report saved: {report_path}")

    try:
        run_with_bus(args.ch, _run)
    except KeyboardInterrupt:
        print("\ninterrupt: monitor stopped")


if __name__ == "__main__":
    main()


# 실행 예시 (기본 채널: can0)
# python3 diag_monitor.py --can_id 1 --quiet --stats_sec 1.0
# python3 diag_monitor.py --can_id 1 2 --quiet --stats_sec 1.0

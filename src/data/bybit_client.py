from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

BYBIT_BASE_URL = "https://api.bybit.com"


@dataclass(slots=True)
class FetchConfig:
    category: str = "linear"
    symbol: str = "BTCUSDT"
    interval: str = "1"  # minutes
    limit: int = 1000
    throttle_sec: float = 0.2
    max_retries: int = 5


def _ensure_ms(ts: datetime | int | float) -> int:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp() * 1000)
    value = int(ts)
    if value < 10_000_000_000:
        return value * 1000
    return value


def _interval_ms(interval: str) -> int:
    minutes = int(interval)
    return minutes * 60 * 1000


def _request_with_retry(endpoint: str, params: dict, cfg: FetchConfig) -> list[list[str]]:
    url = f"{BYBIT_BASE_URL}{endpoint}"
    backoff = cfg.throttle_sec
    for attempt in range(cfg.max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") != 0:
                raise RuntimeError(
                    f"Bybit error retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}"
                )
            result = payload.get("result") or {}
            rows = result.get("list") or []
            return rows
        except (requests.exceptions.RequestException, json.JSONDecodeError, RuntimeError) as exc:
            if attempt == cfg.max_retries - 1:
                raise RuntimeError(f"Failed to fetch Bybit data: {exc}") from exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
    return []


def _rows_to_df(rows: Iterable[Iterable[str]]) -> pd.DataFrame:
    columns = ["start_ms", "open", "high", "low", "close", "volume", "turnover"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])

    frame["start_ms"] = frame["start_ms"].astype("int64")
    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values("start_ms", inplace=True)
    frame["timestamp"] = pd.to_datetime(frame["start_ms"], unit="ms", utc=True)
    frame.set_index("timestamp", inplace=True)
    return frame[["open", "high", "low", "close", "volume", "turnover"]]


def _fetch_klines(
    cfg: FetchConfig,
    start: datetime,
    end: datetime,
    show_progress: bool = False,
) -> pd.DataFrame:
    start_ms = _ensure_ms(start)
    end_ms = _ensure_ms(end)
    step_ms = cfg.limit * _interval_ms(cfg.interval)

    total_steps = max(1, math.ceil((end_ms - start_ms + 1) / step_ms))
    progress = tqdm(
        total=total_steps,
        desc=f"{cfg.symbol} {cfg.interval}m",
        disable=not show_progress,
        leave=False,
    )

    rows: list[list[str]] = []
    cursor = start_ms
    while cursor <= end_ms:
        chunk_end = min(end_ms, cursor + step_ms - 1)
        params = {
            "category": cfg.category,
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "start": cursor,
            "end": chunk_end,
            "limit": cfg.limit,
        }
        chunk_rows = _request_with_retry("/v5/market/kline", params, cfg)
        if chunk_rows:
            chunk_rows.sort(key=lambda item: int(item[0]))
            rows.extend(chunk_rows)
            cursor = int(chunk_rows[-1][0]) + _interval_ms(cfg.interval)
        else:
            cursor = chunk_end + _interval_ms(cfg.interval)
        progress.update(1)
        time.sleep(cfg.throttle_sec)

    progress.close()
    return _rows_to_df(rows)


def _month_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    ranges: list[tuple[datetime, datetime]] = []
    current = datetime(start_utc.year, start_utc.month, 1, tzinfo=timezone.utc)
    while current <= end_utc:
        if current.month == 12:
            nxt = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
        range_start = max(current, start_utc)
        range_end = min(end_utc, nxt - timedelta(milliseconds=1))
        if range_start <= range_end:
            ranges.append((range_start, range_end))
        current = nxt
    return ranges


def _collect_df(
    cfg: FetchConfig,
    start: datetime,
    end: datetime,
    show_progress: bool = False,
    sequential: bool = True,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    ranges = _month_ranges(start, end)

    if len(ranges) <= 1 or sequential:
        frames: list[pd.DataFrame] = []
        iterator = enumerate(ranges, start=1)
        progress_bar: Optional[tqdm] = None
        if show_progress and len(ranges) > 1:
            progress_bar = tqdm(total=len(ranges), desc="Months", leave=False)
        for idx, (range_start, range_end) in iterator:
            frame = _fetch_klines(cfg, range_start, range_end, show_progress=False)
            if frame is not None and not frame.empty:
                frames.append(frame)
            if progress_bar is not None:
                progress_bar.update(1)
            if progress_hook is not None:
                progress_hook(idx, len(ranges))
            time.sleep(cfg.throttle_sec)
        if progress_bar is not None:
            progress_bar.close()
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
        return pd.concat(frames).sort_index()

    results: list[Optional[pd.DataFrame]] = [None] * len(ranges)
    max_workers = min(len(ranges), 6)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_klines, cfg, rng[0], rng[1], False): idx for idx, rng in enumerate(ranges)}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Months",
            disable=not show_progress,
            leave=False,
        ):
            idx = futures[future]
            results[idx] = future.result()
            if progress_hook is not None:
                progress_hook(len([frame for frame in results if frame is not None]), len(ranges))

    frames = [frame for frame in results if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    return pd.concat(frames).sort_index()


def fetch_bybit_klines(
    start: datetime,
    end: datetime,
    *,
    interval_minutes: str = "1",
    symbol: str = "BTCUSDT",
    category: str = "linear",
    show_progress: bool = False,
    sequential: bool = True,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    cfg = FetchConfig(category=category, symbol=symbol, interval=str(interval_minutes))
    data = _collect_df(
        cfg,
        start,
        end,
        show_progress=show_progress,
        sequential=sequential,
        progress_hook=progress_hook,
    )
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()
    data = data.reset_index().rename(columns={"timestamp": "timestamp"})
    data["timestamp"] = data["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return data

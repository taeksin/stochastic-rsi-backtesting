from __future__ import annotations

import pandas as pd
from xlsxwriter.utility import xl_col_to_name


def style_trade_sheet(
    workbook,
    worksheet,
    dataframe: pd.DataFrame,
    header_row: int = 0,
) -> None:
    """Apply styling rules to the exported trades worksheet."""
    if dataframe is None or dataframe.empty:
        return

    start_row = header_row + 1
    end_row = start_row + len(dataframe) - 1
    end_col = len(dataframe.columns) - 1
    if end_row < start_row or end_col < 0:
        return

    column_map = {name: idx for idx, name in enumerate(dataframe.columns)}
    exit_type_idx = column_map.get("청산 유형")
    position_idx = column_map.get("포지션")
    percent_idx = column_map.get("수익률(%)")
    rsi_idx = column_map.get("진입 RSI")
    datetime_indices = [idx for name, idx in column_map.items() if name in {"진입 시각", "청산 시각"}]
    currency_indices = [
        column_map.get("수익 (₩)"),
        column_map.get("투입 자본 (₩)"),
        column_map.get("진입가 (₩)"),
        column_map.get("진입 MA 값"),
        column_map.get("청산가 (₩)"),
        column_map.get("손절 기준 (₩)"),
        column_map.get("익절 기준 (₩)"),
        column_map.get("잔액 (₩)"),
    ]
    currency_indices = [idx for idx in currency_indices if idx is not None]

    take_profit_fmt = workbook.add_format({"bg_color": "#E6F4EA"})
    stop_loss_fmt = workbook.add_format({"bg_color": "#FCE8E6"})
    position_long_fmt = workbook.add_format({"bold": True, "font_color": "#1B5E20"})
    position_short_fmt = workbook.add_format({"bold": True, "font_color": "#B71C1C"})
    currency_fmt = workbook.add_format({"num_format": "#,##0"})
    percent_fmt = workbook.add_format({"num_format": "0.00%"})
    rsi_fmt = workbook.add_format({"num_format": "0.00"})
    datetime_fmt = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm"})

    if exit_type_idx is not None:
        exit_col_letter = xl_col_to_name(exit_type_idx)
        first_excel_row = start_row + 1
        worksheet.conditional_format(
            start_row,
            0,
            end_row,
            end_col,
            {
                "type": "formula",
                "criteria": f'=${exit_col_letter}{first_excel_row}="익절"',
                "format": take_profit_fmt,
            },
        )
        worksheet.conditional_format(
            start_row,
            0,
            end_row,
            end_col,
            {
                "type": "formula",
                "criteria": f'=${exit_col_letter}{first_excel_row}="손절"',
                "format": stop_loss_fmt,
            },
        )

    if position_idx is not None:
        worksheet.conditional_format(
            start_row,
            position_idx,
            end_row,
            position_idx,
            {
                "type": "text",
                "criteria": "containing",
                "value": "롱",
                "format": position_long_fmt,
            },
        )
        worksheet.conditional_format(
            start_row,
            position_idx,
            end_row,
            position_idx,
            {
                "type": "text",
                "criteria": "containing",
                "value": "숏",
                "format": position_short_fmt,
            },
        )

    for idx in currency_indices:
        worksheet.set_column(idx, idx, 15, currency_fmt)

    if percent_idx is not None:
        worksheet.set_column(percent_idx, percent_idx, 12, percent_fmt)

    if rsi_idx is not None:
        worksheet.set_column(rsi_idx, rsi_idx, 10, rsi_fmt)

    for idx in datetime_indices:
        worksheet.set_column(idx, idx, 18, datetime_fmt)

    worksheet.set_column(0, 0, 6)
    worksheet.set_column(end_col, end_col, 32)

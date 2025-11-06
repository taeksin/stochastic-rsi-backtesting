from __future__ import annotations

from typing import Optional

import pandas as pd
from xlsxwriter.utility import xl_col_to_name


def style_trade_sheet(
    workbook,
    worksheet,
    dataframe: pd.DataFrame,
    header_row: int = 0,
) -> None:
    """
    Apply styling rules to the exported trades worksheet.

    - Shade entire rows green for take-profit exits and red for stop-loss exits.
    - Render the position column in bold with green text for long and red text for short.
    - Assumes the dataframe has already been written to the worksheet starting at row 0.
    """
    if dataframe is None or dataframe.empty:
        return

    # Determine row/column bounds for data region.
    start_row = header_row + 1  # first data row (Excel index starting at 0)
    end_row = start_row + len(dataframe) - 1
    end_col = len(dataframe.columns) - 1

    if end_row < start_row or end_col < 0:
        return

    # Locate key columns.
    try:
        exit_type_idx = dataframe.columns.get_loc("청산 유형")
    except KeyError:
        exit_type_idx = None

    try:
        position_idx = dataframe.columns.get_loc("포지션")
    except KeyError:
        position_idx = None

    # Formats.
    take_profit_fmt = workbook.add_format({"bg_color": "#E6F4EA"})
    stop_loss_fmt = workbook.add_format({"bg_color": "#FCE8E6"})
    position_long_fmt = workbook.add_format({"bold": True, "font_color": "#1B5E20"})
    position_short_fmt = workbook.add_format({"bold": True, "font_color": "#B71C1C"})

    # Apply row highlighting based on exit type.
    if exit_type_idx is not None:
        exit_col_letter = xl_col_to_name(exit_type_idx)
        first_data_excel_row = start_row + 1  # convert to 1-based Excel row number
        worksheet.conditional_format(
            start_row,
            0,
            end_row,
            end_col,
            {
                "type": "formula",
                "criteria": f'=${exit_col_letter}{first_data_excel_row}="익절"',
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
                "criteria": f'=${exit_col_letter}{first_data_excel_row}="손절"',
                "format": stop_loss_fmt,
            },
        )

    # Apply bold/colored text for position column.
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

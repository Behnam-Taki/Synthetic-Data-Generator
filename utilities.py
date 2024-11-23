from typing import Callable


def to_sub_description(description: str, format_for_one_line: bool = False) -> str:
    return ('\n' + description).replace('\n', '\n   ') \
        if '\n' in description or format_for_one_line \
        else ' ' + description


def bin_search_on_answer(
        func: Callable[[float], float],
        search_for: float,
        start: float,
        end: float,
        precision=1e-3
) -> float:
    func_is_acsending = func(start) < func(end)
    while end - start > precision:
        mid = (end + start) / 2
        if (func(mid) > search_for and func_is_acsending) or (func(mid) < search_for and not func_is_acsending):
            end = mid
        else:
            start = mid
    return start

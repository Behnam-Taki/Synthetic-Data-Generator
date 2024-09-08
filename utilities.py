def to_sub_description(description: str, format_for_one_line: bool = False) -> str:
    return ('\n' + description).replace('\n', '\n   ') \
        if '\n' in description or format_for_one_line \
        else ' ' + description

from lets_plot import *
LetsPlot.setup_html()

from . import transform_data as utd

def read_data(path, *, monospaced=None, family=None, family_col="font_family", dtype={"text": str}):
    import pandas as pd

    TARGET_FAMILIES = ["Arial", "Courier", "Geneva", "Georgia", "Helvetica", "Lucida Console", "Lucida Grande", "Times New Roman", "Verdana"]

    df = pd.read_csv(path, dtype=dtype)
    if monospaced is not None:
        if monospaced:
            df = utd.narrow_to_one_value(df, "is_monospaced", True)
        else:
            df = utd.narrow_to_one_value(df, "is_monospaced", False)
    if family is not None:
        if isinstance(family, list):
            df = df[df[family_col].isin(family)]
        elif family == "target":
            df = df[df[family_col].isin(TARGET_FAMILIES)]
        else:
            df = df[df[family_col] == family]

    return df

def plot_matrix(plots=[], width=400, height=300, columns=2):
    bunch = GGBunch()
    for i in range(len(plots)):
        row = int(i / columns)
        column = i % columns
        bunch.add_plot(plots[i], column * width, row * height, width, height)
    return bunch.show()

def to_kotlin_list(l, *, new_lines=True, replace_value_str=True, quotation_mark='"'):
    new_line_str = "\n" if new_lines else " "
    value_str = quotation_mark + "{0}" + quotation_mark if replace_value_str and isinstance(l[0], str) else "{0}"
    return "listOf(\n{0}\n)".format(("," + new_line_str).join([value_str.format(v) for v in l]))

def to_kotlin_map(d, *, new_lines=True, replace_key_str=True, replace_value_str=True, quotation_mark='"'):
    key_str = quotation_mark + "{0}" + quotation_mark if replace_key_str and isinstance(list(d.keys())[0], str) else "{0}"
    value_str = quotation_mark + "{1}" + quotation_mark if replace_value_str and isinstance(list(d.values())[0], str) else "{1}"
    new_line_str = "\n" if new_lines else " "
    return "mapOf(\n{0}\n)".format(("," + new_line_str).join([(key_str + " to " + value_str).format(k, v) for k, v in d.items()]))
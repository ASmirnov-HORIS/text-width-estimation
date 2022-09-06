import numpy as np
import pandas as pd

from . import font as ufont

def train_test_split_by_column(df, column, *, train_values=None, train_size=None, train_frac=.75, random_state=42, reset_index=True):
    values = train_values if train_values is not None \
                          else df[column].drop_duplicates().sample(n=train_size, frac=train_frac, random_state=random_state).values
    train_df, test_df = df[df[column].isin(values)], df[~df[column].isin(values)]
    if reset_index:
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df

def narrow_by_column(df, column, *, values=None, size=None, frac=.1, random_state=42, reset_index=True):
    return train_test_split_by_column(
        df, column,
        train_values=values, train_size=size, train_frac=frac,
        random_state=random_state, reset_index=reset_index
    )[0]

def narrow_to_one_value(df, column, value, *, reset_index=True):
    result_df = df[df[column] == value]
    result_df = result_df.drop(columns=[column])
    if reset_index:
        result_df.reset_index(drop=True, inplace=True)
    return result_df

def filter_by_font(
        df, font=ufont.BASIC_FONT, *,
        filters=["family", "size", "face"], reset_index=True,
        family_col="font_family", size_col="font_size", face_col="font_face"
    ):
    if "family" in filters:
        df = narrow_to_one_value(df, family_col, font.family, reset_index=reset_index)
    if "size" in filters:
        df = narrow_to_one_value(df, size_col, font.size, reset_index=reset_index)
    if "face" in filters:
        df = narrow_to_one_value(df, face_col, str(font.face), reset_index=reset_index)
    return df

def calc_char_widths(df, *, width_col="width", char_col="char", agg_fun=lambda r: r.mean()):
    s = df.groupby(char_col)[width_col].agg(agg_fun)
    s.name = "width"
    return s

def calc_char_orders(df, *, width_col="width", char_col="char", font_cols=[], agg_fun=lambda r: r.mean()):
    def as_tuple(t):
        if isinstance(t, str):
            return (t,)
        return t

    def slice_df(sliced_df, group_names, group_values):
        if len(group_names) == 0:
            return sliced_df
        return slice_df(sliced_df[sliced_df[group_names[0]] == group_values[0]], group_names[1:], group_values[1:])

    def get_row_df(group_df):
        group_df = group_df.groupby(char_col)[width_col].agg(agg_fun).reset_index()
        if group_df[width_col].min() == group_df[width_col].max():
            return group_df.assign(result=lambda x: np.nan).set_index(char_col).result
        group_df = group_df.sort_values(by=width_col)
        group_df["result"] = group_df.assign(u=1)\
                                     .groupby(width_col)["u"]\
                                     .transform(lambda x: (x.cumsum() / x.count()).astype(int) * x.count())\
                                     .cumsum()\
                                     .shift(periods=1, fill_value=0)
        return group_df.set_index(char_col).result

    if len(font_cols) == 0:
        return get_row_df(df).to_frame("order").T

    result_df = pd.concat([
        get_row_df(
            slice_df(df, font_cols, as_tuple(t))
        ).to_frame(name=as_tuple(t)).T
        for t in df.groupby(font_cols)[width_col].max().index
    ])
    result_index = pd.MultiIndex.from_tuples(result_df.index, names=font_cols) \
        if len(font_cols) > 1 else result_df.index
    result_df.set_index(result_index, inplace=True)

    return result_df

def calc_char_weights(texts_s, *, acceptable_chars=None):
    def occurrences_number(series, symbol):
        try:
            s = series.str.count(symbol).sum()
            return s - series.size if symbol in ['$', '^'] else s
        except:
            return 0

    chars = [char
        for char in texts_s.str.split("").apply(pd.Series).stack().drop_duplicates().sort_values().values
        if char != "" and (acceptable_chars is None or char in acceptable_chars)
    ]
    result_s = pd.Series({c: occurrences_number(texts_s, c) for c in chars}, name="weight")
    result_s.index.name = "char"

    return result_s
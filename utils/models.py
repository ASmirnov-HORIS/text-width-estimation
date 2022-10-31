import json

import numpy as np
import pandas as pd

from . import misc as um
from . import font as ufont
from . import transform_data as utd

def predict(text, predictor, *,
            name=None,
            round_result=True,
            allow_extra_symbols=True, extra_symbol_width=0,
            cluster_width_col="cluster_width"):
    import pandas as pd

    def predict_char_width(c):
        try:
            return predictor.loc[c][cluster_width_col]
        except KeyError as e:
            if allow_extra_symbols:
                return extra_symbol_width
            else:
                raise e

    def predict_line(line):
        result = sum([predict_char_width(c) for c in line])
        if round_result:
            return round(result)
        else:
            return result

    def predict_for_series(text_s, name=None):
        def split_string(s):
            return pd.Series([s[i:i+1] for i in range(len(s))])
        df = text_s.astype(str).apply(split_string)
        splitted_df = df.replace(predictor[cluster_width_col]).fillna(0)
        cols = splitted_df.columns
        result = splitted_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)\
                                  .fillna(extra_symbol_width).sum(axis=1)
        if round_result:
            result = result.round().astype(int)
        result.name = name or "predict_{0}".format(text_s.name)
        return result

    if isinstance(text, str):
        return predict_line(text)
    elif isinstance(text, pd.core.series.Series):
        return predict_for_series(text, name if (isinstance(name, str)) else None)
    elif isinstance(text, pd.core.frame.DataFrame):
        return pd.concat([
            predict_for_series(text[column], name[column] if (isinstance(name, dict)) else None)
            for column in text.columns
        ], axis="columns")
    else:
        raise Exception("Bad type of input: {0}".format(type(text)))

def prepare_char_data(char_widths_df, texts_s, *, width_col="width", char_col="char", font_cols=[], agg_fun=lambda r: r.mean()):
    char_widths_s = utd.calc_char_widths(char_widths_df, width_col=width_col, char_col=char_col, \
                                         agg_fun=agg_fun)
    char_weights_s = utd.calc_char_weights(texts_s, acceptable_chars=char_widths_s.index)
    char_orders_s = utd.calc_char_orders(char_widths_df, width_col=width_col, char_col=char_col, \
                                         font_cols=font_cols, agg_fun=agg_fun).loc["order"]
    df = pd.concat([char_widths_s, char_weights_s, char_orders_s], axis="columns")
    df.weight = (df.weight.fillna(0) + 1).astype(int)

    return df

class ClusteringModel:
    from sklearn.cluster import KMeans

    INDEX_NAME = "char"
    CLUSTERING_COL = "order"
    SIZE_COL = "width"
    WEIGHT_COL = "weight"
    CLUSTER_COL = "cluster"
    CLUSTER_SIZE_COL = "cluster_width"

    predictor = None
    extra_symbol_width = None

    def __init__(self,
                 predictor=None,
                 extra_symbol_width=None,
                 calc_cluster_size=None,
                 allow_extra_symbols=True,
                 index_name=INDEX_NAME,
                 clustering_col=CLUSTERING_COL,
                 size_col=SIZE_COL,
                 weight_col=WEIGHT_COL,
                 cluster_col=CLUSTER_COL,
                 cluster_size_col=CLUSTER_SIZE_COL,
                 **kmean_parameters):
        self.predictor = predictor
        self.extra_symbol_width = extra_symbol_width
        self.allow_extra_symbols = allow_extra_symbols
        self._calc_cluster_size = calc_cluster_size
        self._kmean_parameters = kmean_parameters
        self._index_name = index_name
        self._clustering_col = clustering_col
        self._size_col = size_col
        self._weight_col = weight_col
        self._cluster_col = cluster_col
        self._cluster_size_col = cluster_size_col

    def fit(self, char_data, *, admixture_col=None):
        if admixture_col is None:
            self.predictor = self._prepare_predictor(char_data, self._kmean_parameters)
        else:
            self.predictor = pd.concat([
                self._prepare_predictor(
                    char_data[char_data[admixture_col] == admixture_key],
                    {**self._kmean_parameters, **{"n_clusters": admixture_n_clusters}},
                    "{0}-".format(admixture_id)
                )
                for admixture_id, (admixture_key, admixture_n_clusters) \
                    in enumerate(self._calc_admixture_clusters(char_data[admixture_col].value_counts()).items())
            ])
        self.extra_symbol_width = (self._calc_cluster_size or self._def_calc_cluster_size)(self.predictor)
        return self

    def predict(self, text, *, name=None, round_result=True):
        return predict(
            text, self.predictor,
            name=name, round_result=round_result,
            allow_extra_symbols=self.allow_extra_symbols, extra_symbol_width=self.extra_symbol_width
        )

    def save(self, csv_path, json_path):
        self.predictor.to_csv(csv_path)
        json_data = {
            "extra_symbol_width": self.extra_symbol_width,
            "allow_extra_symbols": self.allow_extra_symbols,
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f)

    @classmethod
    def load(cls, csv_path, json_path):
        predictor = pd.read_csv(csv_path, index_col=0)
        json_data = {}
        with open(json_path, "r") as f:
            json_data = json.load(f)
        return cls(predictor=predictor,
                   extra_symbol_width=json_data["extra_symbol_width"],
                   allow_extra_symbols=json_data["allow_extra_symbols"])

    def _calc_admixture_clusters(self, admixture_counts):
        n_admixtures = admixture_counts.shape[0]
        n_clusters = max(n_admixtures, self._kmean_parameters.get("n_clusters", n_admixtures))
        current_n_clusters = n_clusters
        admixture_ratios = admixture_counts.sort_values(ascending=False) / admixture_counts.sum()
        result = {admixture: 0 for admixture in admixture_counts.keys()}
        while current_n_clusters > 0:
            for admixture, ratio in admixture_ratios.iteritems():
                if result[admixture] < max(1, round(n_clusters * ratio)):
                    result[admixture] += 1
                    current_n_clusters -= 1
                    if current_n_clusters == 0:
                        break
        return result

    def _prepare_predictor(self, char_data, kmean_parameters, admixture_prefix=""):
        predictor_cols = [self._size_col, self._weight_col] if self._clustering_col == self._size_col \
                                                            else [self._clustering_col, self._size_col, self._weight_col]
        predictor_df = char_data[predictor_cols].copy()
        predictor_df.index.name = self._index_name
        # Set clusters
        predictor_df[self._cluster_col] = self.KMeans(**kmean_parameters).fit(predictor_df[[self._clustering_col]]).labels_
        # Set cluster widths
        cluster_widths = predictor_df.groupby(self._cluster_col).apply(self._calc_cluster_size or self._def_calc_cluster_size)
        predictor_df[self._cluster_size_col] = predictor_df[self._cluster_col].replace(cluster_widths)
        # Sort clusters
        predictor_df.sort_values(by=self._cluster_size_col, inplace=True)
        predictor_df.cluster.replace(
            {cluster_id: "{0}{1}".format(admixture_prefix, i) \
             for i, cluster_id in enumerate(predictor_df[self._cluster_col].unique())},
            inplace=True
        )
        return predictor_df

    def _def_calc_cluster_size(self, r):
        return (r[self._size_col] * r[self._weight_col]).sum() / r[self._weight_col].sum()

    def __repr__(self):
        result = ""
        clustering_s = self.predictor.cluster
        clustering_d = clustering_s.to_frame().reset_index().groupby("cluster").char.agg(lambda r: list(r)).to_dict()
        result += "Clusters:\n\n{0}\n\n---\n\n".format(um.to_kotlin_map({
            k: um.to_kotlin_list(v, new_lines=False, quotation_mark='\'') \
            for k, v in clustering_d.items()
        }, replace_key_str=False, replace_value_str=False))
        cluster_widths_s = self.predictor[["cluster", "cluster_width"]].drop_duplicates()\
                               .set_index("cluster").cluster_width.sort_values()
        result += "Cluster widths:\n\n{0}\n\n---\n\n".format(um.to_kotlin_map(cluster_widths_s.to_dict(), replace_key_str=False))
        result += "Extra symbol width: {0}".format(self.extra_symbol_width)
        return result

class FullModel:
    def __init__(self, clustering_model, *,
                 family_coeff=None, face_coeff=None, size_coeff=None, exagg_coeff=1.0):
        self.clustering_model = clustering_model
        self.family_coeff = family_coeff
        self.face_coeff = face_coeff
        self.size_coeff = size_coeff
        self.exagg_coeff = exagg_coeff

    def fit(self, control_df, *,
            exagg_search_space=np.linspace(1.0, 1.5, 5001), exagg_target_score=.975,
            text_col="text", text_size_col="symbols_count",
            family_col="font_family", face_col="font_face", size_col="font_size"):
        # Prepare family coefficients
        family_df = utd.filter_by_font(control_df, filters=["size", "face"])
        family_df["predicted_width"] = self._basic_prediction(family_df[text_col])
        self.family_coeff = self._get_coefficients_s(family_df, "font_family").to_dict()
        # Prepare face coefficients
        face_df = utd.filter_by_font(control_df, filters=["size"])
        face_df["predicted_width"] = (
            self._basic_prediction(face_df[text_col]) + \
            face_df.symbols_count * face_df.font_family.replace(self.family_coeff)
        )
        self.face_coeff = self._get_coefficients_s(face_df, "font_face").to_dict()
        # Prepare size coefficient
        size_df = control_df.copy()
        size_df["predicted_width"] = (
            self._basic_prediction(size_df[text_col]) + \
            size_df.symbols_count * (
                size_df.font_family.replace(self.family_coeff) + \
                size_df.font_face.replace(self.face_coeff)
            )
        )
        sizes_coeff = self._get_coefficients_s(size_df, "font_size", additive=False)
        self.size_coeff = (sizes_coeff * ufont.BASIC_FONT_SIZE /  pd.Series(ufont.FONT_SIZES, index=ufont.FONT_SIZES)).mean()
        # Prepare exagg coefficient
        exagg_train_df = control_df.copy()
        exagg_train_df["predicted_width"] = self._font_based_prediction(exagg_train_df[text_col], \
                                                                        exagg_train_df[family_col], \
                                                                        exagg_train_df[face_col], \
                                                                        exagg_train_df[size_col], \
                                                                        symbols_count=exagg_train_df[text_size_col])
        self.exagg_coeff = self._get_exagg_coeff(exagg_train_df, exagg_search_space, exagg_target_score)
        return self

    def predict(self, text, *,
                use_exagg=True, name=None,
                text_col="text", text_size="symbols_count", family="font_family", face="font_face", size="font_size"):
        exagg_coeff = self.exagg_coeff if use_exagg else 1.0
        return exagg_coeff * self._font_based_prediction(text, family, face, size, \
                                                         text_col=text_col, symbols_count=text_size, name=name)

    def save(self, csv_path, json_path):
        self.clustering_model.save(csv_path, json_path)
        json_data = {
            "extra_symbol_width": self.clustering_model.extra_symbol_width,
            "allow_extra_symbols": self.clustering_model.allow_extra_symbols,
            "family_coeff": self.family_coeff,
            "face_coeff": self.face_coeff,
            "size_coeff": self.size_coeff,
            "exagg_coeff": self.exagg_coeff,
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f)

    @classmethod
    def load(cls, csv_path, json_path):
        clustering_model = ClusteringModel.load(csv_path, json_path)
        json_data = {}
        with open(json_path, "r") as f:
            json_data = json.load(f)
        return cls(clustering_model=clustering_model,
                   family_coeff=json_data["family_coeff"],
                   face_coeff=json_data["face_coeff"],
                   size_coeff=json_data["size_coeff"],
                   exagg_coeff=json_data["exagg_coeff"])

    def _basic_prediction(self, text, *, name=None):
        return self.clustering_model.predict(text, name=name, round_result=False)

    def _font_based_prediction(self, text, family, face, size, *,
                               text_col=None,
                               symbols_count=None, name=None):
        if isinstance(text, pd.core.frame.DataFrame):
            return (
                self._basic_prediction(text[text_col], name=name) + \
                text[symbols_count] * (text[family].replace(self.family_coeff) + text[face].replace(self.face_coeff))
            ) * self.size_coeff * text[size] / ufont.BASIC_FONT_SIZE
        elif isinstance(text, pd.core.frame.Series):
            return (
                self._basic_prediction(text, name=name) + \
                symbols_count * (family.replace(self.family_coeff) + face.replace(self.face_coeff))
            ) * self.size_coeff * size / ufont.BASIC_FONT_SIZE
        elif isinstance(text, str):
            return (
                self._basic_prediction(text, name=name) + \
                len(text) * (self.family_coeff[family] + self.face_coeff[face])
            ) * self.size_coeff * size / ufont.BASIC_FONT_SIZE
        else:
            raise Exception("Bad type of input: {0}".format(type(text)))

    def _get_coefficients_s(self, df, target_col, *, additive=True):
        if additive:
            df["coeff"] = (df["width"] - df["predicted_width"]) / df["symbols_count"]
            return df.groupby(target_col)["coeff"].mean()
        else:
            df = df.assign(
                numerator=df["width"] * df["predicted_width"] / df["symbols_count"].pow(2),
                denominator=df["predicted_width"].pow(2) / df["symbols_count"].pow(2)
            )
            grouped_df = df.groupby(target_col).agg({"numerator": ["sum"], "denominator": ["sum"]})
            return (grouped_df.numerator / grouped_df.denominator)["sum"].rename("coeff")

    def _get_exagg_coeff(self, df, search_space, target_score):
        def exagg_score(ec):
            s = ec * df["predicted_width"] / df["width"]
            w = np.where(s >= 1, True, False)
            return w[w].size / w.size

        best_exagg_coeff = search_space[0]
        best_score = 0
        for exagg_coeff in search_space:
            score = exagg_score(exagg_coeff)
            if score >= target_score:
                return exagg_coeff
            if score > best_score:
                best_score = score
                best_exagg_coeff = exagg_coeff
        return best_exagg_coeff

    def __repr__(self):
        result = str(self.clustering_model) + "\n\n---\n\n"
        result += "Family coefficients:\n\n{0}\n\n---\n\n".format(um.to_kotlin_map(self.family_coeff))
        result += "Face coefficients:\n\n{0}\n\n---\n\n".format(um.to_kotlin_map(self.face_coeff))
        result += "Size coefficient:\n\n{0}\n\n---\n\n".format(self.size_coeff)
        result += "Exagg coefficient:\n\n{0}".format(self.exagg_coeff)
        return result
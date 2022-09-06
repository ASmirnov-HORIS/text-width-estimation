import pandas as pd

from . import transform_data as utd

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
                 calc_cluster_size=None,
                 allow_extra_symbols=True,
                 index_name=INDEX_NAME,
                 clustering_col=CLUSTERING_COL,
                 size_col=SIZE_COL,
                 weight_col=WEIGHT_COL,
                 cluster_col=CLUSTER_COL,
                 cluster_size_col=CLUSTER_SIZE_COL,
                 **kmean_parameters):
        self.calc_cluster_size = calc_cluster_size
        self.allow_extra_symbols = allow_extra_symbols
        self.kmean_parameters = kmean_parameters
        self.index_name = index_name
        self.clustering_col = clustering_col
        self.size_col = size_col
        self.weight_col = weight_col
        self.cluster_col = cluster_col
        self.cluster_size_col = cluster_size_col

    def fit(self, char_data, *, admixture_col=None):
        if admixture_col is None:
            self.predictor = self._prepare_predictor(char_data, self.kmean_parameters)
        else:
            self.predictor = pd.concat([
                self._prepare_predictor(
                    char_data[char_data[admixture_col] == admixture_key],
                    {**self.kmean_parameters, **{"n_clusters": admixture_n_clusters}},
                    "{0}-".format(admixture_id)
                )
                for admixture_id, (admixture_key, admixture_n_clusters) \
                    in enumerate(self._calc_admixture_clusters(char_data[admixture_col].value_counts()).items())
            ])
        self.extra_symbol_width = (self.calc_cluster_size or self._calc_cluster_size)(self.predictor)
        return self

    def predict(self, text, name=None):
        return predict(
            text, self.predictor,
            name=name, allow_extra_symbols=self.allow_extra_symbols, extra_symbol_width=self.extra_symbol_width
        )

    def _calc_admixture_clusters(self, admixture_counts):
        n_admixtures = admixture_counts.shape[0]
        n_clusters = max(n_admixtures, self.kmean_parameters.get("n_clusters", n_admixtures))
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
        predictor_cols = [self.size_col, self.weight_col] if self.clustering_col == self.size_col \
                                                          else [self.clustering_col, self.size_col, self.weight_col]
        predictor_df = char_data[predictor_cols].copy()
        predictor_df.index.name = self.index_name
        # Set clusters
        predictor_df[self.cluster_col] = self.KMeans(**kmean_parameters).fit(predictor_df[[self.clustering_col]]).labels_
        # Set cluster widths
        cluster_widths = predictor_df.groupby(self.cluster_col).apply(self.calc_cluster_size or self._calc_cluster_size)
        predictor_df[self.cluster_size_col] = predictor_df[self.cluster_col].replace(cluster_widths)
        # Sort clusters
        predictor_df.sort_values(by=self.cluster_size_col, inplace=True)
        predictor_df.cluster.replace(
            {cluster_id: "{0}{1}".format(admixture_prefix, i) \
             for i, cluster_id in enumerate(predictor_df[self.cluster_col].unique())},
            inplace=True
        )
        return predictor_df

    def _calc_cluster_size(self, r):
        return (r[self.size_col] * r[self.weight_col]).sum() / r[self.weight_col].sum()

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
        df = text_s.apply(split_string)
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
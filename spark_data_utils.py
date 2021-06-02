from functools import reduce

from pyspark.sql import functions as F

from pyspark.sql.types import *


def show_df_partitioning_stats(df):
    rows_per_partition_df = df.select(F.spark_partition_id().alias("pid")).groupBy('pid').count()
    rows_per_partition_df.summary().show()


def count_nulls(spark, df, cols=None):
    total_count = df.count()
    _cols = df.columns if cols is None else cols
    res = []
    for col in _cols:
        col_nulls_count = df.where(F.col(col).isNull()).count()
        col_nulls_ratio = round(col_nulls_count / float(total_count), 3)
        entry = dict(col=col, nulls_count=col_nulls_count, nulls_ratio=col_nulls_ratio)
        res.append(entry)
    nulls_info_df = spark.createDataFrame(res)
    nulls_info_df.orderBy(F.desc('nulls_count')).show(len(res), False)
    return res


def lowercase_all_string_cols(df):
    schema_fields = df.schema.fields
    res_cols = []
    for field in schema_fields:
        col_name = field.name
        if field.dataType.typeName() == 'string':
            col_expr = F.lower(F.col(col_name)).alias(col_name)
        else:
            col_expr = col_name
        res_cols.append(col_expr)
    res_df = df.select(res_cols)
    return res_df


def add_cumsum_col(df, col_to_sum, orderby_col):
    from pyspark.sql import Window

    windowval = (Window.orderBy(orderby_col)
                 .rangeBetween(Window.unboundedPreceding, 0))
    df_w_cumsum = df.withColumn('{}_cum_sum'.format(col_to_sum), F.sum(col_to_sum).over(windowval))
    return df_w_cumsum


def add_prefix_to_cols_names(df, prefix=None, cols=None, exclude=()):
    cols = cols if cols else df.columns
    prefixed_cols_names = []
    for col in cols:
        if (col in exclude) or (col.startswith("{}_".format(prefix))):
            continue
        else:
            prefixed_col = "{}_{}".format(prefix, col)
            prefixed_cols_names.append(prefixed_col)
            df = df.withColumnRenamed(col, prefixed_col)
    return df, prefixed_cols_names


def plot_hist_from_df(df, col, bins=10):
    import pandas as pd
    get_ipython().magic(u'matplotlib inline')

    df_hist = df.select(col).rdd.flatMap(lambda x: x).histogram(bins)
    pd.DataFrame(list(zip(*df_hist)),
                 columns=['bin', 'frequency']).set_index('bin').plot(kind='bar')


def parse_domains(df, url_col='url'):
    import tldextract

    domain_features_schema = StructType([StructField("subdomain", StringType()),
                                         StructField("domain", StringType()),
                                         StructField("suffix", StringType()),
                                         StructField("registered_domain", StringType())])

    def parse_domain(d):
        res = {}
        ex_res = tldextract.extract(d)
        res['subdomain'] = ex_res.subdomain if ex_res.subdomain != '' else None
        res['domain'] = ex_res.domain if ex_res.domain != '' else None
        res['suffix'] = ex_res.suffix if ex_res.suffix != '' else None
        res['registered_domain'] = ex_res.registered_domain

        return res

    parse_udf = F.udf(parse_domain, domain_features_schema)

    res_df = df.select('*', F.explode(F.array(parse_udf(url_col))).alias('temp_col')).select('*', 'temp_col.*').drop(
        'temp_col')

    return res_df


def create_switch_case_expr(case_list, default_value):
    """
    Create switch case expression using nested 'when' Spark SQL expressions
    :param case_list: list of boolean Column expression and result tuples. e.g:
            [(F.col('label') == 'white', 0.0),
            (F.col('label') == 'black', 1.0)]
    :type case_list: list[(pyspark.sql.Column, any)]
    :param default_value: return value for unmatched condition
    :type default_value: any
    :return: Spark SQL nested 'when' expression
    :rtype: pyspark.sql.Column
    """
    expr = reduce(
        lambda acc, kv: F.when(kv[0], kv[1]).otherwise(acc),
        case_list,
        default_value
    )
    return expr


# case_list = [(F.col('label') == 'white', 0.0),
#              (F.col('label') == 'black', 1.0)]
# create_switch_case_expr(case_list, 2.0)


def stringfy_spark_array(col, quote=False):
    """

    :param col:
    :type col: pyspark.sql.column.Column
    :param quote:
    :type quote: bool
    :return:
    :rtype:
    """
    if quote:
        list_open = "\"["
        list_close = "]\""
    else:
        list_open = "["
        list_close = "]"
    expr = F.concat(F.lit(list_open), F.concat_ws(', ', col), F.lit(list_close))
    return expr


def get_col_top_n_vals(df, col, n):
    top_values = df.groupBy(col).count().orderBy(F.desc('count')).rdd.map(lambda r: r[col]).take(n)
    return top_values


def drop_top_n_values(spark, df, col, n):
    top_n_vals = get_col_top_n_vals(df, col, n)
    top_n_vals_bc_var = spark.sparkContext.broadcast(top_n_vals)
    res_df = df.filter(~F.col(col).isin(top_n_vals_bc_var.value))
    top_n_vals_bc_var.unpersist()
    return res_df


def drop_consecutive_dups(df, col, partition_by_vals, order_by_vals):
    from pyspark.sql import Window
    win = Window.partitionBy(*partition_by_vals).orderBy(*order_by_vals)
    res_df = df.withColumn('eq_to_pre', F.lag(col).over(win) == F.col(col)) \
        .orderBy('german_datetime')
    res_df = res_df.filter(~F.col('eq_to_pre') | F.col('eq_to_pre').isNull())
    res_df = res_df.drop('eq_to_pre')
    return res_df


def sample_by_col(spark, df, col, n):
    import random
    col_vals = df.select(col).distinct().rdd.map(lambda r: r[0]).collect()
    col_vals_subset = random.sample(col_vals, n)
    col_vals_subset_bc_var = spark.sparkContext.broadcast(col_vals_subset)
    res_df = df.filter(F.col(col).isin(col_vals_subset_bc_var.value))
    col_vals_subset_bc_var.unpersist()
    return res_df


def get_df_features_types(df, ftrs_cols=None):
    ftrs_cols = ftrs_cols if ftrs_cols else df.columns
    features = {}

    features['categorical'] = [f[0] for f in df.dtypes
                               if ((f[0] in ftrs_cols) and (f[1] == 'string'))]
    features['boolean'] = [f[0] for f in df.dtypes
                           if ((f[0] in ftrs_cols) and (f[1] == 'boolean'))]
    features['numerical'] = list(set(ftrs_cols) - set(features['categorical'] + features['boolean']))
    return features


def get_cols_minmax_dict(df, cols=None):
    cols = cols if cols else df.columns
    df = df.select(*cols)
    minmax_summary_dicts = df.summary('min', 'max').rdd.map(lambda r: r.asDict()).collect()

    summary_dict = {}
    for d in minmax_summary_dicts:
        summary_dict[d.pop('summary')] = d
        for k, v in d.items():
            d[k] = float(v)

    return summary_dict


def add_minmax_scale_cols(df, cols):
    minmax_dict = get_cols_minmax_dict(df, cols)

    for col in cols:
        minmax_scale_expr = ((F.col(col) - minmax_dict['min'][col]) /
                           (minmax_dict['max'][col] - minmax_dict['min'][col]))
        df = df.withColumn('{}_scaled'.format(col),
                           F.round(minmax_scale_expr, 5))
    return df


# UDFs


@F.udf(returnType=DoubleType())
def vec_ith(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None


@F.udf(returnType=ArrayType(DoubleType()))
def vec_to_array(v):
    return v.toArray().tolist()


@F.udf(returnType=StringType())
def string_jaccard(s1, s2, n=1):
    s1_grams = set([s1[i:i + n] for i in range(len(s1) + 1 - n)])
    s2_grams = set([s2[i:i + n] for i in range(len(s2) + 1 - n)])
    intersection_size = len(s1_grams & s2_grams)
    union_size = len(s1_grams | s2_grams)
    jaccard_idx = float(intersection_size) / union_size
    return jaccard_idx

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(gini_test, gini_train, mo):
    mo.md(rf"""
    ## Коефіцієнти Gini

    Тренувальна вибірка: {gini_train:.4f}

    Тестова вибірка: {gini_test:.4f}
    """)
    return


@app.cell(hide_code=True)
def _(feature_importance, plt, sns):
    sns.set_style("whitegrid")

    plt.rcParams["figure.dpi"] = 160
    plt.figure(figsize=(10, 5))

    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)

    plt.title('Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data playground
    """)
    return


@app.cell(hide_code=True)
def _(app_activity, mo):
    app_activity
    table_selector = mo.ui.dropdown(["app_activity", "communications", "transactions"], value="app_activity", label="Таблиця")
    target_switch = mo.ui.checkbox(label="Відкрив депозит")
    return table_selector, target_switch


@app.cell(hide_code=True)
def _(mo, table_selector, target_switch):
    mo.md(rf"""
    {table_selector}

    {target_switch}
    """)
    return


@app.cell
def _(mo, table_selector, target_switch):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            {table_selector.value}
            JOIN clients_sample ON {table_selector.value}.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            clients_sample.IS_TRAIN = 'true'
            AND clients_sample.TARGET = '{target_switch.value}'
        LIMIT
            1000000;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Features
    """)
    return


@app.cell
def _(app_activity, clients_sample, mo):
    count_app_activity_per_user_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            app_activity
            JOIN clients_sample ON app_activity.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return (count_app_activity_per_user_df,)


@app.cell
def _(clients_sample, mo, transactions):
    count_transactions_per_user_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            transactions
            JOIN clients_sample ON transactions.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return (count_transactions_per_user_df,)


@app.cell
def _(clients_sample, communications, mo):
    count_communications_per_user_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            communications
            JOIN clients_sample ON communications.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return (count_communications_per_user_df,)


@app.cell
def _(clients_sample, mo, transactions):
    avg_transactions_FLOAT_C18_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            AVG(transactions.FLOAT_C18) AS 'avg',
        FROM
            transactions
            JOIN clients_sample ON transactions.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return


@app.cell
def _(clients_sample, mo, transactions):
    count_transactions_INT_C19_eq_minus_1_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            transactions
            JOIN clients_sample ON transactions.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            transactions.INT_C19 = -1
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return


@app.cell
def _(clients_sample, mo, transactions):
    count_transactions_INT_C19_eq_plus_1_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            transactions
            JOIN clients_sample ON transactions.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            transactions.INT_C19 = 1
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return


@app.cell
def _(clients_sample, communications, mo):
    count_communications_CAT_C4_eq_3_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            communications
            JOIN clients_sample ON communications.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            communications.CAT_C4 = '3'
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return


@app.cell
def _(app_activity, clients_sample, mo):
    count_app_activity_CAT_C6_eq_1_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            app_activity
            JOIN clients_sample ON app_activity.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            app_activity.CAT_C6 = '1'
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return


@app.cell
def _(app_activity, clients_sample, mo):
    count_app_activity_CAT_C9_eq_1_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            COUNT(clients_sample.CLIENT_ID) AS 'count',
        FROM
            app_activity
            JOIN clients_sample ON app_activity.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            app_activity.CAT_C9 = '1'
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return


@app.cell
def _(app_activity, clients_sample, mo):
    num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            MIN(app_activity.ACTIVITY_DATE) AS 'min'
        FROM
            app_activity
            JOIN clients_sample ON app_activity.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """,
        output=False
    )
    return (num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df,)


@app.cell
def _(num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df, pd):
    target_date = pd.to_datetime('2025-09-01')
    num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df['time_diff'] = target_date - num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df['min']
    num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df['int'] = num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df['time_diff'].dt.days
    return


@app.cell
def _(
    avg_transactions_float_c18_df,
    clients_sample,
    count_app_activity_cat_c6_eq_1_df,
    count_app_activity_cat_c9_eq_1_df,
    count_app_activity_per_user_df,
    count_communications_cat_c4_eq_3_df,
    count_communications_per_user_df,
    count_transactions_int_c19_eq_minus_1_df,
    count_transactions_int_c19_eq_plus_1_df,
    count_transactions_per_user_df,
    mo,
    num_of_days_from_app_activity_min_activity_date_to_2025_09_01_df,
):
    features_df = mo.sql(
        f"""
        SELECT
            COALESCE(
                ANY_VALUE(count_app_activity_per_user_df.count),
                0
            ) as 'count_app_activity_per_user',
            COALESCE(
                ANY_VALUE(count_transactions_per_user_df.count),
                0
            ) as 'count_transactions_per_user',
            COALESCE(
                ANY_VALUE(count_communications_per_user_df.count),
                0
            ) as 'count_communications_per_user',
            COALESCE(ANY_VALUE(avg_transactions_FLOAT_C18_df.avg), 0) as 'avg_transactions_FLOAT_C18',
            COALESCE(
                ANY_VALUE(count_transactions_INT_C19_eq_minus_1_df.count),
                0
            ) / (
                COALESCE(
                    ANY_VALUE(count_transactions_INT_C19_eq_minus_1_df.count),
                    1
                ) + COALESCE(
                    ANY_VALUE(count_transactions_INT_C19_eq_plus_1_df.count),
                    0
                )
            ) as 'percent_count_transactions_INT_C19_eq_minus_1',
            COALESCE(
                ANY_VALUE(count_communications_CAT_C4_eq_3_df.count),
                0
            ) / COALESCE(
                ANY_VALUE(count_communications_per_user_df.count),
                1
            ) as 'percent_communications_CAT_C4_eq_3',
            COALESCE(
                ANY_VALUE(count_app_activity_CAT_C6_eq_1_df.count),
                0
            ) / COALESCE(
                ANY_VALUE(count_app_activity_per_user_df.count),
                1
            ) as 'percent_app_activity_CAT_C6_eq_1',
            COALESCE(
                ANY_VALUE(count_app_activity_CAT_C9_eq_1_df.count),
                0
            ) / COALESCE(
                ANY_VALUE(count_app_activity_per_user_df.count),
                1
            ) as 'percent_app_activity_CAT_C9_eq_1',
            COALESCE(
                ANY_VALUE(
                    num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df.int
                ),
                180
            ) as 'num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01',
            ANY_VALUE(clients_sample.TARGET) AS 'TARGET',
            ANY_VALUE(clients_sample.IS_TRAIN) AS 'IS_TRAIN'
        FROM
            clients_sample
            LEFT JOIN count_app_activity_per_user_df ON count_app_activity_per_user_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_transactions_per_user_df ON count_transactions_per_user_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_communications_per_user_df ON count_communications_per_user_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN avg_transactions_FLOAT_C18_df ON avg_transactions_FLOAT_C18_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_transactions_INT_C19_eq_minus_1_df ON count_transactions_INT_C19_eq_minus_1_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_transactions_INT_C19_eq_plus_1_df ON count_transactions_INT_C19_eq_plus_1_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_communications_CAT_C4_eq_3_df ON count_communications_CAT_C4_eq_3_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_app_activity_CAT_C6_eq_1_df ON count_app_activity_CAT_C6_eq_1_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN count_app_activity_CAT_C9_eq_1_df ON count_app_activity_CAT_C9_eq_1_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df ON num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01_df.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """
    )
    return (features_df,)


@app.cell
def _():
    features = ['count_app_activity_per_user', 'count_transactions_per_user', 'count_communications_per_user', 'avg_transactions_FLOAT_C18', 'percent_count_transactions_INT_C19_eq_minus_1', 'percent_communications_CAT_C4_eq_3', 'percent_app_activity_CAT_C6_eq_1', 'percent_app_activity_CAT_C9_eq_1', 'num_of_days_from_app_activity_min_ACTIVITY_DATE_to_2025_09_01']
    return (features,)


@app.cell
def _(features_df, mo):
    training_df = mo.sql(
        f"""
        SELECT
            *
        FROM
            features_df
        WHERE
            IS_TRAIN = 'true';
        """
    )
    return (training_df,)


@app.cell
def _(features_df, mo):
    test_df = mo.sql(
        f"""
        SELECT
            *
        FROM
            features_df
        WHERE
            IS_TRAIN = 'false';
        """
    )
    return (test_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Модель
    """)
    return


@app.function(hide_code=True)
def gini(roc_auc):
    return 2 * roc_auc - 1


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    StandardScaler,
    features,
    roc_auc_score,
    test_df,
    training_df,
):
    features_train = training_df[features]
    target_train = training_df['TARGET'].astype(int)

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    model = LogisticRegression(random_state=32, class_weight='balanced')
    model.fit(features_train_scaled, target_train)

    train_proba = model.predict_proba(features_train_scaled)[:, 1]

    features_test = test_df[features]
    target_test = test_df['TARGET'].astype(int)
    features_test_scaled = scaler.transform(features_test)
    target_pred = model.predict(features_test_scaled)
    target_proba = model.predict_proba(features_test_scaled)[:, 1]

    roc_auc_train = roc_auc_score(target_train, train_proba)
    gini_train = gini(roc_auc_train)

    roc_auc_test = roc_auc_score(target_test, target_proba)
    gini_test = gini(roc_auc_test)
    return gini_test, gini_train, model, target_pred, target_proba


@app.cell(hide_code=True)
def _(target_pred, target_proba, test_df):
    result_df = test_df
    result_df['pred_prob'] = target_proba
    result_df['pred_bool'] = target_pred.astype(bool)
    result_df['match'] = result_df['TARGET'] == result_df['pred_bool']
    result_df
    return


@app.cell
def _(clients_sample, mo, transactions):
    ts_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            transactions.TRAN_DATE,
            transactions.FLOAT_C18,
            transactions.FLOAT_C21
        FROM
            transactions
            JOIN clients_sample ON transactions.CLIENT_ID = clients_sample.CLIENT_ID
        WHERE
            clients_sample.IS_TRAIN = 'true'
        LIMIT
            200000;
        """,
        output=False
    )
    return (ts_df,)


@app.cell
def _(extract_features, ts_df):
    my_extracted_features = extract_features(ts_df, column_id="CLIENT_ID", column_sort="TRAN_DATE")
    return (my_extracted_features,)


@app.cell
def _(my_extracted_features):
    indexed_my_extracted_features = my_extracted_features.reset_index()
    indexed_my_extracted_features
    return (indexed_my_extracted_features,)


@app.cell
def _(clients_sample, indexed_my_extracted_features, mo):
    fefe_df = mo.sql(
        f"""
        SELECT
            *
        FROM
            indexed_my_extracted_features
            JOIN clients_sample ON indexed_my_extracted_features.index = clients_sample.CLIENT_ID;
        """
    )
    return (fefe_df,)


@app.cell
def _(fefe_df):
    fefeatures = list(fefe_df.columns)
    fefeatures.remove("index")
    fefeatures.remove("COMMUNICATION_MONTH")
    fefeatures.remove("IS_TRAIN")
    fefeatures.remove("CLIENT_ID")
    fefeatures.remove("FLOAT_C18__sample_entropy")
    fefeatures.remove("FLOAT_C21__sample_entropy")
    fefeatures.remove("FLOAT_C18__query_similarity_count__query_None__threshold_0.0")
    fefeatures.remove("FLOAT_C21__query_similarity_count__query_None__threshold_0.0")
    return (fefeatures,)


@app.cell
def _(fefe_df_filled):
    import numpy as np
    numeric_df = fefe_df_filled.select_dtypes(include=np.number)
    is_inf_df = numeric_df.apply(np.isinf)
    columns_with_inf = is_inf_df.any(axis=0)
    columns_with_inf
    return np, numeric_df


@app.cell
def _(np, numeric_df):
    is_nan_df = numeric_df.apply(np.isnan)
    columns_with_nan = is_nan_df.any(axis=0)
    columns_with_nan
    return


@app.cell
def _(fefe_df, fefeatures):
    fefeatures_df = fefe_df[fefeatures]
    column_medians = fefeatures_df.median()

    fefe_df_filled = fefeatures_df.fillna(column_medians)
    return (fefe_df_filled,)


@app.cell
def _(fefe_df, fefe_df_filled, train_test_split):
    fetargets = fefe_df['TARGET'].astype(int)
    train_fefeatures, test_fefeatures, train_fetarget, test_fetarget = train_test_split(fefe_df_filled, fetargets, test_size=0.3, random_state=32)
    return test_fefeatures, test_fetarget, train_fefeatures, train_fetarget


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    roc_auc_score,
    test_fefeatures,
    test_fetarget,
    train_fefeatures,
    train_fetarget,
):
    fescaler = StandardScaler()
    fefeatures_train_scaled = fescaler.fit_transform(train_fefeatures)

    femodel = LogisticRegression(random_state=32, class_weight='balanced')
    femodel.fit(fefeatures_train_scaled, train_fetarget)

    train_feproba = femodel.predict_proba(fefeatures_train_scaled)[:, 1]

    fefeatures_test_scaled = fescaler.transform(test_fefeatures)
    target_feproba = femodel.predict_proba(fefeatures_test_scaled)[:, 1]

    roc_auc_fetrain = roc_auc_score(train_fetarget, train_feproba)
    gini_fetrain = gini(roc_auc_fetrain)

    roc_auc_fetest = roc_auc_score(test_fetarget, target_feproba)
    gini_fetest = gini(roc_auc_fetest)
    return gini_fetest, gini_fetrain


@app.cell
def _(gini_fetest, gini_fetrain, mo):
    mo.md(rf"""
    Тренувальна вибірка: {gini_fetrain:.4f}

    Тестова вибірка: {gini_fetest:.4f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature importance
    """)
    return


@app.cell(hide_code=True)
def _(features, model, pd):
    coefficients = model.coef_[0]

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    })

    feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()

    feature_importance = feature_importance.sort_values(
        by='Absolute Coefficient', 
        ascending=False
    )
    return (feature_importance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Завантаження даних
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    # read client sample
    client_sample_dtypes = {
       'CLIENT_ID': 'uint64',
       'TARGET': 'boolean',
       'IS_TRAIN': 'boolean',
    }
    clients_sample = pd.read_csv(
       r"/CLIENTS_SAMPLE.csv",
       sep=',',
       dtype=client_sample_dtypes
    )
    return (clients_sample,)


@app.cell(hide_code=True)
def _(pd):
    # read app activity
    app_activity_dtypes = {
       'CLIENT_ID': 'uint64',
       'DEVICE_ID': 'uint64',
       'CAT_C3': 'category',
       'CAT_C4': 'category',
       'CAT_C5': 'category',
       'CAT_C6': 'category',
       'CAT_C8': 'category',
       'CAT_C9': 'category',
       'CAT_C10': 'category',
       'FLOAT_C11': 'float32',
       'FLOAT_C12': 'float32',
       'FLOAT_C13': 'float32',
       'FLOAT_C14': 'float32',
       'FLOAT_C15': 'float32',
       'FLOAT_C16': 'float32',
       'FLOAT_C17': 'float32'
    }

    app_activity = pd.read_csv(
       r"/APP_ACTIVITY.csv",
       sep=',',
       dtype=app_activity_dtypes,
       parse_dates=["ACTIVITY_DATE"],
    )
    return (app_activity,)


@app.cell(hide_code=True)
def _(pd):
    # read communications
    communications_dtypes = {
       'CLIENT_ID': 'uint64',
       "CAT_C2": "category",
       "CAT_C3": "category",
       "CAT_C4": "category",
       "CAT_C5": "category",

    }

    communications = pd.read_csv(
       r"/COMMUNICATIONS.csv",
       sep=',',
       dtype=communications_dtypes,
       parse_dates=["CONTACT_DATE"],
    )
    return (communications,)


@app.cell(hide_code=True)
def _(pd):
    # read transactions
    transactions_dtypes = {
       'CLIENT_ID': 'uint64',
       'CAT_C2': 'category',
       'CAT_C3': 'category',
       'CAT_C4': 'category',
       'FL_C6': 'bool',
       'FL_C7': 'bool',
       'FL_C8': 'bool',
       'FL_C9': 'bool',
       'FL_C10': 'bool',
       'FL_C11': 'bool',
       'FL_C12': 'bool',
       'FL_C13': 'bool',
       'FL_C14': 'bool',
       'FL_C15': 'bool',
       'FLOAT_C16': 'float32',
       'FLOAT_C17': 'float32',
       'FLOAT_C18': 'float32',
       'INT_C19': 'int32',
       'FLOAT_C20': 'float32',
       'FLOAT_C21': 'float32'
    }

    transactions = pd.read_csv(
       r"/TRANSACTIONS.csv",
       sep=',',
       dtype=transactions_dtypes,
       parse_dates=["TRAN_DATE"],
    )
    return (transactions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Бібліотеки
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from tsfresh import extract_features
    return (
        LogisticRegression,
        StandardScaler,
        extract_features,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


if __name__ == "__main__":
    app.run()

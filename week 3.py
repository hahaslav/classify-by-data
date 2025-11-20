import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature importance
    """)
    return


@app.cell(hide_code=True)
def _(feature_importance, features_list, plt, sns):
    sns.set_style("whitegrid")

    plt.rcParams["figure.dpi"] = 160
    plt.figure(figsize=(10, len(features_list) / 4.5))

    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)

    plt.title('Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo, train_gini, validate_gini):
    mo.md(rf"""
    ## Коефіцієнти Gini

    Тренувальна вибірка: {train_gini:.4f}

    Валідаційна вибірка: {validate_gini:.4f}

    {mo.accordion(
        {
            "Тестова вибірка: ": mo.md(f"Поточна: {0:.4f}\n\nМинула: 0.4921")}
    )}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Модель
    """)
    return


@app.function(hide_code=True)
def gini(roc_auc):
    return 2 * roc_auc - 1


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    features_list,
    roc_auc_score,
    train_df,
    validate_df,
):
    train_features = train_df[features_list]
    train_target = train_df['TARGET'].astype(int)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    model = LogisticRegression(random_state=32, max_iter=250, class_weight='balanced')
    model.fit(train_features_scaled, train_target)

    train_proba = model.predict_proba(train_features_scaled)[:, 1]

    validate_features = validate_df[features_list]
    validate_target = validate_df['TARGET'].astype(int)

    validate_features_scaled = scaler.transform(validate_features)
    validate_proba = model.predict_proba(validate_features_scaled)[:, 1]

    train_gini = gini(roc_auc_score(train_target, train_proba))
    validate_gini = gini(roc_auc_score(validate_target, validate_proba))
    return model, train_gini, validate_gini


@app.cell(hide_code=True)
def _(features_list, model, pd):
    coefficients = model.coef_[0]

    feature_importance = pd.DataFrame({
        'Feature': features_list,
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
    ## Підготовка згенерованих датасетів до тренування
    """)
    return


@app.cell
def _(
    train_app_activity_extracted_features,
    train_communications_extracted_features,
    train_transactions_extracted_features,
):
    features_list = list(train_transactions_extracted_features.columns) + list(train_app_activity_extracted_features.columns) + list(train_communications_extracted_features.columns) + ["count_app_activity_per_user", "count_communications_per_user", "count_transactions_per_user", "days_old_ACTIVITY_DATE"]
    train_transactions_extracted_features_medians = train_transactions_extracted_features.median()
    train_app_activity_extracted_features_medians = train_app_activity_extracted_features.median()
    train_communications_extracted_features_medians = train_communications_extracted_features.median()
    return (
        features_list,
        train_app_activity_extracted_features_medians,
        train_communications_extracted_features_medians,
        train_transactions_extracted_features_medians,
    )


@app.cell
def _(
    clients_df,
    duckdb,
    indexed_app_activity_extracted_features,
    indexed_communications_extracted_features,
    indexed_transactions_extracted_features,
    my_features,
    train_app_activity_extracted_features_medians,
    train_communications_extracted_features_medians,
    train_transactions_extracted_features_medians,
):
    def prepare_dataset(clients_df, transactions_extracted_features, app_activity_extracted_features, communications_extracted_features, my_features):
        indexed_transactions_extracted_features = transactions_extracted_features.reset_index()
        indexed_app_activity_extracted_features = app_activity_extracted_features.reset_index()
        indexed_communications_extracted_features = communications_extracted_features.reset_index()
        df_with_all_clients = duckdb.sql(
            f"""
            SELECT
                *
            FROM
                clients_df
                LEFT JOIN indexed_transactions_extracted_features ON indexed_transactions_extracted_features.index = clients_df.CLIENT_ID
                LEFT JOIN indexed_app_activity_extracted_features ON indexed_app_activity_extracted_features.index = clients_df.CLIENT_ID
                LEFT JOIN indexed_communications_extracted_features ON indexed_communications_extracted_features.index = clients_df.CLIENT_ID
                LEFT JOIN my_features ON my_features.CLIENT_ID = clients_df.CLIENT_ID
            """
        ).df()
        return df_with_all_clients.fillna(train_transactions_extracted_features_medians).fillna(train_app_activity_extracted_features_medians).fillna(train_communications_extracted_features_medians)
    return (prepare_dataset,)


@app.cell
def _(
    prepare_dataset,
    train_app_activity_extracted_features,
    train_clients,
    train_communications_extracted_features,
    train_my_features,
    train_transactions_extracted_features,
    validate_app_activity_extracted_features,
    validate_clients,
    validate_communications_extracted_features,
    validate_my_features,
    validate_transactions_extracted_features,
):
    train_df = prepare_dataset(train_clients, train_transactions_extracted_features, train_app_activity_extracted_features, train_communications_extracted_features, train_my_features)
    validate_df = prepare_dataset(validate_clients, validate_transactions_extracted_features, validate_app_activity_extracted_features, validate_communications_extracted_features, validate_my_features)
    return train_df, validate_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Власні фічі
    """)
    return


@app.cell
def _(clients_df, df, duckdb):
    def count_per_user(df, clients_df):
        return duckdb.sql(
            f"""
            SELECT
                clients_df.CLIENT_ID,
                COUNT(clients_df.CLIENT_ID) AS 'count',
            FROM
                df
                JOIN clients_df ON df.CLIENT_ID = clients_df.CLIENT_ID
            GROUP BY
                clients_df.CLIENT_ID;
            """).df()
    return (count_per_user,)


@app.cell
def _(
    app_activity_df,
    clients_df,
    count_app_activity_per_user,
    count_communications_per_user,
    count_per_user,
    count_transactions_per_user,
    days_old_activity_date,
    df_with_all_clients_days_fillna,
    duckdb,
    pd,
):
    def prepair_my_features(app_activity_df, communications_df, transactions_df, clients_df):
        count_app_activity_per_user = count_per_user(app_activity_df, clients_df)
        count_communications_per_user = count_per_user(communications_df, clients_df)
        count_transactions_per_user = count_per_user(transactions_df, clients_df)
        days_old_ACTIVITY_DATE = duckdb.sql(
            """
            SELECT
                clients_df.CLIENT_ID,
                MIN(app_activity_df.ACTIVITY_DATE) AS 'min'
            FROM
                app_activity_df
                JOIN clients_df ON app_activity_df.CLIENT_ID = clients_df.CLIENT_ID
            GROUP BY
                clients_df.CLIENT_ID;
            """
        ).df()
        target_date = pd.to_datetime('2025-09-01')
        days_old_ACTIVITY_DATE['time_diff'] = target_date - days_old_ACTIVITY_DATE['min']
        days_old_ACTIVITY_DATE['int'] = days_old_ACTIVITY_DATE['time_diff'].dt.days
        df_with_all_clients_days = duckdb.sql(
            """
            SELECT
                clients_df.CLIENT_ID,
                days_old_ACTIVITY_DATE.int as 'days_old_ACTIVITY_DATE',
            FROM
                clients_df
                LEFT JOIN days_old_ACTIVITY_DATE ON days_old_ACTIVITY_DATE.CLIENT_ID = clients_df.CLIENT_ID
            """
        ).df()
        df_with_all_clients_days_fillna = df_with_all_clients_days.fillna(180)
        df_with_all_clients = duckdb.sql(
            """
            SELECT
                df_with_all_clients_days_fillna.CLIENT_ID,
                df_with_all_clients_days_fillna.days_old_ACTIVITY_DATE,
                count_app_activity_per_user.count as 'count_app_activity_per_user',
                count_communications_per_user.count as 'count_communications_per_user',
                count_transactions_per_user.count as 'count_transactions_per_user'
            FROM
                df_with_all_clients_days_fillna
                LEFT JOIN count_app_activity_per_user ON count_app_activity_per_user.CLIENT_ID = df_with_all_clients_days_fillna.CLIENT_ID
                LEFT JOIN count_communications_per_user ON count_communications_per_user.CLIENT_ID = df_with_all_clients_days_fillna.CLIENT_ID
                LEFT JOIN count_transactions_per_user ON count_transactions_per_user.CLIENT_ID = df_with_all_clients_days_fillna.CLIENT_ID
            """
        ).df()
        return df_with_all_clients.fillna(0)
    return (prepair_my_features,)


@app.cell
def _(
    prepair_my_features,
    train_app_activity,
    train_clients,
    train_communications,
    train_transactions,
    validate_app_activity,
    validate_clients,
    validate_communications,
    validate_transactions,
):
    train_my_features = prepair_my_features(train_app_activity, train_communications, train_transactions, train_clients)
    validate_my_features = prepair_my_features(validate_app_activity, validate_communications, validate_transactions, validate_clients)
    return train_my_features, validate_my_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Генерація фіч
    """)
    return


@app.cell
def _(df, duckdb):
    def numerify_transactions(df):
        return duckdb.sql(
            f"""
            SELECT
                df.CLIENT_ID,
                df.TRAN_DATE,
                df.FLOAT_C16 as transactions_FLOAT_C16,
                df.FLOAT_C17 as transactions_FLOAT_C17,
                df.FLOAT_C18 as transactions_FLOAT_C18,
                df.INT_C19 as transactions_INT_C19,
                df.FLOAT_C20 as transactions_FLOAT_C20,
                df.FLOAT_C21 as transactions_FLOAT_C21
            FROM
                df
            """
        ).df()
    return (numerify_transactions,)


@app.cell
def _(numerify_transactions, train_transactions):
    train_transactions_numerified = numerify_transactions(train_transactions)
    return (train_transactions_numerified,)


@app.cell
def _(numerify_transactions, validate_transactions):
    validate_transactions_numerified = numerify_transactions(validate_transactions)
    return (validate_transactions_numerified,)


@app.cell
def _(MinimalFCParameters, extract_features):
    def extract_transactions_features(df):
        return extract_features(df, column_id="CLIENT_ID", column_sort="TRAN_DATE", default_fc_parameters=MinimalFCParameters())
    return (extract_transactions_features,)


@app.cell
def _(extract_transactions_features, train_transactions_numerified):
    train_transactions_extracted_features = extract_transactions_features(train_transactions_numerified)
    return (train_transactions_extracted_features,)


@app.cell
def _(extract_transactions_features, validate_transactions_numerified):
    validate_transactions_extracted_features = extract_transactions_features(validate_transactions_numerified)
    return (validate_transactions_extracted_features,)


@app.cell
def _(df, duckdb):
    def numerify_app_activity(df):
        df = df.copy()
        df.loc[:, "app_activity_CAT_C3_float"] = df["CAT_C3"].astype("float64").fillna(0)
        df.loc[:, "app_activity_CAT_C4_float"] = df["CAT_C4"].astype("float64").fillna(0)
        df.loc[:, "app_activity_CAT_C5_float"] = df["CAT_C5"].astype("float64").fillna(0)
        df.loc[:, "app_activity_CAT_C6_float"] = df["CAT_C6"].astype("float64").fillna(0)
        df.loc[:, "app_activity_CAT_C8_int"] = df["CAT_C8"].cat.codes
        df.loc[:, "app_activity_CAT_C9_float"] = df["CAT_C9"].astype("float64").fillna(0)
        df.loc[:, "app_activity_CAT_C10_int"] = df["CAT_C10"].cat.codes
        return duckdb.sql(
            f"""
            SELECT
                df.CLIENT_ID,
                df.ACTIVITY_DATE,
                df.DEVICE_ID as app_activity_DEVICE_ID,
                df.app_activity_CAT_C3_float,
                df.app_activity_CAT_C4_float,
                df.app_activity_CAT_C5_float,
                df.app_activity_CAT_C6_float,
                df.app_activity_CAT_C8_int,
                df.app_activity_CAT_C9_float,
                df.app_activity_CAT_C10_int
            FROM
            	df
            """
        ).df()
    return (numerify_app_activity,)


@app.cell
def _(numerify_app_activity, train_app_activity):
    train_app_activity_numerified = numerify_app_activity(train_app_activity)
    return (train_app_activity_numerified,)


@app.cell
def _(numerify_app_activity, validate_app_activity):
    validate_app_activity_numerified = numerify_app_activity(validate_app_activity)
    return (validate_app_activity_numerified,)


@app.cell
def _(MinimalFCParameters, extract_features):
    def extract_app_activity_features(df):
        return extract_features(df, column_id="CLIENT_ID", column_sort="ACTIVITY_DATE", default_fc_parameters=MinimalFCParameters())
    return (extract_app_activity_features,)


@app.cell
def _(extract_app_activity_features, train_app_activity_numerified):
    train_app_activity_extracted_features = extract_app_activity_features(train_app_activity_numerified)
    return (train_app_activity_extracted_features,)


@app.cell
def _(extract_app_activity_features, validate_app_activity_numerified):
    validate_app_activity_extracted_features = extract_app_activity_features(validate_app_activity_numerified)
    return (validate_app_activity_extracted_features,)


@app.cell
def _(df, duckdb):
    def numerify_communications(df):
        df = df.copy()
        df.loc[:, "communications_CAT_C3_float"] = df["CAT_C3"].astype("float64").fillna(0)
        df.loc[:, "communications_CAT_C4_float"] = df["CAT_C4"].astype("float64").fillna(0)
        return duckdb.sql(
            f"""
            SELECT
                df.CLIENT_ID,
                df.CONTACT_DATE,
                df.communications_CAT_C3_float,
                df.communications_CAT_C4_float
            FROM
            	df
            """
        ).df()
    return (numerify_communications,)


@app.cell
def _(numerify_communications, train_communications):
    train_communications_numerified = numerify_communications(train_communications)
    return (train_communications_numerified,)


@app.cell
def _(numerify_communications, validate_communications):
    validate_communications_numerified = numerify_communications(validate_communications)
    return (validate_communications_numerified,)


@app.cell
def _(MinimalFCParameters, extract_features):
    def extract_communications_features(df):
        return extract_features(df, column_id="CLIENT_ID", column_sort="CONTACT_DATE", default_fc_parameters=MinimalFCParameters())
    return (extract_communications_features,)


@app.cell
def _(extract_communications_features, train_communications_numerified):
    train_communications_extracted_features = extract_communications_features(train_communications_numerified)
    return (train_communications_extracted_features,)


@app.cell
def _(extract_communications_features, validate_communications_numerified):
    validate_communications_extracted_features = extract_communications_features(validate_communications_numerified)
    return (validate_communications_extracted_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data playground
    """)
    return


@app.cell(hide_code=True)
def _(mo, train_app_activity):
    train_app_activity
    table_selector = mo.ui.dropdown(["train_app_activity", "train_communications", "train_transactions"], value="train_app_activity", label="Таблиця")
    return (table_selector,)


@app.cell(hide_code=True)
def _(mo, table_selector):
    mo.md(rf"""
    {table_selector}
    """)
    return


@app.cell(hide_code=True)
def _(mo, table_selector):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            {table_selector.value}
            JOIN clients_sample ON {table_selector.value}.CLIENT_ID = clients_sample.CLIENT_ID
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Підготовка валідаційного датасету
    """)
    return


@app.cell
def _(clients_sample, mo):
    non_test_clients_df = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            clients_sample.TARGET,
            clients_sample.COMMUNICATION_MONTH
        FROM
            clients_sample
        WHERE
            clients_sample.IS_TRAIN = TRUE
        """,
        output=False
    )
    return (non_test_clients_df,)


@app.cell
def _(non_test_clients_df, train_test_split):
    train_clients, validate_clients = train_test_split(non_test_clients_df, test_size=0.25, random_state=32)
    return train_clients, validate_clients


@app.function
def filter_dataset_by_clients(df, filter):
    return df[df['CLIENT_ID'].isin(filter['CLIENT_ID'])]


@app.cell
def _(
    app_activity,
    communications,
    train_clients,
    transactions,
    validate_clients,
):
    train_app_activity = filter_dataset_by_clients(app_activity, train_clients)
    train_communications = filter_dataset_by_clients(communications, train_clients)
    train_transactions = filter_dataset_by_clients(transactions, train_clients)

    validate_app_activity = filter_dataset_by_clients(app_activity, validate_clients)
    validate_communications = filter_dataset_by_clients(communications, validate_clients)
    validate_transactions = filter_dataset_by_clients(transactions, validate_clients)
    return (
        train_app_activity,
        train_communications,
        train_transactions,
        validate_app_activity,
        validate_communications,
        validate_transactions,
    )


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
    from tsfresh.feature_extraction import MinimalFCParameters
    return (
        LogisticRegression,
        MinimalFCParameters,
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
    import duckdb
    return duckdb, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Перший тиждень
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Коефіцієнти Gini

    Тренувальна вибірка: 0.4918

    Тестова вибірка: 0.4921
    """)
    return


if __name__ == "__main__":
    app.run()

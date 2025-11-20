import marimo

__generated_with = "0.17.7"
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
    ## Формування фічей

    - *act*, *tr*, *com* - підрахунок кількості записів у кожній таблиці на користувача;
    - *tr_c18* - середнє значення поля FLOAT_C18 із таблиці *transactions* для кожного користувача;
    - *tr_c19* - відсоток кількості значень '-1' серед кількості значень '-1' та '1' із поля INT_C19 із таблиці *transactions* для кожного користувача;
    - *com_c4_3*, *act_c6_1*, *act_c9_1* - відсоток кількості значень '3' із поля "CAT_C4" серед кількості рядків таблиці *communications* для кожного користувача. Аналогічно, кількість значень '1' із полей "CAT_C6" та "CAT_C9" серед кількості рядків таблиці *app_activity* для кожного користувача;
    - *act_min_date* - кількість днів від найдавнішого запису користувача в таблиці *app_activity* до 1 вересня 2025 р. Якщо записів немає, то береться число 180.

    Якщо не вказано інакше, то порожні записи у таблиці з фічами замінюються числом 0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data playground
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
    act_count_df = mo.sql(
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
    return (act_count_df,)


@app.cell
def _(clients_sample, mo, transactions):
    tr_count_df = mo.sql(
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
    return (tr_count_df,)


@app.cell
def _(clients_sample, communications, mo):
    com_count_df = mo.sql(
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
    return (com_count_df,)


@app.cell
def _(clients_sample, mo, transactions):
    tr_avg_c18_df = mo.sql(
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
    return (tr_avg_c18_df,)


@app.cell
def _(clients_sample, mo, transactions):
    tr_c19_f_df = mo.sql(
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
    return (tr_c19_f_df,)


@app.cell
def _(clients_sample, mo, transactions):
    tr_c19_t_df = mo.sql(
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
    return (tr_c19_t_df,)


@app.cell
def _(clients_sample, communications, mo):
    com_c4_3_df = mo.sql(
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
    return (com_c4_3_df,)


@app.cell
def _(app_activity, clients_sample, mo):
    act_c6_1_df = mo.sql(
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
    return (act_c6_1_df,)


@app.cell
def _(app_activity, clients_sample, mo):
    act_c9_1_df = mo.sql(
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
    return (act_c9_1_df,)


@app.cell
def _(app_activity, clients_sample, mo):
    act_min_date_df = mo.sql(
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
    return (act_min_date_df,)


@app.cell
def _(act_min_date_df, pd):
    target_date = pd.to_datetime('2025-09-01')
    act_min_date_df['time_diff'] = target_date - act_min_date_df['min']
    act_min_date_df['int'] = act_min_date_df['time_diff'].dt.days
    return


@app.cell
def _(
    act_c6_1_df,
    act_c9_1_df,
    act_count_df,
    act_min_date_df,
    clients_sample,
    com_c4_3_df,
    com_count_df,
    mo,
    tr_avg_c18_df,
    tr_c19_f_df,
    tr_c19_t_df,
    tr_count_df,
):
    features_df = mo.sql(
        f"""
        SELECT
            COALESCE(ANY_VALUE(act_count_df.count), 0) as 'act',
            COALESCE(ANY_VALUE(tr_count_df.count), 0) as 'tr',
            COALESCE(ANY_VALUE(com_count_df.count), 0) as 'com',
            COALESCE(ANY_VALUE(tr_avg_c18_df.avg), 0) as 'tr_c18',
            COALESCE(ANY_VALUE(tr_c19_f_df.count), 0) / (
                COALESCE(ANY_VALUE(tr_c19_f_df.count), 1) + COALESCE(ANY_VALUE(tr_c19_t_df.count), 0)
            ) as 'tr_c19',
            COALESCE(ANY_VALUE(com_c4_3_df.count), 0) / COALESCE(ANY_VALUE(com_count_df.count), 1) as 'com_c4_3',
            COALESCE(ANY_VALUE(act_c6_1_df.count), 0) / COALESCE(ANY_VALUE(act_count_df.count), 1) as 'act_c6_1',
            COALESCE(ANY_VALUE(act_c9_1_df.count), 0) / COALESCE(ANY_VALUE(act_count_df.count), 1) as 'act_c9_1',
            COALESCE(ANY_VALUE(act_min_date_df.int), 180) as 'act_min_date',
            ANY_VALUE(clients_sample.TARGET) AS 'TARGET',
            ANY_VALUE(clients_sample.IS_TRAIN) AS 'IS_TRAIN'
        FROM
            clients_sample
            LEFT JOIN act_count_df ON act_count_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN tr_count_df ON tr_count_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN com_count_df ON com_count_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN tr_avg_c18_df ON tr_avg_c18_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN tr_c19_f_df ON tr_c19_f_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN tr_c19_t_df ON tr_c19_t_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN com_c4_3_df ON com_c4_3_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN act_c6_1_df ON act_c6_1_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN act_c9_1_df ON act_c9_1_df.CLIENT_ID = clients_sample.CLIENT_ID
            LEFT JOIN act_min_date_df ON act_min_date_df.CLIENT_ID = clients_sample.CLIENT_ID
        GROUP BY
            clients_sample.CLIENT_ID;
        """
    )
    return (features_df,)


@app.cell
def _():
    features = ['act', 'tr', 'com', 'tr_c18', 'tr_c19', 'com_c4_3', 'act_c6_1', 'act_c9_1', 'act_min_date']
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
    features_test_scaled = scaler.fit_transform(features_test)
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
    from sklearn.metrics import classification_report, roc_auc_score
    return LogisticRegression, StandardScaler, roc_auc_score


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

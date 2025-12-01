import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(feature_importance, features_list, plt, sns):
    sns.set_style("whitegrid")

    top_n_features = min(400, len(features_list))

    plt.rcParams["figure.dpi"] = 120
    plt.figure(figsize=(10, top_n_features / 4.5))

    sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(top_n_features))

    plt.title('Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(features_list, mo, train_gini, validate_gini):
    mo.md(rf"""
    ## Коефіцієнти Gini

    Тренувальна вибірка: {train_gini:.4f} (0.6587) (0.9078)

    Валідаційна вибірка: {validate_gini:.4f} (0.5703) (0.4202)

    Усього фіч: {len(features_list)}
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

    model = LogisticRegression(random_state=32, max_iter=1200, class_weight='balanced')
    model.fit(train_features_scaled, train_target)

    train_proba = model.predict_proba(train_features_scaled)[:, 1]

    validate_features = validate_df[features_list]
    validate_target = validate_df['TARGET'].astype(int)

    validate_features_scaled = scaler.transform(validate_features)
    validate_proba = model.predict_proba(validate_features_scaled)[:, 1]

    train_gini = gini(roc_auc_score(train_target, train_proba))
    validate_gini = gini(roc_auc_score(validate_target, validate_proba))
    return model, scaler, train_gini, validate_gini


@app.cell
def _(features_list, model, roc_auc_score, scaler, test_df):
    test_features = test_df[features_list]
    test_target = test_df['TARGET'].astype(int)

    test_features_scaled = scaler.transform(test_features)
    test_proba = model.predict_proba(test_features_scaled)[:, 1]

    test_gini = gini(roc_auc_score(test_target, test_proba))
    print(f"Тестова вибірка: {test_gini:.4f}")
    return


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
    features_list = list(train_transactions_extracted_features.columns) + list(train_app_activity_extracted_features.columns) + list(train_communications_extracted_features.columns) + ["count_app_activity_per_user", "count_communications_per_user", "count_transactions_per_user", "days_old_ACTIVITY_DATE", "days_old_COMMUNICATION_MONTH"]
    train_transactions_extracted_features_medians = train_transactions_extracted_features.median().fillna(0)
    train_app_activity_extracted_features_medians = train_app_activity_extracted_features.median().fillna(0)
    train_communications_extracted_features_medians = train_communications_extracted_features.median().fillna(0)
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
    np,
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
        complete_df = df_with_all_clients.fillna(train_transactions_extracted_features_medians).fillna(train_app_activity_extracted_features_medians).fillna(train_communications_extracted_features_medians)
        complete_df.replace(-np.inf, -1.08, inplace=True)
        return complete_df
    return (prepare_dataset,)


@app.cell
def _(
    prepare_dataset,
    train_app_activity_extracted_features,
    train_clients,
    train_communications_extracted_features,
    train_my_features,
    train_transactions_extracted_features,
):
    train_df = prepare_dataset(train_clients, train_transactions_extracted_features, train_app_activity_extracted_features, train_communications_extracted_features, train_my_features)
    return (train_df,)


@app.cell
def _(
    prepare_dataset,
    validate_app_activity_extracted_features,
    validate_clients,
    validate_communications_extracted_features,
    validate_my_features,
    validate_transactions_extracted_features,
):
    validate_df = prepare_dataset(validate_clients, validate_transactions_extracted_features, validate_app_activity_extracted_features, validate_communications_extracted_features, validate_my_features)
    return (validate_df,)


@app.cell
def _(
    prepare_dataset,
    test_app_activity_extracted_features,
    test_clients,
    test_communications_extracted_features,
    test_my_features,
    test_transactions_extracted_features,
):
    test_df = prepare_dataset(test_clients, test_transactions_extracted_features, test_app_activity_extracted_features, test_communications_extracted_features, test_my_features)
    return (test_df,)


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
                clients_df.COMMUNICATION_MONTH
            FROM
                clients_df
                LEFT JOIN days_old_ACTIVITY_DATE ON days_old_ACTIVITY_DATE.CLIENT_ID = clients_df.CLIENT_ID
            """
        ).df()
        df_with_all_clients_days_fillna = df_with_all_clients_days.fillna(180)
        df_with_all_clients_days_fillna["month_diff"] = target_date - pd.to_datetime(df_with_all_clients_days_fillna["COMMUNICATION_MONTH"])
        df_with_all_clients_days_fillna['int'] = df_with_all_clients_days_fillna['month_diff'].dt.days
        df_with_all_clients = duckdb.sql(
            """
            SELECT
                df_with_all_clients_days_fillna.CLIENT_ID,
                df_with_all_clients_days_fillna.days_old_ACTIVITY_DATE,
                df_with_all_clients_days_fillna.int as 'days_old_COMMUNICATION_MONTH',
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


@app.cell
def _(
    prepair_my_features,
    test_app_activity,
    test_clients,
    test_communications,
    test_transactions,
):
    test_my_features = prepair_my_features(test_app_activity, test_communications, test_transactions, test_clients)
    return (test_my_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Генерація фіч
    """)
    return


@app.cell(hide_code=True)
def _(mo, recalculate_extracted_features_switch):
    mo.md(rf"""
    {recalculate_extracted_features_switch}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    recalculate_extracted_features_switch = mo.ui.switch(label="Recalculate extracted features", value=False)
    test_recalculate_extracted_features_switch = mo.ui.switch(label="Recalculate extracted features for test", value=False)
    return (
        recalculate_extracted_features_switch,
        test_recalculate_extracted_features_switch,
    )


@app.cell
def _(
    extract_transactions_features,
    pd,
    recalculate_extracted_features_switch,
    train_transactions_numerified,
):
    if recalculate_extracted_features_switch.value:
        train_transactions_extracted_features = extract_transactions_features(train_transactions_numerified)
        train_transactions_extracted_features.to_pickle("train_transactions_extracted_features.pkl")
    else:
        train_transactions_extracted_features = pd.read_pickle("train_transactions_extracted_features.pkl")
    return (train_transactions_extracted_features,)


@app.cell
def _(
    extract_transactions_features,
    pd,
    recalculate_extracted_features_switch,
    validate_transactions_numerified,
):
    if recalculate_extracted_features_switch.value:
        validate_transactions_extracted_features = extract_transactions_features(validate_transactions_numerified)
        validate_transactions_extracted_features.to_pickle("validate_transactions_extracted_features.pkl")
    else:
        validate_transactions_extracted_features = pd.read_pickle("validate_transactions_extracted_features.pkl")
    return (validate_transactions_extracted_features,)


@app.cell
def _(
    extract_app_activity_features,
    pd,
    recalculate_extracted_features_switch,
    train_app_activity_numerified,
):
    if recalculate_extracted_features_switch.value:
        train_app_activity_extracted_features = extract_app_activity_features(train_app_activity_numerified)
        train_app_activity_extracted_features.to_pickle("train_app_activity_extracted_features.pkl")
    else:
        train_app_activity_extracted_features = pd.read_pickle("train_app_activity_extracted_features.pkl")
    return (train_app_activity_extracted_features,)


@app.cell
def _(
    extract_app_activity_features,
    pd,
    recalculate_extracted_features_switch,
    validate_app_activity_numerified,
):
    if recalculate_extracted_features_switch.value:
        validate_app_activity_extracted_features = extract_app_activity_features(validate_app_activity_numerified)
        validate_app_activity_extracted_features.to_pickle("validate_app_activity_extracted_features.pkl")
    else:
        validate_app_activity_extracted_features = pd.read_pickle("validate_app_activity_extracted_features.pkl")
    return (validate_app_activity_extracted_features,)


@app.cell
def _(
    extract_communications_features,
    pd,
    recalculate_extracted_features_switch,
    train_communications_numerified,
):
    if recalculate_extracted_features_switch.value:
        train_communications_extracted_features = extract_communications_features(train_communications_numerified)
        train_communications_extracted_features.to_pickle("train_communications_extracted_features.pkl")
    else:
        train_communications_extracted_features = pd.read_pickle("train_communications_extracted_features.pkl")
    return (train_communications_extracted_features,)


@app.cell
def _(
    extract_communications_features,
    pd,
    recalculate_extracted_features_switch,
    validate_communications_numerified,
):
    if recalculate_extracted_features_switch.value:
        validate_communications_extracted_features = extract_communications_features(validate_communications_numerified)
        validate_communications_extracted_features.to_pickle("validate_communications_extracted_features.pkl")
    else:
        validate_communications_extracted_features = pd.read_pickle("validate_communications_extracted_features.pkl")
    return (validate_communications_extracted_features,)


@app.cell
def _(df, duckdb):
    def numerify_transactions(df):
        df = df.copy()
        df.loc[:, "transactions_CAT_C2_int"] = df["CAT_C2"].cat.codes
        df.loc[:, "transactions_CAT_C3_int"] = df["CAT_C3"].cat.codes
        df.loc[:, "transactions_CAT_C4_int"] = df["CAT_C4"].cat.codes
        # df.loc[:, "transactions_FL_C6_int"] = df["FL_C6"].astype("int64")
        # df.loc[:, "transactions_FL_C7_int"] = df["FL_C7"].astype("int64")
        # df.loc[:, "transactions_FL_C8_int"] = df["FL_C8"].astype("int64")
        # df.loc[:, "transactions_FL_C9_int"] = df["FL_C9"].astype("int64")
        # df.loc[:, "transactions_FL_C10_int"] = df["FL_C10"].astype("int64")
        # df.loc[:, "transactions_FL_C11_int"] = df["FL_C11"].astype("int64")
        df.loc[:, "transactions_FL_C12_int"] = df["FL_C12"].astype("int64")
        df.loc[:, "transactions_FL_C13_int"] = df["FL_C13"].astype("int64")
        df.loc[:, "transactions_FL_C14_int"] = df["FL_C14"].astype("int64")
        df.loc[:, "transactions_FL_C15_int"] = df["FL_C15"].astype("int64")
        return duckdb.sql(
            f"""
            SELECT
                df.CLIENT_ID,
                df.TRAN_DATE,
                df.transactions_CAT_C2_int,
                df.transactions_CAT_C3_int,
                df.transactions_CAT_C4_int,
                df.transactions_FL_C12_int,
                df.transactions_FL_C13_int,
                df.transactions_FL_C14_int,
                df.transactions_FL_C15_int,
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
def _(numerify_transactions, test_transactions):
    test_transactions_numerified = numerify_transactions(test_transactions)
    return (test_transactions_numerified,)


@app.cell
def _(extract_features, myFCParameters):
    def extract_transactions_features(df):
        return extract_features(df, column_id="CLIENT_ID", column_sort="TRAN_DATE", default_fc_parameters=myFCParameters)
    return (extract_transactions_features,)


@app.cell(hide_code=True)
def _(mo, test_recalculate_extracted_features_switch):
    mo.md(rf"""
    {test_recalculate_extracted_features_switch}
    """)
    return


@app.cell
def _(
    extract_transactions_features,
    pd,
    test_recalculate_extracted_features_switch,
    test_transactions_numerified,
):
    if test_recalculate_extracted_features_switch.value:
        test_transactions_extracted_features = extract_transactions_features(test_transactions_numerified)
        test_transactions_extracted_features.to_pickle("test_transactions_extracted_features.pkl")
    else:
        test_transactions_extracted_features = pd.read_pickle("test_transactions_extracted_features.pkl")
    return (test_transactions_extracted_features,)


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
        df.loc[:, "app_activity_FLOAT_C11"] = df["FLOAT_C11"].fillna(0)
        df.loc[:, "app_activity_FLOAT_C12"] = df["FLOAT_C12"].fillna(0)
        # df.loc[:, "app_activity_FLOAT_C13"] = df["FLOAT_C13"].fillna(0)
        df.loc[:, "app_activity_FLOAT_C14"] = df["FLOAT_C14"].fillna(0)
        # df.loc[:, "app_activity_FLOAT_C15"] = df["FLOAT_C15"].fillna(0)
        # df.loc[:, "app_activity_FLOAT_C16"] = df["FLOAT_C16"].fillna(0)
        # df.loc[:, "app_activity_FLOAT_C17"] = df["FLOAT_C17"].fillna(0)
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
                df.app_activity_CAT_C10_int,
                df.app_activity_FLOAT_C11,
                df.app_activity_FLOAT_C12,
                df.app_activity_FLOAT_C14
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
def _(numerify_app_activity, test_app_activity):
    test_app_activity_numerified = numerify_app_activity(test_app_activity)
    return (test_app_activity_numerified,)


@app.cell
def _(extract_features, myFCParameters):
    def extract_app_activity_features(df):
        return extract_features(df, column_id="CLIENT_ID", column_sort="ACTIVITY_DATE", default_fc_parameters=myFCParameters)
    return (extract_app_activity_features,)


@app.cell
def _(
    extract_app_activity_features,
    pd,
    test_app_activity_numerified,
    test_recalculate_extracted_features_switch,
):
    if test_recalculate_extracted_features_switch.value:
        test_app_activity_extracted_features = extract_app_activity_features(test_app_activity_numerified)
        test_app_activity_extracted_features.to_pickle("test_app_activity_extracted_features.pkl")
    else:
        test_app_activity_extracted_features = pd.read_pickle("test_app_activity_extracted_features.pkl")
    return (test_app_activity_extracted_features,)


@app.cell
def _(df, duckdb):
    def numerify_communications(df):
        df = df.copy()
        df.loc[:, "communications_CAT_C2_int"] = df["CAT_C2"].cat.codes
        df.loc[:, "communications_CAT_C3_float"] = df["CAT_C3"].astype("float64").fillna(0)
        df.loc[:, "communications_CAT_C4_float"] = df["CAT_C4"].astype("float64").fillna(0)
        df.loc[:, "communications_CAT_C5_int"] = df["CAT_C5"].cat.codes
        return duckdb.sql(
            f"""
            SELECT
                df.CLIENT_ID,
                df.CONTACT_DATE,
                df.communications_CAT_C2_int,
                df.communications_CAT_C3_float,
                df.communications_CAT_C4_float,
                df.communications_CAT_C5_int
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
def _(numerify_communications, test_communications):
    test_communications_numerified = numerify_communications(test_communications)
    return (test_communications_numerified,)


@app.cell
def _(extract_features, myFCParameters):
    def extract_communications_features(df):
        return extract_features(df, column_id="CLIENT_ID", column_sort="CONTACT_DATE", default_fc_parameters=myFCParameters)
    return (extract_communications_features,)


@app.cell
def _(
    extract_communications_features,
    pd,
    test_communications_numerified,
    test_recalculate_extracted_features_switch,
):
    if test_recalculate_extracted_features_switch.value:
        test_communications_extracted_features = extract_communications_features(test_communications_numerified)
        test_communications_extracted_features.to_pickle("test_communications_extracted_features.pkl")
    else:
        test_communications_extracted_features = pd.read_pickle("test_communications_extracted_features.pkl")
    return (test_communications_extracted_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Підбір агрегацій
    """)
    return


@app.cell
def _(EfficientFCParameters):
    myFCParameters = EfficientFCParameters() # comprehensive has methods that crash
    del myFCParameters["binned_entropy"] # crashes with app_activity
    del myFCParameters["augmented_dickey_fuller"] # very long
    del myFCParameters["number_cwt_peaks"] # very long
    # del myFCParameters["cwt_coefficients"] # seems to slightly grow in memory and time usage
    del myFCParameters["change_quantiles"] # very long
    myFCParameters["fft_coefficient"] = [{"coeff":0,"attr":"real"},{"coeff":1,"attr":"real"},{"coeff":2,"attr":"real"},{"coeff":3,"attr":"real"},{"coeff":4,"attr":"real"},{"coeff":5,"attr":"real"},{"coeff":6,"attr":"real"},{"coeff":7,"attr":"real"},{"coeff":8,"attr":"real"},{"coeff":9,"attr":"real"},{"coeff":10,"attr":"real"},{"coeff":0,"attr":"imag"},{"coeff":1,"attr":"imag"},{"coeff":2,"attr":"imag"},{"coeff":3,"attr":"imag"},{"coeff":4,"attr":"imag"},{"coeff":5,"attr":"imag"},{"coeff":6,"attr":"imag"},{"coeff":7,"attr":"imag"},{"coeff":8,"attr":"imag"},{"coeff":9,"attr":"imag"},{"coeff":10,"attr":"imag"},{"coeff":0,"attr":"abs"},{"coeff":1,"attr":"abs"},{"coeff":2,"attr":"abs"},{"coeff":3,"attr":"abs"},{"coeff":4,"attr":"abs"},{"coeff":5,"attr":"abs"},{"coeff":6,"attr":"abs"},{"coeff":7,"attr":"abs"},{"coeff":8,"attr":"abs"},{"coeff":9,"attr":"abs"},{"coeff":10,"attr":"abs"},{"coeff":0,"attr":"angle"},{"coeff":1,"attr":"angle"},{"coeff":2,"attr":"angle"},{"coeff":3,"attr":"angle"},{"coeff":4,"attr":"angle"},{"coeff":5,"attr":"angle"},{"coeff":6,"attr":"angle"},{"coeff":7,"attr":"angle"},{"coeff":8,"attr":"angle"},{"coeff":9,"attr":"angle"},{"coeff":10,"attr":"angle"}] # takes too much time and memory
    del myFCParameters["agg_linear_trend"] # very long
    del myFCParameters["lempel_ziv_complexity"] # long
    # del myFCParameters["fourier_entropy"] # long
    # del myFCParameters["permutation_entropy"] # long
    myFCParameters["count_above"] = [{"t": 0}, {"t": 1}, {"t": 2}] # added two move values
    myFCParameters["count_below"] = [{"t": 0}, {"t": 1}, {"t": 2}] # added two move values
    return (myFCParameters,)


@app.cell
def _(mo, myFCParameters):
    aggregation_selector_dropdown = mo.ui.dropdown(label="Aggregation", options=list(myFCParameters.keys()), value="length")
    return (aggregation_selector_dropdown,)


@app.cell(hide_code=True)
def _(aggregation_selector_dropdown, mo):
    mo.md(rf"""
    {aggregation_selector_dropdown}
    """)
    return


@app.cell
def _(aggregation_selector_dropdown, myFCParameters):
    newFCset = {aggregation_selector_dropdown.value: myFCParameters[aggregation_selector_dropdown.value]}
    # validate_extraction_speed = extract_features(validate_app_activity_numerified, column_id="CLIENT_ID", column_sort="ACTIVITY_DATE", default_fc_parameters=newFCset)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data playground
    """)
    return


@app.cell(hide_code=True)
def _(mo, train_app_activity):
    data_playground_switch = mo.ui.switch(label="Показати", value=False)
    train_app_activity
    table_selector = mo.ui.dropdown(["train_app_activity", "train_communications", "train_transactions"], value="train_app_activity", label="Таблиця")
    return data_playground_switch, table_selector


@app.cell(hide_code=True)
def _(data_playground_switch, mo, table_selector):
    mo.md(rf"""
    {data_playground_switch}

    {table_selector}
    """)
    return


@app.cell(hide_code=True)
def _(data_playground_switch, mo, table_selector):
    if data_playground_switch.value:
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
def _(app_activity, communications, train_clients, transactions):
    train_app_activity = filter_dataset_by_clients(app_activity, train_clients)
    train_communications = filter_dataset_by_clients(communications, train_clients)
    train_transactions = filter_dataset_by_clients(transactions, train_clients)
    return train_app_activity, train_communications, train_transactions


@app.cell
def _(app_activity, communications, transactions, validate_clients):
    validate_app_activity = filter_dataset_by_clients(app_activity, validate_clients)
    validate_communications = filter_dataset_by_clients(communications, validate_clients)
    validate_transactions = filter_dataset_by_clients(transactions, validate_clients)
    return (
        validate_app_activity,
        validate_communications,
        validate_transactions,
    )


@app.cell(disabled=True)
def _(clients_sample, mo):
    test_clients = mo.sql(
        f"""
        SELECT
            clients_sample.CLIENT_ID,
            clients_sample.TARGET,
            clients_sample.COMMUNICATION_MONTH
        FROM
            clients_sample
        WHERE
            clients_sample.IS_TRAIN = FALSE
        """,
        output=False
    )
    return (test_clients,)


@app.cell
def _(app_activity, communications, test_clients, transactions):
    test_app_activity = filter_dataset_by_clients(app_activity, test_clients)
    test_communications = filter_dataset_by_clients(communications, test_clients)
    test_transactions = filter_dataset_by_clients(transactions, test_clients)
    return test_app_activity, test_communications, test_transactions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Завантаження даних
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    # read client sample
    # client_sample_dtypes = {
    #    'CLIENT_ID': 'uint64',
    #    'TARGET': 'boolean',
    #    'IS_TRAIN': 'boolean',
    # }
    # clients_sample = pd.read_csv(
    #    r"/CLIENTS_SAMPLE.csv",
    #    sep=',',
    #    dtype=client_sample_dtypes
    # )
    # clients_sample.to_pickle("clients_sample.pkl")
    clients_sample = pd.read_pickle("clients_sample.pkl")
    return (clients_sample,)


@app.cell(hide_code=True)
def _(pd):
    # read app activity
    # app_activity_dtypes = {
    #    'CLIENT_ID': 'uint64',
    #    'DEVICE_ID': 'uint64',
    #    'CAT_C3': 'category',
    #    'CAT_C4': 'category',
    #    'CAT_C5': 'category',
    #    'CAT_C6': 'category',
    #    'CAT_C8': 'category',
    #    'CAT_C9': 'category',
    #    'CAT_C10': 'category',
    #    'FLOAT_C11': 'float32',
    #    'FLOAT_C12': 'float32',
    #    'FLOAT_C13': 'float32',
    #    'FLOAT_C14': 'float32',
    #    'FLOAT_C15': 'float32',
    #    'FLOAT_C16': 'float32',
    #    'FLOAT_C17': 'float32'
    # }

    # app_activity = pd.read_csv(
    #    r"/APP_ACTIVITY.csv",
    #    sep=',',
    #    dtype=app_activity_dtypes,
    #    parse_dates=["ACTIVITY_DATE"],
    # )
    # app_activity.to_pickle("app_activity.pkl")
    app_activity = pd.read_pickle("app_activity.pkl")
    return (app_activity,)


@app.cell(hide_code=True)
def _(pd):
    # read communications
    # communications_dtypes = {
    #    'CLIENT_ID': 'uint64',
    #    "CAT_C2": "category",
    #    "CAT_C3": "category",
    #    "CAT_C4": "category",
    #    "CAT_C5": "category",

    # }

    # communications = pd.read_csv(
    #    r"/COMMUNICATIONS.csv",
    #    sep=',',
    #    dtype=communications_dtypes,
    #    parse_dates=["CONTACT_DATE"],
    # )
    # communications.to_pickle("communications.pkl")
    communications = pd.read_pickle("communications.pkl")
    return (communications,)


@app.cell(hide_code=True)
def _(pd):
    # read transactions
    # transactions_dtypes = {
    #    'CLIENT_ID': 'uint64',
    #    'CAT_C2': 'category',
    #    'CAT_C3': 'category',
    #    'CAT_C4': 'category',
    #    'FL_C6': 'bool',
    #    'FL_C7': 'bool',
    #    'FL_C8': 'bool',
    #    'FL_C9': 'bool',
    #    'FL_C10': 'bool',
    #    'FL_C11': 'bool',
    #    'FL_C12': 'bool',
    #    'FL_C13': 'bool',
    #    'FL_C14': 'bool',
    #    'FL_C15': 'bool',
    #    'FLOAT_C16': 'float32',
    #    'FLOAT_C17': 'float32',
    #    'FLOAT_C18': 'float32',
    #    'INT_C19': 'int32',
    #    'FLOAT_C20': 'float32',
    #    'FLOAT_C21': 'float32'
    # }

    # transactions = pd.read_csv(
    #    r"/TRANSACTIONS.csv",
    #    sep=',',
    #    dtype=transactions_dtypes,
    #    parse_dates=["TRAN_DATE"],
    # )
    # transactions.to_pickle("transactions.pkl")
    transactions = pd.read_pickle("transactions.pkl")
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
    from tsfresh.feature_extraction import EfficientFCParameters
    return (
        EfficientFCParameters,
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
    import numpy as np
    import duckdb
    return duckdb, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Третій тиждень

    ## Коефіцієнти Gini

    Тренувальна вибірка: 0.6587

    Валідаційна вибірка: 0.5703

    Тестова вибірка: 0.5859
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Перший тиждень

    ## Коефіцієнти Gini

    Тренувальна вибірка: 0.4918

    Тестова вибірка: 0.4921
    """)
    return


if __name__ == "__main__":
    app.run()

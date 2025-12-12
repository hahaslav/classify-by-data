import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(feature_importance, features_list, plt, sns):
    sns.set_style("whitegrid")

    top_n_features = min(400, len(features_list))

    plt.rcParams["figure.dpi"] = 120
    plt.figure(figsize=(10, top_n_features / 4.5))

    sns.barplot(x='Importance Score', y='Feature', data=feature_importance.head(top_n_features))

    plt.title('Feature Importance')
    plt.xlabel('Importance Score (Gain)')
    plt.ylabel('Feature')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(features_list, mo, train_gini, validate_gini):
    mo.md(rf"""
    ## Коефіцієнти Gini

    Тренувальна вибірка: {train_gini:.4f} (0.6587) (0.9998)

    Валідаційна вибірка: {validate_gini:.4f} (0.5703) (0.5706)

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
    StandardScaler,
    features_list,
    roc_auc_score,
    train_df,
    validate_df,
    xgb,
):
    train_features = train_df[features_list]
    train_target = train_df['TARGET'].astype(int)

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    model = xgb.XGBClassifier(objective='binary:logistic', random_state=32, eval_metric='logloss')
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
def _(model, pd):
    importance_dict = model.get_booster().get_score(importance_type='gain')

    feature_importance = pd.DataFrame(
        list(importance_dict.items()),
        columns=['Feature', 'Importance Score']
    ).sort_values(by='Importance Score', ascending=False)
    return (feature_importance,)


@app.cell
def _(features_list, model, pd):
    importance_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': model.feature_importances_
    })

    importance_df[importance_df['Importance'] == 0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Очищення датасетів
    """)
    return


@app.cell
def _(VarianceThreshold, features_list_full, train_full):
    full_df = train_full[features_list_full]

    variance_filter = VarianceThreshold(threshold=0.1)
    variance_df = variance_filter.fit_transform(full_df)
    return


@app.cell
def _(features_list_full):
    # features_list = variance_df.columns
    features_list = features_list_full
    return (features_list,)


@app.cell
def _(train_full):
    # train_df = variance_df.copy()
    # train_df["TARGET"] = train_full["TARGET"]
    train_df = train_full
    return (train_df,)


@app.cell
def _(validate_full):
    # validate_df = validate_full[features_list].copy()
    # validate_df["TARGET"] = validate_full["TARGET"]
    validate_df = validate_full
    return (validate_df,)


@app.cell
def _(test_full):
    # test_df = test_full[features_list].copy()
    # test_df["TARGET"] = test_full["TARGET"]
    test_df = test_full
    return (test_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Завантаження згенерованих датасетів
    """)
    return


@app.cell
def _(pd):
    train_full = pd.read_pickle("train_full.pkl")
    return (train_full,)


@app.cell
def _(pd):
    validate_full = pd.read_pickle("validate_full.pkl")
    return (validate_full,)


@app.cell(disabled=True)
def _(pd):
    test_full = pd.read_pickle("test_full.pkl")
    return (test_full,)


@app.cell
def _(json):
    with open("features_list_full.json", 'r', encoding="UTF-8") as fin:
        features_list_full = json.load(fin)
    return (features_list_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Бібліотеки
    """)
    return


@app.cell
def _():
    from sklearn import set_config
    set_config(transform_output="pandas")
    return


@app.cell
def _():
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    return StandardScaler, VarianceThreshold, roc_auc_score, xgb


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
    import json
    return json, mo, pd


if __name__ == "__main__":
    app.run()

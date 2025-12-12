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

    Валідаційна вибірка: {validate_gini:.4f} (0.5703) (0.5634)

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
def _(features_list, model, pd):
    importance_dict = model.get_booster().get_score(importance_type='gain')
    feature_map = {f'f{i}': name for i, name in enumerate(features_list)}
    mapped_importance = {feature_map[k]: v for k, v in importance_dict.items() if k in feature_map}

    feature_importance = pd.DataFrame(
        list(mapped_importance.items()), 
        columns=['Feature', 'Importance Score']
    ).sort_values(by='Importance Score', ascending=False)
    return (feature_importance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Підготовка згенерованих датасетів до тренування
    """)
    return


@app.cell
def _(pd):
    train_df = pd.read_pickle("train_df_full.pkl")
    return (train_df,)


@app.cell
def _(pd):
    validate_df = pd.read_pickle("validate_df_full.pkl")
    return (validate_df,)


@app.cell(disabled=True)
def _(pd):
    test_df = pd.read_pickle("test_df_full.pkl")
    return (test_df,)


@app.cell
def _(json):
    with open("features_list_full.json", 'r', encoding="UTF-8") as fin:
        features_list = json.load(fin)
    return (features_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Бібліотеки
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    return StandardScaler, roc_auc_score, xgb


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

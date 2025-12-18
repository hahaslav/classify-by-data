import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(feature_importance, features_list, plt, sns):
    sns.set_style("whitegrid")

    top_n_features = min(300, len(features_list))

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


@app.cell(hide_code=True)
def _(mo):
    train_model_button = mo.ui.run_button(label="Тренувати модель")
    train_model_button
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    features_list,
    roc_auc_score,
    train_df,
    validate_df,
):
    # mo.stop(not train_model_button.value)
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


@app.cell(disabled=True)
def _(feature_importance, json):
    with open("top_sm_features.json", 'w', encoding="UTF-8") as fout:
        json.dump(list(feature_importance["Feature"]), fout)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Очищення датасетів
    """)
    return


@app.cell
def _(load_json):
    top_decorr_columns = load_json("top_144_features.json")
    return (top_decorr_columns,)


@app.cell
def _(random):
    def get_random_columns(columns, n=50):
        return random.sample(columns, n)
    return (get_random_columns,)


@app.cell
def _(mo):
    minutes_search_input = mo.ui.number(start=0, stop=60, value=0.01, label="Хвилини для пошуку моделі")
    minutes_search_input
    return (minutes_search_input,)


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    features_list_full,
    get_random_columns,
    minutes_search_input,
    roc_auc_score,
    top_decorr_columns,
    train_full,
    validate_full,
):
    def find_best_model(columns, minutes=1):
        full_df = train_full[features_list_full]
        top_gini = 0
        top_features = []
        for i in range(int(minutes * 400)):
            features_list = get_random_columns(columns)

            train_df = full_df[features_list].copy()
            train_df["TARGET"] = train_full["TARGET"]
        
            validate_df = validate_full[features_list].copy()
            validate_df["TARGET"] = validate_full["TARGET"]
        
            train_features = train_df[features_list]
            train_target = train_df['TARGET'].astype(int)
        
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
        
            model = LogisticRegression(random_state=32, max_iter=100, class_weight='balanced')
            model.fit(train_features_scaled, train_target)
        
            validate_features = validate_df[features_list]
            validate_target = validate_df['TARGET'].astype(int)
        
            validate_features_scaled = scaler.transform(validate_features)
            validate_proba = model.predict_proba(validate_features_scaled)[:, 1]
        
            validate_gini = gini(roc_auc_score(validate_target, validate_proba))
            if validate_gini > top_gini:
                top_features = features_list
                top_gini = validate_gini
        return top_features

    best_features = find_best_model(top_decorr_columns, minutes_search_input.value)
    return (best_features,)


@app.cell
def _(best_features, json, minutes_search_input):
    def save_json(object, filename):
        with open(filename, 'w', encoding="UTF-8") as fout:
            json.dump(object, fout, ensure_ascii=False)

    save_json(best_features, f"{minutes_search_input.value}m_top_selected_50_features.json")
    return


@app.cell
def _(full_df, load_json):
    top_features_columns = load_json("top_100_features.json")
    top_feat_df = full_df[top_features_columns]
    return (top_feat_df,)


@app.cell
def _(top_feat_df):
    corr_matrix = top_feat_df.corr().abs()
    return (corr_matrix,)


@app.cell
def _(corr_matrix, np, top_feat_df):
    corr_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_to_drop = [c for c in corr_triangle.columns if any(corr_triangle[c] > 0.7)]
    decorr_df = top_feat_df.drop(columns=corr_to_drop)
    return


@app.cell(disabled=True)
def _(full_df):
    deduplicated_df = full_df.T.drop_duplicates().T
    return (deduplicated_df,)


@app.cell
def _(features_list_full, train_full):
    full_df = train_full[features_list_full]
    return (full_df,)


@app.cell
def _(VarianceThreshold, deduplicated_df):
    variance_filter = VarianceThreshold(threshold=0.01)
    variance_df = variance_filter.fit_transform(deduplicated_df)
    return


@app.cell
def _(best_features, full_df, train_full):
    train_df = full_df[best_features].copy()
    features_list = train_df.columns
    train_df["TARGET"] = train_full["TARGET"]
    return features_list, train_df


@app.cell
def _(features_list, validate_full):
    validate_df = validate_full[features_list].copy()
    validate_df["TARGET"] = validate_full["TARGET"]
    return (validate_df,)


@app.cell
def _(features_list, test_full):
    test_df = test_full[features_list].copy()
    test_df["TARGET"] = test_full["TARGET"]
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
    def load_json(filename):
        with open(filename, 'r', encoding="UTF-8") as fin:
            return json.load(fin)

    features_list_full = load_json("features_list_full.json")
    return features_list_full, load_json


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
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression, StandardScaler, VarianceThreshold, roc_auc_score


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
    import random
    return json, mo, np, pd, random


if __name__ == "__main__":
    app.run()

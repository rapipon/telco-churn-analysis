import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return mo, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("data/raw/customer_data.csv")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    df.duplicated().sum()
    return


@app.cell
def _(df):
    df["Churn"].value_counts()
    return


@app.cell
def _(df):
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({
        0: "No",
        1: "Yes"
    })

    df.head()
    return


@app.cell
def _(df, pd):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return


@app.cell
def _(df):
    df["TotalCharges"].dtype
    return


@app.cell
def _(df):
    df["TotalCharges"].isna().sum()
    return


@app.cell
def _(df):
    df[df["tenure"] == 0]
    return


@app.cell
def _(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    cat_cols = df.select_dtypes(include=["str", "object"]).columns.drop("customerID")

    num_cols
    return cat_cols, num_cols


@app.cell
def _(cat_cols):
    cat_cols
    return


@app.cell
def _(df):
    df["Churn"].value_counts(normalize=True)
    return


@app.cell
def _(df, plt, sns):
    sns.countplot(data=df, x ="Churn")

    plt.title("Распределение таргета")
    plt.show()
    return


@app.cell
def _(df, num_cols, plt):
    df[num_cols].hist(figsize=(15,10))

    plt.show()
    return


@app.cell
def _(col, df, num_cols, plt, sns):
    for _col in num_cols:
        plt.figure(figsize=(8, 3))

        sns.boxplot(x=df[col])

        plt.title(col)
        plt.show()
    return


@app.cell
def _(cat_cols, df):
    for _col in cat_cols:
        print(df[_col].value_counts())
    return


@app.cell
def _(cat_cols, df, plt, sns):
    for _col in cat_cols:
        plt.figure(figsize=(8, 4))

        sns.countplot(data=df, x=_col)

        plt.xticks(rotation=45)
        plt.show()
    return


@app.cell
def _(cat_cols, df, pd, plt):
    for _col in cat_cols:
        if _col != "Churn":
            proportions = pd.crosstab(
                df[_col],
                df["Churn"],
                normalize="index"
            )[["Yes", "No"]]

            proportions.plot(
                kind="bar",
                stacked=True,
                figsize=(8, 4)
            )

            plt.xticks(rotation=45)
            plt.show()
    return


@app.cell
def _(df, num_cols, plt, sns):
    corr = df[num_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm"
    )

    plt.show()
    return


@app.cell
def _(df, num_cols, plt, sns):
    for _col in num_cols:
        plt.figure(figsize=(8, 4))

        sns.boxplot(
            data=df,
            x="Churn",
            y=_col
        )
        plt.title(f"Распределение {_col}")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Итог EDA

    - Датасет содержит 7043 клиента
    - Оток составляет 26,5%
    - Столбец customerID является ID и не должен использоваться для обучения модели
    - В столбце TotalCharges есть пропуски из-за новых клиентов с tenure = 0
    - Клиенты с помесячной оплатой уходят чаще
    - Клиенты с более высокой месячной оплатной уходят чаще
    - Есть дисбаланс классов
    """)
    return


if __name__ == "__main__":
    app.run()

def AttributeRatio(df, base1, base2, target):
    if target not in df.columns:
        df[target] = df[base1] / df[base2] - 1
    return df

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(cat_cols, num_cols, scale_num):
    transformers = []

    if cat_cols:
        # transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if num_cols:
        if scale_num:
            # transformers.append(("num", StandardScaler(), num_cols))
            num_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num_scaled", num_pipe, num_cols))
        else:
            # transformers.append(("num", "passthrough", num_cols))
            num_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ])
            transformers.append(("num_unscaled", num_pipe, num_cols))

    return ColumnTransformer(transformers)
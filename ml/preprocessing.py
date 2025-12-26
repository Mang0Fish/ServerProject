from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(cat_cols, num_cols, scale_num):
    transformers = []

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if num_cols:
        if scale_num:
            transformers.append(("num", StandardScaler(), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

    return ColumnTransformer(transformers)
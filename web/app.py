from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from flask import Flask, request, render_template
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

MODEL_DIR = Path(__file__).resolve().parent.parent / "model_export"
_models = None


def load_models():
    global _models
    if _models is not None:
        return _models
    if not MODEL_DIR.exists():
        return None
    meta = joblib.load(MODEL_DIR / "meta.joblib")
    lgb_booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODEL_DIR / "catboost.cbm"))
    ridge_model = meta.get("model_ridge")
    sgd_model = meta.get("model_sgd")
    class_weight_stack = np.array(meta.get("class_weight_stack", [1.0, 1.0, 1.0]))
    _models = {
        "meta_learner": meta["meta_learner"],
        "label_encoder": meta["label_encoder"],
        "class_weight_stack": class_weight_stack,
        "feature_columns": meta["feature_columns"],
        "wave_cols": meta["wave_cols"],
        "savgol_window": meta.get("savgol_window", 11),
        "savgol_poly": meta.get("savgol_poly", 3),
        "lgb_booster": lgb_booster,
        "cat_model": cat_model,
        "ridge_model": ridge_model,
        "sgd_model": sgd_model,
    }
    return _models


REGIONS = [
    "cerebellum_left",
    "cerebellum_right",
    "cortex",
    "cortex_left",
    "cortex_right",
    "striatum_left",
    "striatum_right",
]


def parse_spectrum_file(stream):
    df = pd.read_csv(
        stream,
        sep=r"\s+",
        skiprows=0,
        header=0,
        dtype=np.float64,
        on_bad_lines="skip",
    )
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["Wave", "Intensity"]
    else:
        raise ValueError("Нужны две колонки: Wave и Intensity")
    df = df.dropna()
    df = df.sort_values("Wave", ascending=False).reset_index(drop=True)
    return df["Wave"].values, df["Intensity"].values


def preprocess_spectrum(intensities, wave_cols, savgol_window, savgol_poly):
    X = np.array(intensities, dtype=np.float64).reshape(1, -1)
    baseline = X.min(axis=1, keepdims=True)
    X = X - baseline
    X = np.maximum(X, 0.0)
    try:
        from scipy.signal import savgol_filter
        X = savgol_filter(X, window_length=savgol_window, polyorder=savgol_poly, axis=1)
    except Exception:
        pass
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return X.flatten()


def predict_from_spectrum(wave_upload, intensity_upload, region: str, center_1500: bool, m):
    feature_columns = m["feature_columns"]
    wave_cols = m["wave_cols"]
    wave_model = np.array([float(c.replace("wave_", "")) for c in wave_cols])
    wave_asc = wave_model[::-1]
    intensity_interp = np.interp(wave_asc, wave_upload[::-1], intensity_upload[::-1])
    intensity_interp = intensity_interp[::-1]
    X_spectrum = preprocess_spectrum(
        intensity_interp,
        wave_cols,
        m["savgol_window"],
        m["savgol_poly"],
    )
    row = {}
    for col in feature_columns:
        if col == "center_1500":
            row[col] = 1 if center_1500 else 0
        elif col.startswith("region_"):
            row[col] = 1 if col == f"region_{region}" else 0
        elif col in wave_cols:
            row[col] = X_spectrum[wave_cols.index(col)]
        else:
            row[col] = 0
    X_row = pd.DataFrame([row])[feature_columns]

    p_lgbm = m["lgb_booster"].predict(X_row)
    p_lgbm = np.atleast_2d(p_lgbm)
    p_cat = m["cat_model"].predict_proba(X_row)
    stack_list = [p_lgbm, p_cat]
    if m.get("ridge_model") is not None:
        r = m["ridge_model"]
        d = r.decision_function(X_row)
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        stack_list.append(e / e.sum(axis=1, keepdims=True))
    if m.get("sgd_model") is not None:
        stack_list.append(m["sgd_model"].predict_proba(X_row))
    stack = np.hstack(stack_list)
    meta_proba = m["meta_learner"].predict_proba(stack)[0]
    w = m.get("class_weight_stack", np.ones(3))
    pred_num = int(np.argmax(meta_proba * w))
    class_name = m["label_encoder"].inverse_transform([pred_num])[0]
    return class_name, meta_proba, list(m["label_encoder"].classes_)


@app.route("/", methods=["GET", "POST"])
def index():
    err = None
    result = None
    if request.method == "POST":
        m = load_models()
        if m is None:
            err = "Модели не найдены. Запустите ячейку экспорта в ноутбуке (папка model_export/ в корне проекта)."
        else:
            region = request.form.get("region")
            center_1500 = request.form.get("center") == "1500"
            f = request.files.get("file")
            if not f or f.filename == "":
                err = "Выберите файл с данными (Wave, Intensity)."
            elif region not in REGIONS:
                err = "Выберите регион мозга."
            else:
                try:
                    wave_up, intensity_up = parse_spectrum_file(f.stream)
                    if len(wave_up) < 10:
                        err = "В файле слишком мало точек. Нужен спектр с колонками Wave и Intensity."
                    else:
                        class_name, proba, classes = predict_from_spectrum(
                            wave_up, intensity_up, region, center_1500, m
                        )
                        result = {
                            "class": class_name,
                            "proba": dict(zip(classes, proba.tolist())),
                        }
                except Exception as e:
                    err = f"Ошибка при разборе файла или предсказании: {e}"
    return render_template("index.html", regions=REGIONS, error=err, result=result)


@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return render_template(
        "index.html",
        regions=REGIONS,
        error="Файл слишком большой (макс. 64 МБ). Сожмите файл или загрузите спектр с меньшим числом точек.",
        result=None,
    ), 413


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)

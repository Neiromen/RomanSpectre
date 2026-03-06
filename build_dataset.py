
from pathlib import Path
import pandas as pd
import numpy as np


def parse_region_from_filename(filename: str, class_name: str) -> str:
    """
    Извлекает регион из имени файла: cortex, cortex_left, cortex_right,
    striatum_left, striatum_right, cerebellum_left, cerebellum_right.
    """
    # Паттерн: {region}_{class}_... — всё до _endo, _control или _exo
    for c in ("endo", "control", "exo"):
        suffix = f"_{c}_"
        if suffix in filename:
            return filename.split(suffix)[0]
    return "unknown"


def load_spectrum_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=1,
        names=["X", "Y", "Wave", "Intensity"],
        dtype={"X": np.float64, "Y": np.float64, "Wave": np.float64, "Intensity": np.float64},
    )
    return df


def spectra_to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort_values(["X", "Y", "Wave"]).reset_index(drop=True)

    wave_numbers = df["Wave"].unique()
    wave_numbers = np.sort(wave_numbers)[::-1]

    groups = df.groupby(["X", "Y"], sort=False)

    spectra_list = []
    for (x, y), grp in groups:
        intensities = grp.sort_values("Wave", ascending=False)["Intensity"].values
        spectra_list.append(intensities)

    spectra_matrix = np.vstack(spectra_list)
    return spectra_matrix, wave_numbers


def build_dataset(
    data_dir: Path, output_path: Path, max_files_per_class: int | None = None
) -> None:
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    classes = ["endo", "control", "exo"]
    all_spectra = []
    all_labels = []
    all_meta = []
    wave_cols = None
    expected_n_waves = None

    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Пропуск: {class_dir} не найден")
            continue

        txt_files = list(class_dir.rglob("*.txt"))
        if max_files_per_class:
            txt_files = txt_files[:max_files_per_class]
        print(f"{class_name}: {len(txt_files)} файлов")

        for i, fp in enumerate(txt_files):
            if (i + 1) % 20 == 0:
                print(f"  обработано {i + 1}/{len(txt_files)}")
            try:
                df = load_spectrum_file(fp)
                spectra, waves = spectra_to_matrix(df)

                n_waves = spectra.shape[1]
                if wave_cols is None:
                    wave_cols = [f"wave_{w:.2f}" for w in waves]
                    expected_n_waves = n_waves
                elif n_waves != expected_n_waves:
                    print(f"  Пропуск {fp.name}: {n_waves} волн (ожидается {expected_n_waves})")
                    continue

                n = spectra.shape[0]
                all_spectra.append(spectra)
                all_labels.extend([class_name] * n)

                region = parse_region_from_filename(fp.name, class_name)
                groups = df.groupby(["X", "Y"], sort=False)
                for (x, y), _ in groups:
                    all_meta.append({"file": str(fp.name), "region": region, "X": x, "Y": y})

            except Exception as e:
                print(f"Ошибка при чтении {fp}: {e}")

    if not all_spectra:
        raise ValueError("Не найдено ни одного файла данных")

    X = np.vstack(all_spectra)
    y = np.array(all_labels)
    meta = pd.DataFrame(all_meta)

    df_out = pd.DataFrame(X, columns=wave_cols)
    df_out.insert(0, "class", y)
    df_out.insert(1, "region", meta["region"].values)
    df_out["source_file"] = meta["file"].values
    df_out["X"] = meta["X"].values
    df_out["Y"] = meta["Y"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nДатасет сохранён: {output_path}")
    print(f"Размер: {df_out.shape[0]} образцов, {df_out.shape[1]-5} признаков (волновые числа)")
    print(f"Классы: {pd.Series(y).value_counts().to_dict()}")


if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).parent / "data"
    output_path = Path(__file__).parent / "dataset_combined.csv"
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_dataset(data_dir, output_path, max_files_per_class=max_files)

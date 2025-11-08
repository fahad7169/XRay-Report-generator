import os
import random
import shutil
import string

import pandas as pd

DEMO_DIR = "demo_samples"
DEMO_METADATA = os.path.join(DEMO_DIR, "demo_metadata.csv")
SOURCE_CSV = "Final_CV_Data.csv"
NUM_PATIENTS = 20


def main():
    if not os.path.exists(SOURCE_CSV):
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)
    if "Person_id" not in df.columns:
        raise ValueError("CSV must contain 'Person_id' column")

    # Pick first NUM_PATIENTS unique patients (preserve order)
    unique_patients = (
        df.drop_duplicates(subset=["Person_id"]).head(NUM_PATIENTS)
    )

    os.makedirs(DEMO_DIR, exist_ok=True)

    demo_rows = []
    used_names = set()

    def random_name() -> str:
        while True:
            suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            name = f"patient_{suffix}"
            if name not in used_names:
                used_names.add(name)
                return name

    for _, row in unique_patients.iterrows():
        person_id = str(row["Person_id"])
        img1 = str(row["Image1"])
        img2 = str(row["Image2"])
        report = str(row.get("Report", ""))

        # Destination subdirectory per patient
        demo_name = random_name()
        dest_dir = os.path.join(DEMO_DIR, demo_name)
        os.makedirs(dest_dir, exist_ok=True)

        def copy_image(src_path: str) -> str:
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Image not found: {src_path}")
            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(src_path, dest_path)
            return dest_path

        img1_copy = copy_image(img1)
        img2_copy = copy_image(img2)

        demo_rows.append(
            {
                "Person_id": person_id,
                "Demo_Name": demo_name,
                "Image1": img1_copy,
                "Image2": img2_copy,
                "Original_Image1": img1,
                "Original_Image2": img2,
                "Report": report,
            }
        )

    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(DEMO_METADATA, index=False)
    print(f"Created {len(demo_rows)} demo samples in {DEMO_DIR}")


if __name__ == "__main__":
    main()



import pandas as pd


def cgpa_band(cgpa: float) -> int:
    """
    Bucket CGPA into three bands and return a numeric code:
        0 -> Low      (cgpa < 6.5)
        1 -> Medium   (6.5 <= cgpa < 8.5)
        2 -> High     (cgpa >= 8.5)
    """
    if cgpa < 6.5:
        return 0
    elif cgpa < 8.5:
        return 1
    else:
        return 2


def cgpa_band_label(band: int) -> str:
    """Human readable label for a cgpa_band code."""
    return {0: "Low", 1: "Medium", 2: "High"}.get(band, "Unknown")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a dataframe that already contains the
    raw columns: cgpa, skills, internships, projects.

    New columns added:
        experience_score            = skills + internships + projects
        cgpa_band                    = categorical CGPA bucket (0/1/2)
        cgpa_experience_interaction  = cgpa * experience_score
    """
    df = df.copy()

    df["experience_score"] = df["skills"] + df["internships"] + df["projects"]
    df["cgpa_band"] = df["cgpa"].apply(cgpa_band)
    df["cgpa_experience_interaction"] = df["cgpa"] * df["experience_score"]

    return df


# The final list of feature columns fed into the model.
FEATURE_COLUMNS = [
    "cgpa",
    "skills",
    "internships",
    "projects",
    "experience_score",
    "cgpa_band",
    "cgpa_experience_interaction",
]

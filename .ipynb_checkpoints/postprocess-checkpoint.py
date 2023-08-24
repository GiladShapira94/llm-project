import mlrun
import pandas as pd


def _clean_issue(s: str) -> str:
    """clean issue column: remove "", 1. and ():"""
    s = s.translate({ord(c): None for c in '"()'})
    if s.startswith("1. "):
        s = s.replace("1. ", "")
    return s


def _extract_is_fixed(s: str) -> str:
    """extract is_fixed from content (Yes / No)"""
    s = s.casefold()
    if "not explicitly" in s:
        return "Unknown"
    if any(sub in s for sub in ["yes", "was fixed"]):
        return "Yes"
    if any(sub in s for sub in ["no", "was not fixed"]):
        return "No"
    return "Unknown"


def _extract_tone(s: str) -> str:
    """extract tone from content (Positive / Natural / Negative)"""
    s = s.casefold()
    if "positive" in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    return "Natural"


def postprocess(
    transcript_dataset: mlrun.DataItem,
    qa_dataset: mlrun.DataItem,
):
    # Convert to pd.DataFrame:
    transcript_df = transcript_dataset.as_df()
    qa_df = qa_dataset.as_df()

    # Left join:
    qa_df.rename(columns={"text_file": "transcription_file"}, inplace=True)
    df = pd.merge(transcript_df, qa_df, how="left", on="transcription_file")
    df.dropna(inplace=True)
    # Clean content and extract short answers:
    for column, apply_function in [
        ("Issue", _clean_issue),
        ("is_fixed", _extract_is_fixed),
        ("customer_tone", _extract_tone),
        ("agent_tone", _extract_tone),
    ]:
        df[column] = df[column].apply(lambda s: apply_function(s))
    return df
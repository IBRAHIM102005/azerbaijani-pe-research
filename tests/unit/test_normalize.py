import unicodedata

from src.data.normalize import normalize_text


def test_normalization_preserves_azerbaijani_letters_and_case():
    text = " ƏIİıÖöÜüŞşÇçĞğ  "
    result = normalize_text(text)
    assert result.text == "ƏIİıÖöÜüŞşÇçĞğ"
    assert result.text != result.text.lower()


def test_normalization_uses_nfc_and_preserves_paragraphs():
    decomposed = "s\u0327"  # ş as a decomposed sequence
    result = normalize_text(f"  {decomposed}\r\n\r\n\r\nİki\t  söz  ")
    assert result.text == "ş\n\nİki söz"
    assert unicodedata.is_normalized("NFC", result.text)
    assert result.changed_nfc
    assert result.changed_newlines


def test_control_characters_are_removed_but_newlines_remain():
    result = normalize_text("bir\x00iki\nüç")
    assert result.text == "biriki\nüç"
    assert result.removed_control_characters == 1

from src.data.quality import quality_flags


def test_quality_flags_detect_encoding_and_control_issues():
    flags = quality_flags("Azərbaycan\u00a0mətnində\x00problem\ufffd")

    assert "unusual_whitespace" in flags
    assert "null_byte" in flags
    assert "control_character" in flags
    assert "unicode_replacement" in flags


def test_quality_flags_detect_markup_url_and_repetition():
    assert "html_xml_like_markup" in quality_flags("<p>Azərbaycan</p>")
    assert "url_heavy" in quality_flags("https://x.az")
    assert "repeated_character" in quality_flags("aaaaaa")


def test_quality_flags_detect_extremely_long_documents():
    assert "extremely_long" in quality_flags("a" * 100_000)

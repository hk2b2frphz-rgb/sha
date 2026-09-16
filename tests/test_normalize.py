from eval.normalize import hira_to_kata, kata_to_hira, normalize_reading, normalize_unicode


def test_katakana_to_hiragana_full_range_and_long_mark():
    assert kata_to_hira("ァアヴヵヶー") == "ぁあゔゕゖー"


def test_half_width_kana_is_folded_before_conversion():
    assert kata_to_hira("ｶﾞｯﾂﾎﾟｰｽﾞ") == "がっつぽーず"


def test_hiragana_to_katakana():
    assert hira_to_kata("ぁあゔゕゖー") == "ァアヴヵヶー"


def test_symbol_and_space_normalization():
    assert normalize_unicode("A〜B—C") == "A～B-C"
    assert normalize_reading(" カ タ ") == "かた"


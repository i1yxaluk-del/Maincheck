from secondary_gec import merge_safe


class FakeMorph:
    KNOWN = {"наряда", "нарядов", "факт", "факты", "рабочего", "времени", "часов"}

    def word_is_known(self, word: str) -> bool:
        return word in self.KNOWN


def test_punctuation_insertion_is_accepted():
    corrected, edits = merge_safe(
        "Кроме того имеются факты.",
        "Кроме того, имеются факты.",
        FakeMorph(),
    )
    assert corrected == "Кроме того, имеются факты."
    assert edits


def test_valid_inflection_is_not_overridden():
    corrected, edits = merge_safe(
        "после ночных нарядов",
        "после ночных наряда",
        FakeMorph(),
    )
    assert corrected == "после ночных нарядов"
    assert edits == []


def test_secondary_flood_is_rejected():
    primary = "Альфа бета гамма дельта эпсилон зета."
    secondary = "Альфа, бета, гамма, дельта, эпсилон, зета."
    corrected, edits = merge_safe(primary, secondary, FakeMorph(), max_edits=4)
    assert corrected == primary
    assert edits == []


def test_secondary_lexical_hallucination_is_rejected():
    primary = (
        "Работником Амелиным И.Э., осуществляющим трудовую деятельность удаленно, "
        "нарушены условия трудового договора, заключенного с Центром, в части "
        "продолжительности рабочего времени."
    )
    secondary = (
        "Работником Амелиным И.Э., осуществляющим трудовую деятельность удаленно, "
        "нарушены условия трудового договора, заключенного с Центром, в части "
        "продолжительности 8-ми часов рабочегочасовымирабочего времени."
    )
    corrected, edits = merge_safe(primary, secondary, FakeMorph(), max_edits=4)
    assert corrected == primary
    assert edits == []


def test_single_unknown_spelling_typo_can_be_accepted():
    corrected, edits = merge_safe(
        "Это очепятка.",
        "Это опечатка.",
        FakeMorph(),
    )
    assert corrected == "Это опечатка."
    assert edits

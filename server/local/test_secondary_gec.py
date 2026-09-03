from secondary_gec import merge_safe


class FakeMorph:
    KNOWN = {"наряда", "нарядов", "факт", "факты"}

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

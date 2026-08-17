"""Tests for query resolution against the real ImageNet-1k vocabulary.

These are the tests that would have caught the original defect. The old bucket
map sent ``African elephant`` to *Insect* via ``ant``; the first test here fails
if word-boundary matching is ever weakened back to substring matching.
"""

from __future__ import annotations

import pytest

from vie.species import GENERIC_TERMS, matches, normalise, resolve

# A representative slice of the real ImageNet-1k names, including every class
# the original mapping got wrong plus the non-animals it wrongly absorbed.
CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "cock", "hen", "ostrich", "brambling", "goldfinch", "junco", "robin",
    "bulbul", "jay", "magpie", "chickadee", "kite", "bald eagle", "vulture",
    "great grey owl", "European fire salamander", "bullfrog", "tree frog",
    "loggerhead", "leatherback turtle", "banded gecko", "green iguana",
    "African chameleon", "Komodo dragon", "African crocodile", "boa constrictor",
    "rock python", "Indian cobra", "green mamba", "sea snake", "sidewinder",
    "trilobite", "harvestman", "scorpion", "black and gold garden spider",
    "tick", "centipede", "black grouse", "ptarmigan", "peacock", "quail",
    "partridge", "macaw", "toucan", "drake", "goose", "black swan",
    "tusker", "echidna", "platypus", "wallaby", "koala", "wombat",
    "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode",
    "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus",
    "Dungeness crab", "rock crab", "fiddler crab", "king crab", "American lobster",
    "crayfish", "hermit crab", "isopod", "white stork", "black stork",
    "spoonbill", "flamingo", "little blue heron", "American egret", "bittern",
    "crane", "limpkin", "European gallinule", "American coot", "bustard",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale",
    "dugong", "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog",
    "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier",
    "Rhodesian ridgeback", "Afghan hound", "basset", "beagle", "bloodhound",
    "Bedlington terrier", "Border terrier", "Kerry blue terrier",
    "golden retriever", "Labrador retriever", "German shepherd",
    "timber wolf", "white wolf", "red wolf", "coyote", "dingo", "dhole",
    "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
    "grey fox", "tabby, tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard", "jaguar",
    "lion", "tiger", "cheetah", "brown bear", "American black bear",
    "ice bear", "sloth bear", "mongoose", "meerkat", "tiger beetle",
    "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
    "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking stick",
    "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
    "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
    "wood rabbit", "hare", "Angora", "hamster", "porcupine", "fox squirrel",
    "marmot", "beaver", "guinea pig", "sorrel", "zebra", "hog", "wild boar",
    "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram",
    "bighorn", "ibex", "hartebeest", "impala", "gazelle", "Arabian camel",
    "llama", "weasel", "mink", "polecat", "black-footed ferret", "otter",
    "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas",
    "baboon", "macaque", "langur", "colobus", "proboscis monkey", "marmoset",
    "capuchin", "howler monkey", "titi", "spider monkey", "squirrel monkey",
    "Madagascar cat", "indri", "Indian elephant", "African elephant",
    "lesser panda", "giant panda", "barracouta", "eel", "coho", "rock beauty",
    "anemone fish", "sturgeon", "gar", "lionfish", "puffer",
    # non-animals the original mapping wrongly absorbed
    "mailbox", "bathtub", "bath towel", "beer bottle", "beer glass",
    "mixing bowl", "soup bowl", "cocktail shaker", "steel drum", "car wheel",
    "paddlewheel", "garbage truck", "restaurant", "lipstick", "matchstick",
    "crate", "pirate", "refrigerator", "cowboy hat", "cowboy boot",
    "bullet train", "bulletproof vest", "hotdog", "dogsled", "mousetrap",
    "mouse", "hen-of-the-woods", "jack-o'-lantern",
]


# ── the defect that started all of this ────────────────────────────────


@pytest.mark.parametrize(
    "term",
    ["ant", "boa", "ox", "bat", "bee", "owl", "cock", "rat", "eel", "gar"],
)
def test_short_terms_do_not_match_inside_longer_words(term: str) -> None:
    """The exact failure mode of the original: substring, not word, matching.

    ``"ant" in "African elephant"`` is True. ``\\bant\\b`` is not.
    """
    culprits = {
        "ant": "African elephant", "boa": "wild boar",
        "ox": "mailbox", "bat": "bathtub", "bee": "beer bottle",
        "owl": "mixing bowl", "cock": "cockroach", "rat": "crate",
        "eel": "steel drum", "gar": "garbage truck",
    }
    victim = culprits[term]
    assert term in victim.lower(), "precondition: substring matching would hit"
    assert not matches(term, victim), f"'{term}' must not match '{victim}'"


def test_elephant_resolves_to_elephants_not_insects() -> None:
    r = resolve("elephant", CLASSES)
    names = {CLASSES[i] for i in r.class_ids}
    assert "African elephant" in names
    assert "Indian elephant" in names
    assert "ant" not in names


def test_searching_ant_does_not_return_elephants() -> None:
    names = {CLASSES[i] for i in resolve("ant", CLASSES).class_ids}
    assert names == {"ant"}


def test_no_non_animal_leaks_into_an_animal_query() -> None:
    """`mailbox`, `bathtub`, `beer bottle` were all classified as animals."""
    for query in ("ox", "bat", "bee", "owl", "rat", "eel"):
        names = {CLASSES[i] for i in resolve(query, CLASSES).class_ids}
        for junk in ("mailbox", "bathtub", "beer bottle", "mixing bowl",
                     "crate", "steel drum", "refrigerator", "cowboy hat"):
            assert junk not in names, f"'{query}' wrongly matched '{junk}'"


def test_boar_is_not_a_snake() -> None:
    assert not matches("boa", "wild boar")
    assert matches("boa", "boa constrictor")


# ── ordinary resolution ────────────────────────────────────────────────


def test_exact_species_resolves() -> None:
    r = resolve("zebra", CLASSES)
    assert r.in_vocabulary and r.via == "exact"
    assert {CLASSES[i] for i in r.class_ids} == {"zebra"}


def test_synonyms_after_a_comma_are_searched() -> None:
    """ImageNet names carry synonyms: 'tabby, tabby cat'."""
    assert matches("tabby cat", "tabby, tabby cat")


def test_lion_does_not_return_sea_lion() -> None:
    """'lion' is a whole word inside 'sea lion', so word boundaries alone are not
    enough — an exact name match must win."""
    names = {CLASSES[i] for i in resolve("lion", CLASSES).class_ids}
    assert names == {"lion"}


def test_sea_lion_still_resolves_to_itself() -> None:
    names = {CLASSES[i] for i in resolve("sea lion", CLASSES).class_ids}
    assert names == {"sea lion"}


def test_generic_dog_expands_to_breeds() -> None:
    r = resolve("dog", CLASSES)
    assert r.via == "generic"
    names = {CLASSES[i] for i in r.class_ids}
    assert "golden retriever" in names
    assert "Bedlington terrier" in names
    assert "hotdog" not in names and "dogsled" not in names


def test_generic_cat_does_not_include_big_cats() -> None:
    names = {CLASSES[i] for i in resolve("cat", CLASSES).class_ids}
    assert "Egyptian cat" in names
    assert "lion" not in names and "tiger" not in names


def test_big_cat_is_its_own_query() -> None:
    names = {CLASSES[i] for i in resolve("big cat", CLASSES).class_ids}
    assert {"lion", "tiger", "cheetah"} <= names


def test_out_of_vocabulary_query_is_unresolved() -> None:
    """This is the signal to fall back to CLIP — not a failure."""
    r = resolve("red panda", CLASSES)
    assert not r.in_vocabulary and r.via == "unresolved"


def test_quokka_is_unresolved() -> None:
    assert not resolve("quokka", CLASSES).in_vocabulary


# ── plurals ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("plural", "expected"),
    [("wolves", "wolf"), ("geese", "goose"), ("mice", "mouse"),
     ("zebras", "zebra"), ("foxes", "fox"), ("butterflies", "butterfly"),
     ("sheep", "sheep"), ("octopi", "octopus")],
)
def test_plural_normalisation(plural: str, expected: str) -> None:
    """A naive trailing-'s' stemmer turned 'wolves' into 'wolve'."""
    assert normalise(plural) == expected


def test_plural_query_finds_the_species() -> None:
    names = {CLASSES[i] for i in resolve("wolves", CLASSES).class_ids}
    assert "timber wolf" in names


def test_zebras_resolves_like_zebra() -> None:
    assert resolve("zebras", CLASSES).class_ids == resolve("zebra", CLASSES).class_ids


# ── the generic table must stay honest ─────────────────────────────────


def test_every_generic_term_resolves_to_real_classes() -> None:
    """Guards against the table drifting into fiction, which is how the
    original keyword lists filled up with dead entries."""
    unusable = []
    for generic, terms in GENERIC_TERMS.items():
        hits = [t for t in terms if any(matches(t, c) for c in CLASSES)]
        if not hits:
            unusable.append(generic)
    assert not unusable, f"generic terms matching nothing: {unusable}"


def test_case_and_whitespace_are_ignored() -> None:
    assert resolve("  ZEBRA  ", CLASSES).class_ids == resolve("zebra", CLASSES).class_ids

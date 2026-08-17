"""Species vocabulary: resolving a user query against ImageNet-1k class names.

Why this exists
---------------
The original collapsed ImageNet's 1000 class names into 8 coarse buckets using
unanchored substring tests. That was the defect — ``"African elephant"``
contains ``ant``, so it landed in *Insect*.

The classifier itself was never wrong. MobileNetV3 reported ``African
elephant`` correctly; the bucket layer destroyed it. So the fix is not to
replace the classifier, it is to **delete the bucket layer** and match the query
against the real class names with word boundaries.

Doing that keeps what ImageNet is genuinely good at — 398 animal classes, ~120
of them dog breeds, from a supervised model that saw those exact labels — while
removing the layer that broke it.

Design
------
:func:`resolve` answers one question: *which ImageNet classes does this query
mean?* If it can answer, the query is served from precomputed predictions with
**no inference at all**. If it cannot (``red panda`` is not an ImageNet class),
the caller falls back to CLIP, which is ~373x the compute per crop and is
therefore reserved for the case that actually needs it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

#: Generic terms a photo editor actually types, expanded to the specific
#: ImageNet names that belong to them. This is query expansion, not a
#: reclassification of the label space: every entry is matched against real
#: class names with word boundaries, and `tests/test_species.py` asserts that
#: each one resolves to at least one genuine class.
GENERIC_TERMS: dict[str, tuple[str, ...]] = {
    "dog": ("terrier", "retriever", "hound", "shepherd dog", "spaniel", "collie",
            "poodle", "pug", "bulldog", "corgi", "husky", "malamute", "chihuahua",
            "beagle", "dalmatian", "rottweiler", "schnauzer", "setter", "pointer",
            "sheepdog", "mastiff", "pinscher", "griffon", "papillon", "pekinese",
            "chow", "keeshond", "samoyed", "pomeranian", "basenji", "boxer"),
    "cat": ("tabby", "Persian cat", "Siamese cat", "Egyptian cat"),
    "big cat": ("lion", "tiger", "leopard", "snow leopard", "jaguar", "cheetah",
                "cougar", "lynx"),
    "bird": ("cock", "hen", "ostrich", "brambling", "goldfinch", "house finch",
             "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie",
             "chickadee", "water ouzel", "kite", "bald eagle", "vulture",
             "great grey owl", "peacock", "quail", "partridge", "macaw",
             "toucan", "drake", "goose", "black swan", "pelican", "king penguin",
             "albatross", "flamingo", "crane bird", "stork"),
    "horse": ("sorrel", "zebra"),
    "bear": ("brown bear", "American black bear", "ice bear", "sloth bear"),
    "monkey": ("macaque", "langur", "baboon", "guenon", "colobus", "marmoset",
               "capuchin", "howler monkey", "titi", "spider monkey",
               "squirrel monkey"),
    "ape": ("gorilla", "chimpanzee", "orangutan", "gibbon", "siamang"),
    "snake": ("snake", "boa constrictor", "rock python", "sidewinder",
              "night snake", "vine snake", "green mamba"),
    "fish": ("tench", "goldfish", "great white shark", "tiger shark",
             "hammerhead", "electric ray", "stingray", "barracouta", "eel",
             "coho", "rock beauty", "anemone fish", "sturgeon", "gar",
             "lionfish", "puffer"),
    "insect": ("bee", "ant", "grasshopper", "cricket", "walking stick",
               "cockroach", "mantis", "cicada", "leafhopper", "lacewing",
               "dragonfly", "damselfly", "admiral", "ringlet", "monarch",
               "cabbage butterfly", "sulphur butterfly", "lycaenid",
               "ladybug", "ground beetle", "weevil", "fly"),
    "elephant": ("African elephant", "Indian elephant", "tusker"),
}

#: Cheap plural handling. English is irregular; a stemmer that only strips a
#: trailing "s" turns "wolves" into "wolve" and matches nothing, so the
#: irregular cases that matter for animals are listed explicitly.
IRREGULAR_PLURALS: dict[str, str] = {
    "wolves": "wolf", "geese": "goose", "mice": "mouse", "oxen": "ox",
    "sheep": "sheep", "deer": "deer", "fish": "fish", "fishes": "fish",
    "leaves": "leaf", "calves": "calf", "octopuses": "octopus",
    "octopi": "octopus", "puppies": "puppy", "butterflies": "butterfly",
    "ponies": "pony", "foxes": "fox", "finches": "finch", "ostriches": "ostrich",
}


def normalise(query: str) -> str:
    """Lower-case, trim, and singularise a query."""
    q = " ".join(query.lower().split())
    if q in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[q]
    if q.endswith("ies") and len(q) > 4:
        return q[:-3] + "y"
    if q.endswith("es") and len(q) > 4 and q[-3] in "sxzh":
        return q[:-2]
    if q.endswith("s") and not q.endswith("ss") and len(q) > 3:
        return q[:-1]
    return q


@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern[str]:
    """Word-boundary matcher.

    This is the whole fix for the original defect. ``re.search(r"\\bant\\b", ...)``
    does not match ``"African elephant"``, whereas ``"ant" in ...`` does.
    """
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def matches(term: str, class_name: str) -> bool:
    """True if ``term`` appears in ``class_name`` as a whole word."""
    # ImageNet names carry comma-separated synonyms: "tabby, tabby cat".
    return any(_pattern(term).search(part.strip()) for part in class_name.split(","))


@dataclass(frozen=True)
class Resolution:
    """What a query resolved to."""

    query: str
    normalised: str
    class_ids: tuple[int, ...]
    via: str          # "exact" | "generic" | "unresolved"

    @property
    def in_vocabulary(self) -> bool:
        return bool(self.class_ids)


def _exact_name_match(term: str, class_name: str) -> bool:
    """True if ``term`` *is* one of the class's names, not merely inside one.

    This precedence matters. ``lion`` appears as a whole word inside
    ``sea lion``, so word-boundary matching alone would return a pinniped for a
    query about big cats. Checking for a full-name match first resolves the
    ambiguity the way a user means it.
    """
    return any(term == part.strip().lower() for part in class_name.split(","))


def resolve(query: str, class_names: list[str]) -> Resolution:
    """Map a query onto ImageNet class indices.

    Precedence: exact name → generic expansion → partial word match. An empty
    resolution means the query is outside ImageNet's vocabulary, which is the
    signal to fall back to CLIP — not a failure.
    """
    norm = normalise(query)

    # 1. the query names a class outright: "lion", "zebra", "ant"
    exact = tuple(i for i, name in enumerate(class_names) if _exact_name_match(norm, name))
    if exact:
        return Resolution(query, norm, exact, "exact")

    # 2. a generic term a photo editor would actually type: "dog", "bird"
    if norm in GENERIC_TERMS:
        ids: set[int] = set()
        for term in GENERIC_TERMS[norm]:
            ids.update(i for i, name in enumerate(class_names) if matches(term, name))
        # include direct hits too: "African hunting dog" is a dog
        ids.update(i for i, name in enumerate(class_names) if matches(norm, name))
        if ids:
            return Resolution(query, norm, tuple(sorted(ids)), "generic")

    # 3. the query appears as a whole word inside class names:
    #    "terrier" -> every terrier breed
    partial = tuple(i for i, name in enumerate(class_names) if matches(norm, name))
    if partial:
        return Resolution(query, norm, partial, "partial")

    # 4. outside ImageNet — CLIP handles it
    return Resolution(query, norm, (), "unresolved")

"""From per-frame scores to "what is this video about".

Everything here is plain arithmetic over numbers the models already produced, so
all of it is testable without a model, and none of it invents anything: every
sentence in a summary traces back to a measured score or a detection.

Duration, not frame count, is what is weighed. A five-second title card with six
sharp keyframes must not outvote a two-minute interview that received three.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vie.video.people import Interval, Person
from vie.video.shots import Shot


@dataclass(frozen=True)
class LabelScore:
    group: str
    label: str
    score: float       # duration-weighted mean probability over the video
    coverage: float    # fraction of the running time where this label is present
    peak: float        # its strongest single shot
    intervals: tuple[Interval, ...] = ()


@dataclass
class ShotLabels:
    shot: int
    labels: dict[str, tuple[str, float]] = field(default_factory=dict)  # group -> (label, score)
    caption: str | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def label_prompts(
    taxonomy: dict[str, list[str]], template: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """Flatten the taxonomy into ``(group, label)`` pairs and their prompts."""
    pairs = [(group, label) for group, labels in taxonomy.items() for label in labels]
    return pairs, [template.format(label=label) for _, label in pairs]


def shot_scores(
    probabilities: np.ndarray, shot_of_keyframe: Sequence[int], shot_count: int
) -> np.ndarray:
    """Average keyframe probabilities within each shot.

    Returns ``(shots, labels)``. A shot with no keyframe (possible only when the
    budget is exhausted) scores zero rather than borrowing from a neighbour.
    """
    labels = probabilities.shape[1] if probabilities.size else 0
    scores = np.zeros((shot_count, labels), dtype=np.float32)
    counts = np.zeros(shot_count, dtype=np.int32)
    for row, shot in zip(probabilities, shot_of_keyframe, strict=True):
        scores[shot] += row
        counts[shot] += 1
    present = counts > 0
    scores[present] = scores[present] / counts[present, None]
    return scores


def aggregate_labels(
    per_shot: np.ndarray,
    shots: Sequence[Shot],
    pairs: Sequence[tuple[str, str]],
    threshold: float,
    merge_gap: float = 1.0,
) -> list[LabelScore]:
    """Roll shot-level scores up to the video, keeping what clears the floor."""
    if not len(shots) or not per_shot.size:
        return []

    durations = np.array([max(s.duration, 1e-3) for s in shots], dtype=np.float32)
    total = float(durations.sum())
    results: list[LabelScore] = []

    for column, (group, label) in enumerate(pairs):
        values = per_shot[:, column]
        present = values >= threshold
        coverage = float(durations[present].sum()) / total
        if not present.any():
            continue
        spans: list[Interval] = []
        for shot, is_present in zip(shots, present, strict=True):
            if not is_present:
                continue
            if spans and shot.start - spans[-1].end <= merge_gap:
                spans[-1] = Interval(spans[-1].start, shot.end)
            else:
                spans.append(Interval(shot.start, shot.end))
        results.append(
            LabelScore(
                group=group,
                label=label,
                score=float((values * durations).sum()) / total,
                coverage=coverage,
                peak=float(values.max()),
                intervals=tuple(spans),
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def label_shots(
    per_shot: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    threshold: float,
    shot_count: int,
) -> list[ShotLabels]:
    """The strongest label in each group, per shot."""
    results = [ShotLabels(i) for i in range(shot_count)]
    if not per_shot.size:
        return results
    for row, entry in zip(per_shot, results, strict=True):
        for value, (group, label) in zip(row.tolist(), pairs, strict=True):
            if value >= threshold and value > entry.labels.get(group, ("", 0.0))[1]:
                entry.labels[group] = (label, value)
    return results


def top_by_group(labels: Sequence[LabelScore], per_group: int = 3) -> dict[str, list[LabelScore]]:
    grouped: dict[str, list[LabelScore]] = {}
    for item in labels:
        bucket = grouped.setdefault(item.group, [])
        if len(bucket) < per_group:
            bucket.append(item)
    return grouped


def format_clock(seconds: float) -> str:
    total = int(round(seconds))
    hours, minutes, secs = total // 3600, total % 3600 // 60, total % 60
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _format_length(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return f"{total}-second"
    minutes = round(total / 60)
    return f"{minutes}-minute"


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _fa(number: int | str) -> str:
    return str(number).translate(_FA_DIGITS)


@dataclass(frozen=True)
class SummaryFacts:
    """Everything a sentence is allowed to say."""

    duration: float
    shot_count: int
    labels: Sequence[LabelScore]
    people: Sequence[Person]
    animals: Sequence[str]
    animal_seconds: float
    food_seconds: float
    faces_checked: bool = True
    transcript: Sequence[TranscriptSegment] = ()
    language: str | None = None


def _pace(duration: float, shots: int) -> str:
    average = duration / max(shots, 1)
    if average < 3:
        return "fast-cut"
    if average < 10:
        return "edited"
    if shots <= 2:
        return "single-take"
    return "slow-paced"


_FA_PACE = {
    "fast-cut": "پرتدوین",
    "edited": "تدوین‌شده",
    "single-take": "تک‌برداشت",
    "slow-paced": "آرام",
}


#: Persian for the default taxonomy. A label added to the config with no entry here
#: is shown as written rather than machine-translated into something wrong.
FA_LABELS = {
    "a news broadcast": "پخش خبری",
    "a press conference": "نشست خبری",
    "a political speech": "سخنرانی سیاسی",
    "a sports match": "مسابقه‌ی ورزشی",
    "a music performance": "اجرای موسیقی",
    "a cooking show": "برنامه‌ی آشپزی",
    "a wildlife documentary": "مستند حیات وحش",
    "a nature documentary": "مستند طبیعت",
    "a lecture or classroom": "سخنرانی یا کلاس درس",
    "an interview": "مصاحبه",
    "a video game": "بازی ویدیویی",
    "an animated cartoon": "انیمیشن",
    "a movie scene": "صحنه‌ی فیلم سینمایی",
    "a travel vlog": "ولاگ سفر",
    "a product advertisement": "تبلیغ محصول",
    "a street protest or crowd": "تجمع خیابانی",
    "a wedding or ceremony": "عروسی یا مراسم",
    "a screen recording of software": "ضبط صفحه‌ی نرم‌افزار",
    "indoors in an office": "داخل دفتر کار",
    "indoors in a home": "داخل خانه",
    "in a studio": "استودیو",
    "on a stage": "روی صحنه",
    "in a city street": "خیابان شهر",
    "in a forest": "جنگل",
    "on a beach or by the sea": "ساحل یا کنار دریا",
    "in the mountains": "کوهستان",
    "in a stadium": "ورزشگاه",
    "in a kitchen": "آشپزخانه",
    "in a restaurant": "رستوران",
    "in a classroom": "کلاس درس",
    "in a vehicle": "داخل خودرو",
    "at night": "شب",
    "people talking": "گفت‌وگوی افراد",
    "people walking": "راه رفتن افراد",
    "people running": "دویدن افراد",
    "people dancing": "رقص افراد",
    "people singing or playing instruments": "آواز یا نواختن ساز",
    "people eating": "غذا خوردن افراد",
    "people cooking": "آشپزی افراد",
    "people playing a sport": "ورزش کردن افراد",
    "people fighting": "درگیری افراد",
    "someone driving": "رانندگی",
    "animals moving": "حرکت حیوانات",
    "text and graphics on screen": "متن و گرافیک روی تصویر",
}


def _fa_labels(labels: Sequence[str]) -> str:
    return "، ".join(FA_LABELS.get(label, label) for label in labels)


#: Fewer seconds than this is one stray detection, not something a video contains.
MENTION_SECONDS = 3.0

#: Below this share of the running time, a label is not worth a sentence.
MENTION_COVERAGE = 0.10


def _labelled(items: Sequence[LabelScore], limit: int = 2) -> tuple[list[str], float]:
    """Labels that cover enough of the video, and the share of the strongest."""
    kept = [item for item in items if item.coverage >= MENTION_COVERAGE]
    kept = sorted(kept, key=lambda item: item.coverage, reverse=True)[:limit]
    if not kept:
        return [], 0.0
    # One sentence makes one claim, so a weak label must not borrow a strong one's
    # wording: "often shows walking and cooking" about 42% and 11% says too much.
    lead = kept[0].coverage
    return [item.label for item in kept if item.coverage >= lead * 0.6], lead


def _setting_sentences(labels: list[str], share: float) -> tuple[str, str]:
    joined_en, joined_fa = _join(labels), _fa_labels(labels)
    if share >= 0.5:
        return f"Most of it takes place {joined_en}.", f"بیشتر آن در این فضا می‌گذرد: {joined_fa}."
    if share >= 0.25:
        return (
            f"Much of it takes place {joined_en}.",
            f"بخش زیادی از آن در این فضا می‌گذرد: {joined_fa}.",
        )
    return f"Parts of it take place {joined_en}.", f"بخش‌هایی از آن در این فضا می‌گذرد: {joined_fa}."


def _activity_sentences(labels: list[str], share: float) -> tuple[str, str]:
    joined_en, joined_fa = _join(labels), _fa_labels(labels)
    if share >= 0.5:
        return f"Mostly it shows {joined_en}.", f"بیشتر زمان آن این را نشان می‌دهد: {joined_fa}."
    if share >= 0.25:
        return f"It often shows {joined_en}.", f"اغلب این را نشان می‌دهد: {joined_fa}."
    return f"At times it shows {joined_en}.", f"گاهی این را نشان می‌دهد: {joined_fa}."


def summarise(facts: SummaryFacts) -> dict[str, str]:
    """Compose a factual description, in English and Persian.

    These are templates over measurements, on purpose. A language model would read
    better and could also state things the measurements do not support, so that is
    left as an optional later layer. Three rules keep the sentences honest:

    * how strongly something is claimed follows how much of the video it covers;
    * a label too rare to matter gets no sentence at all;
    * a stage that did not run says nothing: "no faces" would otherwise be a false
      statement about a video nobody checked for faces.
    """
    grouped = top_by_group(facts.labels, per_group=4)
    topics, topic_share = _labelled(grouped.get("topic", []), limit=1)
    settings, setting_share = _labelled(grouped.get("setting", []))
    activities, activity_share = _labelled(grouped.get("activity", []))

    pace = _pace(facts.duration, facts.shot_count)
    shots = facts.shot_count
    en = [f"A {_format_length(facts.duration)}, {pace} video in {shots} shot{'s' * (shots != 1)}."]
    minutes = max(1, round(facts.duration / 60))
    length_fa = (
        f"{_fa(int(round(facts.duration)))} ثانیه‌ای" if facts.duration < 60
        else f"{_fa(minutes)} دقیقه‌ای"
    )
    fa = [f"ویدیویی {length_fa} و {_FA_PACE[pace]} در {_fa(shots)} نما."]

    if topics:
        strength = "mostly resembles" if topic_share >= 0.5 else "most resembles"
        en.append(f"It {strength} {topics[0]}.")
        fa.append(f"بیش از همه به این شبیه است: {_fa_labels(topics)}.")
    if settings:
        sentence_en, sentence_fa = _setting_sentences(settings, setting_share)
        en.append(sentence_en)
        fa.append(sentence_fa)
    if activities:
        sentence_en, sentence_fa = _activity_sentences(activities, activity_share)
        en.append(sentence_en)
        fa.append(sentence_fa)

    if facts.people:
        count = len(facts.people)
        lead = facts.people[0]
        share = lead.screen_time / max(facts.duration, 1e-3) * 100
        noun = "person appears" if count == 1 else "people appear"
        en.append(
            f"{count} distinct {noun}; the most present is on screen for about "
            f"{share:.0f}% of the running time."
        )
        fa.append(
            f"{_fa(count)} چهره‌ی متمایز دیده می‌شود؛ پرحضورترین آن‌ها حدود "
            f"{_fa(f'{share:.0f}')}٪ از زمان ویدیو در تصویر است."
        )
    elif facts.faces_checked:
        en.append("No faces were detected.")
        fa.append("چهره‌ای تشخیص داده نشد.")

    if facts.animals:
        en.append(f"Animals appear ({_join(list(facts.animals)[:3])}).")
        fa.append(f"حیوان دیده می‌شود ({'، '.join(list(facts.animals)[:3])}).")
    elif facts.animal_seconds >= MENTION_SECONDS:
        en.append(f"Animals are visible for about {format_clock(facts.animal_seconds)}.")
        fa.append(f"حیوان حدود {_fa(format_clock(facts.animal_seconds))} در تصویر است.")
    if facts.food_seconds >= MENTION_SECONDS:
        en.append(f"Food is visible for about {format_clock(facts.food_seconds)}.")
        fa.append(f"غذا حدود {_fa(format_clock(facts.food_seconds))} در تصویر است.")

    if facts.transcript:
        words = sum(len(s.text.split()) for s in facts.transcript)
        # Whisper reports an ISO code; a code nobody here can name is simply left out.
        name = {"en": "English", "fa": "Persian", "ar": "Arabic", "fr": "French",
                "de": "German", "es": "Spanish", "tr": "Turkish"}.get(facts.language or "")
        language = f" in {name}" if name else ""
        en.append(f"It contains speech{language} (about {words} words).")
        fa.append(f"گفتار دارد (حدود {_fa(words)} کلمه).")

    return {"en": " ".join(en), "fa": " ".join(fa)}

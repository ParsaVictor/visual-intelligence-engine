"""Video understanding: what is in this video, and where.

A video is treated as a structured image archive with a clock. It is split into
shots, a few sharp frames are chosen from each, and those frames go through the
same models as the image index. Everything temporal (what a video is mainly about,
who appears when, where a query matches) is then derived from per-frame results by
plain, testable logic in this package.

See ``docs/video.md`` for the design and for what is and is not measured.
"""

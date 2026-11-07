"""Smart Gallery: what is in an indexed archive, and every way to search it.

    python -m vie.gallery --config data/runs/<run>/config.yaml

Everything here reads the index that ``vie index`` built. Recognition, thresholds
and search are the package's own; this layer adds the archive-wide view (people,
species, dishes, subjects), naming people, similar-image search and the interface.
"""

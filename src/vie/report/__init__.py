"""Self-contained visual reports.

Every report is one HTML file: images are embedded, nothing is fetched, and it
opens from a double-click on a machine with no network. That matters for this
project in particular, because a report can show photographs of identifiable
people and must never call out to a third party to render.
"""

from vie.report.search import write_search_report

__all__ = ["write_search_report"]

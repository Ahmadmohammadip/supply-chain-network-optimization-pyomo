"""Multi-echelon supply chain network optimization.

Jointly decides which plants and warehouses to open (strategic) and how to
produce, ship, and hold inventory across a multi-period horizon (operational),
as a single MILP over plants -> warehouses -> customers.

See docs/formulation.md for the model and its citations.
"""

__version__ = "0.1.0"

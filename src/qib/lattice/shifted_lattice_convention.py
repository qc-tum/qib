import enum


class ShiftedLatticeConvention(enum.Enum):
    """
    Convention for the Hexagonal and Brick lattices.
    Example with 1 row and 5 columns for COLS_SHIFTED_UP convention::

         _   _   _        _   _   _
        / \_/ \_/ \      | |_| |_| |
        \_/ \_/ \_/      |_| |_| |_|
          \_/ \_/          |_| |_|

    """
    COLS_SHIFTED_UP   = 1   # even numbered columns are shifted up relative to odd numbered columns
    ROWS_SHIFTED_LEFT = 2   # even numbered rows are shifted left relative to odd numbered rows

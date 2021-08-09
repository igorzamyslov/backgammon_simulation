from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast
from itertools import product, chain
from functools import cache


class Colour(enum.Enum):
    BLACK = enum.auto()
    WHITE = enum.auto()


QuadrantsOrderType = Tuple[int, int, int, int]


COLOUR_QUADRANTS_ORDER: Dict[Colour, QuadrantsOrderType] = {
    Colour.WHITE: (0, 1, 2, 3),
    Colour.BLACK: (2, 3, 0, 1),
}


@dataclass
class Coordinates:
    quadrant: int
    cell: int


@dataclass
class Cell:
    _colour: Optional[Colour] = None
    _count: int = 0

    def __post_init__(self):
        self._verify()

    def _verify(self):
        if self.count:
            assert self.colour
        else:
            assert not self.colour

    @property
    def is_empty(self):
        return self._count == 0

    @property
    def colour(self):
        return self._colour

    @property
    def count(self):
        return self._count

    def update_cell(self, new_count: int, **kwargs):
        not_provided_value = "not_provided"
        new_colour = kwargs.get("new_colour", not_provided_value)
        assert new_count >= 0, "Wrong count, must be >= 0"
        self._count = new_count
        if new_colour is not_provided_value:
            if self.colour is None:
                raise RuntimeError("Colour must be provided when updating empty cell")
            elif self.count == 0:
                self._colour = None
        else:
            if self.colour is None:
                self._colour = new_colour
            elif new_colour != self.colour:
                raise RuntimeError("Cannot change colour during cell update, "
                                   "must be emptied first")
        self._verify()


BoardQuadrantType = Tuple[Cell, Cell, Cell, Cell, Cell, Cell]


@dataclass
class Board:
    quadrants: Tuple[BoardQuadrantType, BoardQuadrantType,
                     BoardQuadrantType, BoardQuadrantType]

    def print_board(self):
        q_2_1 = chain(reversed(self.quadrants[1]), reversed(self.quadrants[0]))
        q_3_4 = chain(self.quadrants[2], self.quadrants[3])
        q_3_2_4_1 = list(zip(q_3_4, q_2_1))

        def get_line_symbol(default_symbol, min_count, cell) -> str:
            colour_symbol = "O" if cell.colour is Colour.WHITE else "X"
            return colour_symbol if cell.count > min_count else default_symbol

        # prepare data line by line
        data = []
        for i in range(19):
            for left_cell, right_cell in q_3_2_4_1:
                default = "-" if i == 9 else " "
                data.append(get_line_symbol(default, 18 - i, left_cell))
                data.append(get_line_symbol(default, i, right_cell))

        print("""
                    QII                              QI
| OFF | 11 | 10 |  9 |  8 |  7 |  6 |  |  5 |  4 |  3 |  2 |  1 |  0 | OFF |
|     |-------P=O Home Board--------|  |---------Outer Board---------|     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     |-{}{}-|-{}{}-|-{}{}-|-{}{}-|-{}{}-|-{}{}-|  |-{}{}-|-{}{}-|-{}{}-|-{}{}-|-{}{}-|-{}{}-|     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |  | {}{} | {}{} | {}{} | {}{} | {}{} | {}{} |     |
|     |--------Outer Board----------|  |-------P=X Home Board--------|     |
| OFF | 12 | 13 | 14 | 15 | 16 | 17 |  | 18 | 19 | 20 | 21 | 22 | 23 | OFF |
                    QIII                                QIV
       """.format(*data))

    @staticmethod
    @cache
    def _get_flattened_coordinates(quadrants_order: QuadrantsOrderType) -> List[Coordinates]:
        return [Coordinates(q, c) for q, c in product(quadrants_order, range(6))]

    def get_cell(self, coords: Coordinates) -> Cell:
        return self.quadrants[coords.quadrant][coords.cell]

    def get_target_cell_coords(self,
                               from_coords: Coordinates,
                               count: int,
                               quadrants_order: QuadrantsOrderType) -> Optional[Coordinates]:
        flattened_coords = self._get_flattened_coordinates(quadrants_order)
        target_index = flattened_coords.index(from_coords) + count
        try:
            return flattened_coords[target_index]
        except IndexError:
            return None

    def move(self, from_coords: Coordinates, count: int):
        cell = self.get_cell(from_coords)
        assert not cell.is_empty, f"Moving from empty cell: {from_coords}"
        colour = cast(Colour, cell.colour)  # get the colour before updating the cell
        # update current cell
        cell.update_cell(new_count=cell.count - 1)

        # update target cell, if required
        quadrants_order = COLOUR_QUADRANTS_ORDER[colour]
        target_cell_coords = self.get_target_cell_coords(from_coords, count, quadrants_order)
        if target_cell_coords is None:
            # out of board situation
            assert from_coords.quadrant == quadrants_order[-1], "Out of board from outer board"
            flattened_coords = self._get_flattened_coordinates(quadrants_order)
            current_index = flattened_coords.index(from_coords)
            previous_cells = (self.get_cell(c) for c in flattened_coords[:current_index])
            assert all(c.is_empty or c.colour != colour for c in previous_cells), \
                "Wrong move, previous cells must be emptied first"
        else:
            target_cell = self.get_cell(target_cell_coords)
            if target_cell.is_empty:
                target_cell.update_cell(new_count=1, new_colour=colour)
            else:
                try:
                    target_cell.update_cell(new_count=target_cell.count + 1, new_colour=colour)
                except RuntimeError as error:
                    raise RuntimeError("Wrong move, target cell is occupied") from error


# if __name__ == "__main__":
#     board = Board((
#         (Cell(Colour.WHITE, 15), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
#         (Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
#         (Cell(Colour.BLACK, 15), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
#         (Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
#     ))
#     board.print_board()
#     for _ in range(15):
#         board.move(Coordinates(0, 0), 23)
#     board.move(Coordinates(3, 5), 1)

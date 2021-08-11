from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast
from itertools import product, chain
from functools import cache
from copy import deepcopy


class Colour(enum.Enum):
    BLACK = enum.auto()
    WHITE = enum.auto()

    def get_opposite_colour(self) -> Colour:
        return Colour.BLACK if self == Colour.WHITE else Colour.WHITE


QuadrantsOrderType = Tuple[int, int, int, int]


COLOUR_QUADRANTS_ORDER: Dict[Colour, QuadrantsOrderType] = {
    Colour.WHITE: (0, 1, 2, 3),
    Colour.BLACK: (2, 3, 0, 1),
}


@dataclass(frozen=True)
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

    def get_possible_turns(self, die1: int, die2: int,
                           colour: Colour) -> List[Tuple[Tuple[Coordinates, int], ...]]:
        moves_permutations: List[Tuple[int, ...]]
        if die1 == die2:
            moves_permutations = [(die1, die1, die1, die1)]
        else:
            moves_permutations = [(die1, die2), (die2, die1)]

        def get_relevant_coordinates(board: Board) -> List[Coordinates]:
            return [Coordinates(q, c)
                    for q, quadrant in enumerate(board.quadrants)
                    for c, cell in enumerate(quadrant)
                    if cell.colour == colour]

        quadrants_order = COLOUR_QUADRANTS_ORDER[colour]

        # moves_permutation + previous moves + board + index
        boards: List[Tuple[List[Tuple[Coordinates, int]], Board, Tuple[int, ...], int]] = \
            [([], self, moves_perm, 0) for moves_perm in moves_permutations]
        possible_turns = set()
        while boards:
            previous_moves, board, moves_perm, index = boards.pop()
            for r_coords in get_relevant_coordinates(board):
                # skip third move from HEAD if it is an exception
                if (index > 1 and r_coords == Coordinates(quadrants_order[0], 0)
                        and (die1, die2) in {(3, 3), (4, 4), (6, 6)}):
                    continue
                # skip second move from HEAD if it is *not* an exception
                if (index > 0 and r_coords == Coordinates(quadrants_order[0], 0)
                        and (die1, die2) not in {(3, 3), (4, 4), (6, 6)}):
                    continue
                new_board = deepcopy(board)
                new_move = (r_coords, moves_perm[index])
                try:
                    new_board.move(*new_move)
                except (AssertionError, RuntimeError):
                    continue
                new_turn = previous_moves + [new_move]
                possible_turns.add(tuple(new_turn))
                if index < len(moves_permutations[0]) - 1:
                    boards.append((new_turn, new_board, moves_perm, index + 1))

        if possible_turns:
            max_turn_length = max(len(turn) for turn in possible_turns)
            return [turn for turn in possible_turns if len(turn) == max_turn_length]
        else:
            return []

    def _check_prime(self, target_coords: Coordinates, colour: Colour):
        """ check if prime will be built and if it is possible """
        quadrants_order = COLOUR_QUADRANTS_ORDER[colour]
        flattened_coords = self._get_flattened_coordinates(quadrants_order)
        target_index = flattened_coords.index(target_coords)
        count = 1
        # count back
        if target_index > 0:
            for coords in flattened_coords[target_index - 1::-1]:
                cell = self.get_cell(coords)
                if cell.colour == colour:
                    count += 1
                else:
                    break
        # count forward
        for coords in flattened_coords[target_index + 1:]:
            cell = self.get_cell(coords)
            if cell.colour == colour:
                count += 1
            else:
                break
        # in case prime was created
        if count >= 6:
            # check if it is valid
            opposite_quadrants_order = COLOUR_QUADRANTS_ORDER[colour.get_opposite_colour()]
            if all(cell.is_empty or cell.colour == colour
                   for cell in self.quadrants[opposite_quadrants_order[-1]]):
                raise RuntimeError("Prime created when not yet possible")

    def move(self, from_coords: Coordinates, count: int):
        cell = self.get_cell(from_coords)
        assert not cell.is_empty, f"Moving from empty cell: {from_coords}"
        colour = cast(Colour, cell.colour)  # get the colour before updating the cell
        # update current cell
        cell.update_cell(new_count=cell.count - 1)

        # update target cell, if required
        quadrants_order = COLOUR_QUADRANTS_ORDER[colour]
        target_cell_coords = self.get_target_cell_coords(from_coords, count, quadrants_order)
        try:
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
                if not target_cell.is_empty and target_cell.colour != colour:
                    raise RuntimeError("Wrong move, target cell is occupied")
                self._check_prime(target_cell_coords, colour)
                target_cell.update_cell(new_count=target_cell.count + 1, new_colour=colour)
        except (AssertionError, RuntimeError):
            # restore previous state
            cell.update_cell(new_count=cell.count + 1, new_colour=colour)
            raise


if __name__ == "__main__":
    board = Board((
        (Cell(Colour.WHITE, 15), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
        (Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
        (Cell(Colour.BLACK, 15), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
        (Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
    ))

    def roll_dice() -> Tuple[int, int]:
        return random.randint(1, 6), random.randint(1, 6)

    def check_win(colour: Colour) -> bool:
        return all(cell.is_empty or cell.colour != colour
                   for quadrant in board.quadrants
                   for cell in quadrant)

    colour = Colour.WHITE
    board = Board((
        (Cell(Colour.WHITE, 15), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
        (Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
        (Cell(Colour.BLACK, 15), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
        (Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0), Cell(None, 0)),
    ))
    while True:
        die1, die2 = roll_dice()
        possible_turns = board.get_possible_turns(die1, die2, colour)
        if possible_turns:
            turn = random.choice(possible_turns)
            for move in turn:
                board.move(*move)
            # board.print_board()
            if check_win(colour):
                print(f"{colour} won!")
                break
        colour = colour.get_opposite_colour()

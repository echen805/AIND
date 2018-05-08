
from utils import *

def diagonal(x,y):
    diag_units = []
    diag_units2 = []
    diagonal_list = []
    x_reverse = x[::-1]
    if len(x) == len(y):
        for count in range(0,len(x)):
            diag_units.append(x[count]+y[count])
        diagonal_list.append(diag_units)
        for count in range(0,len(y)):
            diag_units2.append(x_reverse[count]+y[count])
        diagonal_list.append(diag_units2)
    return diagonal_list

diagonal_units = diagonal(rows,cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diagonal_units = diagonal(rows,cols)

unitlist = row_units + column_units + square_units + diagonal_units

# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)



def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    The naked twins strategy says that if you have two or more unallocated boxes
    in a unit and there are only two digits that can go in those two boxes, then
    those two digits can be eliminated from the possible assignments of all other
    boxes in the same unit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).
    """
    #print("values")
    #display(values)

    possible_twins = [box for box in values.keys() if len(values[box]) == 2]
    twin_matches = [[box1,box2] for box1 in possible_twins for box2 in peers[box1] \
    if set(values[box1]) == set(values[box2])]
    
    #print("twin matches", twin_matches)
    peersList = []
    for i in range(len(twin_matches)):
        box1 = twin_matches[i][0]
        box2 = twin_matches[i][1]
        # Find the same values of the peers between the twins
        peers1 = set(peers[box1])
        peers2 = set(peers[box2])
        peers_int = peers1 & peers2
        # print("peers_int")
        # print(peers_int)
        peersList.append(peers_int)
        # print("peersList")
        # print(peersList)
        # Remove twin from the peers
        # I think I'm removing too much from the board. Remember to exclude the boxes that the twins exist in
        for individualPeerSets in peersList:
            # print("individualPeerInts\r")
            # print(individualPeerInts)
            for peer_val in individualPeerSets:
                print("peer_val")
                print(peer_val)
                if len(values[peer_val]) > 2:
                    for rm_val in values[box1]:
                        values = assign_value(values, peer_val, values[peer_val].replace(rm_val, ''))

    # OLD WORKING CODE EXCEPT PARTIALLY FAILING
    # for i in range(len(twin_matches)):
    #     box1 = twin_matches[i][0]
    #     box2 = twin_matches[i][1]
    #     # Find the same values of the peers between the twins
    #     peers1 = set(peers[box1])
    #     peers2 = set(peers[box2])
    #     peers_int = peers1 & peers2
    #     # Remove twin from the peers
    #     for peer_val in peers_int:
    #         if len(values[peer_val])>2:
    #             for rm_val in values[box1]:
    #                 values = assign_value(values, peer_val, values[peer_val].replace(rm_val,''))
    #print("post naked twin removal")
    #display(values)
    print("values\r")
    print(display(values))
    return values    

def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        #values = naked_twins(values)
        values = only_choice(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    "Using depth-first search and propagation, try all possible values."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes): 
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and 
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    # display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    # display(result)
    naked_twins(grid2values(diag_sudoku_grid))

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')

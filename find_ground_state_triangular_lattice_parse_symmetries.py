import re
from pathlib import Path

from sympy import Rational
from sympy.combinatorics import Permutation


# Test cases
def test_parse_permutation():
    # Test single element permutation
    assert parse_permutation("Permutation(35)") == Permutation(35)
    
    # Test simple cycle
    assert parse_permutation("Permutation(1, 5)") == Permutation([[1, 5]])
    
    # Test multiple cycles
    assert parse_permutation("Permutation(1, 5)(2, 4)") == Permutation([[1, 5], [2, 4]])
    
    # Test longer permutation
    assert parse_permutation("Permutation(1, 5)(2, 4)(6, 30)(7, 35)") == \
           Permutation([[1, 5], [2, 4], [6, 30], [7, 35]])
           
    # Test cycles with more than 2 elements
    assert parse_permutation("Permutation(1, 2, 3)(0, 4)") == \
           Permutation([[1, 2, 3], [0, 4]])
    
    # Test mixed length cycles
    assert parse_permutation("Permutation(1, 2, 3, 4)(5, 6)(7, 8, 9)") == \
           Permutation([[1, 2, 3, 4], [5, 6], [7, 8, 9]])
    

def test_parse_symmetries():
    # Test simple case
    simple_input = "[(Permutation(35), 0)]"
    result = parse_symmetries(simple_input)
    assert len(result) == 1
    assert result[0][0] == Permutation(35)
    assert result[0][1] == Rational(0)
    
    # Test multiple symmetries
    multiple_input = "[(Permutation(35), 0), (Permutation(1, 5)(2, 4), 1/2)]"
    result = parse_symmetries(multiple_input)
    assert len(result) == 2
    assert result[0][0] == Permutation(35)
    assert result[0][1] == Rational(0)
    assert result[1][0] == Permutation([[1, 5], [2, 4]])
    assert result[1][1] == Rational(1, 2)
    
    # Test empty input
    empty_input = "[]"
    assert len(parse_symmetries(empty_input)) == 0
    

def run_tests():
    test_parse_permutation()
    test_parse_symmetries()

output = Path("experiments/find_ground_state_triangular_lattice/TriangularLattice6x6-enumerate-along-x_J2_1.3.txt")

prefix = "ground_state_system.hamiltonian.basis.symmetries="
symmetries = next(line.removeprefix(prefix).strip() for line in output.open() if line.startswith(prefix))

def parse_permutation(perm_str):
    # Remove 'Permutation' from the string if present
    perm_str = perm_str.replace('Permutation', '').strip()
    
    # Handle single element permutation
    if ',' not in perm_str:
        num = int(re.findall(r'\d+', perm_str)[0])
        return Permutation(num)
    
    # Parse cyclic form like (1,2)(3,4) or (1,2,3)(0,4)
    cycles = []
    
    # Split into individual cycles and handle both parenthesized and non-parenthesized formats
    if '(' in perm_str:
        # Multiple cycles in (a,b)(c,d) format
        cycle_matches = re.finditer(r'\(([\d,\s]+)\)', perm_str)
        for match in cycle_matches:
            cycle_str = match.group(1)
            numbers = [int(n) for n in re.findall(r'\d+', cycle_str)]
            cycles.append(numbers)
    else:
        # Single cycle in a,b format
        numbers = [int(n) for n in re.findall(r'\d+', perm_str)]
        cycles.append(numbers)
    
    return Permutation(cycles)

def parse_symmetries(symmetries_str):
    # Remove square brackets and split by '),('
    items = symmetries_str.strip('[]').split('), (')
    
    result = []
    for item in items:
        if not item.strip():  # Skip empty items
            continue
            
        # Clean up the item string
        item = item.strip('()')
        
        # Split into permutation and rational parts
        perm_str, rational_str = item.rsplit(', ', 1)
        
        # Parse permutation
        perm = parse_permutation(perm_str)
        
        # Parse rational number
        if '/' in rational_str:
            num, denom = map(int, rational_str.split('/'))
            rational = Rational(num, denom)
        else:
            rational = Rational(int(rational_str), 1)
        
        result.append((perm, rational))
    
    return result

# Run tests first
run_tests()

# Then process the actual file
output = Path("./symmetries-ground-state-TriangularLattice6x6-enumerate-along-x_J2_1.3.txt")
prefix = "ground_state_system.hamiltonian.basis.symmetries="
symmetries = next(line.removeprefix(prefix).strip() for line in output.open() if line.startswith(prefix))

parsed_symmetries = parse_symmetries(symmetries)

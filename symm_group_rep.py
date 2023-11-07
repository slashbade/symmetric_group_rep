from itertools import combinations
from math import factorial, ceil
from copy import deepcopy


class Tableau:
    """Young Tableau operations
    """
    def __init__(self, tableau:list[list]) -> None:
        self.p = [len(row) for row in tableau]
        self.tableau = tableau
        self.n = sum(self.p)
    
    def __str__(self):
        to_print = ""
        for row in self.tableau:
            to_print += "{}\n".format(' '.join(map(str, row)))
        return to_print
    
    def __rmul__(self, perm: 'Permutation') -> 'Tableau':
        """Perm acts on tableau`

        Args:
            perm (Permutation): perm elm

        Returns:
            Tableau: result tableau
        """
        new_tableau = [[0]*len(row) for row in self.tableau]
        for row_index, row in enumerate(self.tableau):
            for col_index, entry in enumerate(row):
                new_tableau[row_index][col_index] = perm[entry - 1]
        return Tableau(new_tableau)
    
    
    def robinson_schensted_insertion(self, element: int) -> 'Tableau':
        """Inserts one num to tableau

        Args:
            element (int): new num

        Returns:
            Tableau: result tableau
        """
        def insert_location(row: list, element: int) -> int:
            """binary search location
            """
            l, r = 0, len(row)
            while l < r:
                mid = (l + r) // 2
                if row[mid] >= element:
                    r = mid
                else:
                    l = mid + 1
            return l
        
        tableau = self.tableau
        tableau.append([])
        for j in range(len(tableau)):
            if not tableau[j] or element >= tableau[j][-1]:
                tableau[j].append(element)
                break
            else:
                location = insert_location(tableau[j], element)
                temp = tableau[j][location]
                tableau[j][location] = element
                element = temp
        while not tableau[-1]:
            tableau.pop()
        return Tableau(tableau)
    
    @staticmethod
    def robinson_schensted(perm: 'Permutation') -> 'Tableau':
        """Robinson-Schensted Algorithm

        Returns:
            Tableau: corresponded tableau
        """
        _Tableau = Tableau([[]])
        for element in perm.perm:
            _Tableau = _Tableau.robinson_schensted_insertion(element)
        return _Tableau
    
    def as_dict(self) -> dict:
        pass
    
    def column_stablizer(self):
        pass
    
    def row_stablizer(self):
        pass

class Permutation:
    def __init__(self, perm: list) -> None:
        self.perm = perm
    
    def __getitem__(self, index: int):
        return self.perm[index]
    
    def __len__(self):
        return len(self.perm)
    
    def __str__(self):
        return str(self.perm)
    
    def __mul__(self, other: 'Permutation') -> 'Permutation':
        if not isinstance(other, Permutation):
            return NotImplemented
        assert len(self) == len(other), "Length not match"
        prod = []
        for entry in other.perm:
            prod.append(self[entry-1])
        return Permutation(prod)
    
    def from_disjoint_cycles(disjoint_cycles: list[tuple], size: int) -> 'Permutation':
        """Converts perm from disjoint cycles

        Args:
            disjoint_cycles (list[tuple]): list of cycles
            size (int): maximum numbers in the perm

        Returns:
            Permutation: corresponding perm
        """
        perm = Permutation(list(range(1, size+1)))
        for cycle in disjoint_cycles:
            perm *= Permutation._cycle_to_perm(cycle, size)
        return perm
    
    def _cycle_to_perm(cycle: tuple, size: int) -> 'Permutation':
        perm_list = []
        for entry in range(1, size+1):
            if entry not in cycle:
                perm_list.append(entry)
            else:
                perm_list.append(cycle[(cycle.index(entry)+1) % 
                                  len(cycle)])
        return Permutation(perm_list)
            
    
    def as_disjoint_cycles(self) -> list:
        """Converts perm to cycles

        Returns:
            list: corresponding list of cycles
        """
        disjoint_cycles = []
        perm = deepcopy(self.perm)
        perm0 = list(range(1, len(perm)+1))
        while self.perm:
            # get a cycle
            cycle = []
            index = 0
            index_list = [0]
            cycle.append(perm[index])
            while perm[perm0.index(perm[index])] != perm[0]:
                index = perm0.index(perm[index])
                index_list.append(index) # keep a list of index
                cycle.append(perm[index])
            # add the cycle to prod
            if len(cycle) > 1:
                disjoint_cycles.append(cycle)
            # delete the cycle from entry
            index_list.sort(reverse=True)
            for index in index_list:
                perm.pop(index)
                perm0.pop(index)
        return disjoint_cycles

        

class Partition:
    """Stores information of a Young Diagram or partition
    """

    def __init__(self, entry: list = []):
        self.entry = entry

    def __add__(self, other):
        """Obtains the union of two partitions.

        Args:
            other (Partition): Partition object

        Returns:
            Partition: Partition object
        """
        pu = []
        p_1 = deepcopy(self.entry)
        p_2 = deepcopy(other.entry)
        if len(p_1) <= len(p_2):  # fill zeros to achieve same length
            p_1 += (len(p_2) - len(p_1)) * [0]
        else:
            p_2 += (len(p_1) - len(p_2)) * [0]
        for i in range(len(p_1)):
            pu.append(p_1[i] + p_2[i])
        return Partition(pu)
    
    def __eq__(self, other):
        """Compares two partitions (resp. Young Tableau)

        Args:
            other (Partition): Partition object
        
        Returns:
            Bool
        """
        p_1 = self.entry
        p_2 = other.entry
        while p_1[-1] == 0:
            p_1.pop()
        while p_2[-1] == 0:
            p_2.pop()
        return p_1 == p_2
    
    def __lt__(self, other: 'Partition'):
        """Total lexical order
        """
        for i in range(min(len(self), len(other))):
            if self[i] != other[i]:
                if self[i] < other[i]:
                    return True
                else:
                    return False
        return False
    
    def __gt__(self, other: 'Partition'):
        """Total lexical order
        """
        for i in range(min(len(self), len(other))):
            if self[i] != other[i]:
                if self[i] > other[i]:
                    return True
                else:
                    return False
        return False
    
    @staticmethod
    def dominate_partial_order(p1: Permutation, p2:Permutation):
        """Checks if p1 dominates p2
        """
        entry1 = p1.entry
        entry2 = p2.entry
        if len(entry1) <= len(entry2):  # fill zeros to achieve same length
            entry1 += (len(entry2) - len(entry1)) * [0]
        else:
            entry2 += (len(entry1) - len(entry2)) * [0]
        conds = []
        conds.append(entry2[0] <= entry1[0])
        for i in range(1, len(entry1)):
            conds.append(sum(entry2[:i+1]) < sum(entry1[:i+1]))
        return all(conds)

    def __str__(self):
        return str(self.entry)
    
    def __getitem__(self, index):
        return self.entry[index]
    
    def __len__(self):
        return len(self.entry)
    
    def as_ferrers(self, char='#'):
        """
        Prints the ferrer diagram of a partition.
        """
        return "\n".join([char*i for i in self.entry])

    @property
    def conjugate(self):
        p = self.entry
        p_transposed = []
        for num in range(sum(p)):
            count = 0
            for row_len in p:
                if num < row_len:
                    count += 1
            if count == 0:
                break
            else:
                p_transposed.append(count)
        return Partition(p_transposed)


    def is_very_even(self):
        """Checks whether a partition is very even.

        Returns:
            bool: True or False
        """
        newp = deepcopy(self.entry)
        while newp[-1] == 0:  # delete zeros
            newp.pop()
        flag = 0
        for p_k in newp:
            if p_k % 2 == 0 and newp.count(p_k) % 2 == 0:
                flag += 1
        if flag == len(newp):
            return True
        else:
            return False
    
    @property
    def odd_entry(self):
        p = self.entry
        p_even = []
        for i in range(len(p)):
            if i%2 == 0:
                p_even.append(int(p[i]/2))
            else:
                p_even.append(ceil(p[i]/2))
        while p_even[-1] == 0:
            p_even.pop()
        return p_even
    
    @property
    def even_entry(self):
        p = self.entry
        p_odd = []
        for i in range(len(p)):
            if i%2 == 0:
                p_odd.append(ceil(p[i]/2))
            else:
                p_odd.append(int(p[i]/2))
        while p_odd[-1] == 0:
            p_odd.pop()
        return p_odd
    
    @staticmethod
    def generate_all_partitions(n:int) -> list['Partition']:
        l = n
        def backtrack(start, target, path):
            if target == 0:
                p_list.append(Partition(list(path)))
                return 
            for i in range(min(start, target), 0, -1):
                path.append(i)
                backtrack(i, target - i, path)
                path.pop()

        p_list = []
        backtrack(l, l, [])
        return p_list


class Tabloid:
    def __init__(self, tableau) -> None:
        self.p = [len(row) for row in tableau]
        self.tabloid = [set(row) for row in tableau]
        self.n = sum(self.p)
    
    def _in_row_index(self, i: int):
        """Finds the row a number is in

        Args:
            i (int): num
        """
        for row_index in range(len(self.tabloid)):
            if i in self.tabloid[row_index]:
                # print(i, self.t[row_index], row_index)
                return row_index
    
    def __lt__(self, other):
        """Total order for tabloids
        """
        for i in range(self.n-1, -1, -1):
            if self._in_row_index(i) != other._in_row_index(i):
                if self._in_row_index(i) < other._in_row_index(i):
                    return True
                else:
                    return False
        return False
    
    def __gt__(self, other):
        """Total order for tabloids
        """
        for i in range(self.n-1, -1, -1):
            if self._in_row_index(i) != other._in_row_index(i):
                if self._in_row_index(i) > other._in_row_index(i):
                    return True
                else:
                    return False
        return False
    
    def __eq__(self, other):
        return self.tabloid == other.tabloid
    
    def __str__(self):
        return str(self.tabloid)

    @staticmethod
    def generate_all_tabloids(p):
        def backtrack(number_set, tableau, k, p):
            if k >= len(p):
                return [tableau.copy()]
            else:
                tabloids = []
                for row in combinations(number_set, p[k]):
                    new_tableau = tableau + [row]
                    tabloids.extend(backtrack(number_set - set(row), new_tableau, k + 1, p))
                return tabloids
        
        n = sum(p)    
        number_set = set(range(n))
        tabloids = backtrack(number_set, [], 0, p)
        return tabloids




def dim_S_lbd(p: Partition) -> int:
    """Computes the dim of a S_n-rep S

    Args:
        p (Partition): S_n-rep characterized by p

    Returns:
        int: dim of S
    """
    hook_prod = 1
    for row_index in range(len(p)):
        for col_index in range(p[row_index]):
            hook = p[row_index] - 1 - col_index + p.conjugate[col_index] - row_index
            hook_prod *= hook
    return factorial(sum(p))/hook_prod





if __name__ == "__main__":
    s1 = Permutation([2,1,3,4,5,6])
    s2 = Permutation([1,3,2,4,5,6])
    print(s1*s2)
    t = Tableau([[1,2,3,5],[4],[6]])
    print(s1*t)
    
    s = Permutation([5,2,3,1,4])
    t = Tableau.robinson_schensted(s)
    print(t)
    print(Permutation.from_disjoint_cycles([(1,2,3), (4,5)], 5))
    partitions = Partition.generate_all_partitions(5)
    #partitions.sort()
    for p in partitions:
        print(p.as_ferrers())
    
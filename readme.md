# Python Toolbox for Symmetric Group Representation

A Python implementation project for Symmetric Group $S_n$ related computations.

## Introduction

Representation of Symmetric Group, as well as related Young tableau algorithm, is center for various topics in representation theory and combinatorial mathematics. This library is a project on researching and learning representation theory. The toolbox covers

- Group permutations computations and convertions
- Partitions algorithms and order
- Tableaus, and tabloids algorithms and order
- Dimensions of Representation

More fundamental group-perm operations may have been established in classical computer algebra systems such GAP. And this library tend to focus more on characterization of Young tableau and tabloids.

## Usage

```from symm_group_rep import *```

## Examples

Group mult and actions

```Python
s1 = Permutation([2,1,3,4,5,6])
s2 = Permutation([1,3,2,4,5,6])
print(s1*s2)
```

```Python
t = Tableau([[1,2,3,5],[4],[6]])
print(s1*t)
```

Robinson-Schensted-Knuth Correspondence

```Python
s = Permutation([5,2,3,1,4])
t = Tableau.robinson_schensted(s)
print(t)
```

## Reference

(GTM211) Bruce. E. Sagon, The Symmetric Group Representations, Combinotorial Algorithms, and Symmetric Functions, 2001

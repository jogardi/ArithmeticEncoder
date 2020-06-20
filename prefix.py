import heapq
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChildPosition:
    """
    represents a position where a new child could go

    Attributes:
        parent  can be None if the parent is the root
        index is 0 if this is the left child and 1 if this is the right child
    """
    parent: Optional[tuple]
    index: int


class RootsIterator:

    def __init__(self, roots):
        self.current = ChildPosition(None, -1)
        self.roots = roots

    def has_next(self):
        return len(self.roots) > 0 or self.current.index == 0

    def next(self):
        if self.current.index == 0:
            self.current = ChildPosition(self.current.parent, 1)
        else:
            self.current = ChildPosition(self.roots.popleft(), 0)

        return self.current


def prefix_code_for_lengths(lengths):
    """
    :param lengths: a sorted list of lengths satisfying the kraft inequality
    :type lengths: List[int]
    """
    assert sum([2 ** -length for length in lengths]) <= 1
    heapq.heapify(lengths)

    next_length = heapq.heappop(lengths)
    leaves = []

    def make_prefix_tree(roots=RootsIterator(deque([None])), height=0):
        """
        :param roots: This iterator allows us to iterate the next level of the tree
        :param height:
        :return:
        """
        nonlocal next_length

        # add leaves
        current_position = roots.next()

        # add all the leaves for this level
        while next_length == height + 1:
            leaves.append((current_position.index, current_position.parent))
            if len(lengths) == 0:
                return
            else:
                next_length = heapq.heappop(lengths)
            # there has to be a next position because of the kraft inequality
            current_position = roots.next()

        # just add non leaves to the rest
        next_roots = deque()
        while True:
            next_roots.append((current_position.index, current_position.parent))
            if not roots.has_next():
                break

        make_prefix_tree(RootsIterator(next_roots), height + 1)
        
    make_prefix_tree()
    return leaves

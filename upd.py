import array
import math

from collections import defaultdict, Counter
from io import StringIO, BytesIO
from typing import Union, List, Iterator, Any, Dict, Tuple, Callable


def isnull(el: Any) -> bool:
    # TODO: sentinel int nulls?
    if el is None or math.isnan(el):
        return True

    return False


class Column:
    # TODO: could be List[Dict[str, Any]] or List[int/float] for
    # that matter - perhaps List[Any]
    # if we allow iterables, make sure to store the data, otherwise we will
    # lose it upon type inference
    # TODO: revise all 'return Column' code - we lose 'name' and other attributes
    def __init__(self, data: Union[List[str], List[int], List[float]], name: str = None) -> None:
        self._name = name
        self._dtype = Any
        types = set([type(el) for el in data])
        self._null_array = None

        # TODO: dtype = type(None) means null column, dtype = None
        # means no data

        if len(data) == 0:
            self._dtype = None  # or type(None)?
            self._data = []
            return

        if isinstance(data, Column):
            # TODO: very inefficient, why do we have to copy and do type
            # inference?
            data = data.tolist()

        # TODO: test all this inference
        if len(types) == 1:
            self._dtype = types.pop()
            if self._dtype is type(None):
                return

            typemap = {
                float: 'f',
                int: 'l',
                bool: 'b',
            }
            if self._dtype in typemap:
                self._data = array.array(typemap[self._dtype], data)
            else:
                self._data = data
        elif len(types) == 2 and type(None) in types:
            types.remove(type(None))
            self._dtype = types.pop()
            if self._dtype is float:  # can use math.nan
                self._data = array.array('f', [j or math.nan for j in data])
            elif self._dtype is int:
                self._data = array.array('l', [j or 0 for j in data])
                self._null_array = array.array(
                    'b', [1 if j is None else 0 for j in data])
            elif self._dtype is bool:
                self._data = array.array('b', [j or 0 for j in data])
                self._null_array = array.array(
                    'b', [1 if j is None else 0 for j in data])
            else:
                self._data = data
                self._null_array = array.array(
                    'b', [1 if j is None else 0 for j in data])
        elif len(types) == 2 and (float in types and int in types):
            self._dtype = float
            self._data = array.array('f', data)  # coerce ints
        elif len(types) == 3 and (type(None) in types and float in types and int in types):
            self._dtype = float
            self._data = array.array('f', [j or math.nan for j in data])
        else:
            raise NotImplementedError

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Union[str, int, float, bool]]:
        if self._dtype is type(None):
            raise NotImplementedError  # there is no length to a None array
            for j in range(len(self)):
                yield None
            return

        # we could simplify all this by just calling self[j],
        # but since we know about null arrays, we can yield directly
        if self._null_array is None:
            yield from self._data
        else:
            for j, el in enumerate(self._data):
                yield None if self._null_array[j] == 1 else self._data[j]

    def __repr__(self) -> str:
        # raise NotImplementedError
        #  TODO: test printing None-typed columns
        if self._dtype is None:
            return 'no data'
        elif isinstance(self._dtype, type(None)):
            return 'none-typed data'

        return str(self._data)

    def __getitem__(self, loc: Union[int, slice, List[int]]
                    ) -> Union[str, int, float, bool, 'Column']:
        if isinstance(loc, int):
            if self._null_array is not None and self._null_array[loc] == 1:
                return None
            else:
                return self._data[loc]
        elif isinstance(loc, slice):
            if self._null_array is None:
                return Column(self._data[loc])
            else:
                # TODO: test this thoroughly
                start, stop, stride = loc.indices(len(self))
                ret = [self._data[j] if self._null_array[j] == 0 else None
                       for j in range(start, stop)]
                return Column(ret)
        elif isinstance(loc, list):
            return Column([self[j] for j in loc])
        else:
            raise NotImplementedError

    def __setitem__(self, loc: Union[int, slice], value: Any) -> None:
        if isinstance(loc, int):
            # TODO: this will go through with True/False
            if isinstance(value, self._dtype):
                self._data[loc] = value
            else:
                raise ValueError('type mismatch')  #  make it more verbose
        elif isinstance(loc, slice):
            raise NotImplementedError
        else:  #  TODO: List[int] would be fine, too
            raise NotImplementedError

    def apply(self, func) -> 'Column':
        return Column([func(j) for j in self])

    def tolist(self) -> List[Any]:
        return list(self)

    def copy(self) -> 'Column':
        # TODO: when __init__ does a proper copy of data, change this to
        # pass a reference to self._data instead
        return Column(self.tolist())

    def cast(self, to_dtype: Union[type, str]) -> 'Column':
        # TODO: add tests for numeric column types
        if self._dtype in (int, float, 'int',
                           'float') and to_dtype in (str, 'str'):
            return self.apply(lambda x: None if isnull(x) else str(x))

        raise NotImplementedError

    def head(self, n: int = 5) -> 'Column':
        return self[slice(None, n, None)]

    def tail(self, n: int = 5) -> 'Column':
        return self[slice(-n, None, None)]

    def isnull(self) -> 'Column':  #  TODO: custom types - eg Column[bool]
        return self.apply(isnull)

    def all(self):
        if self._dtype is not bool:
            raise TypeError('cannot apply .map on a non-boolean column')

        for el in self:
            if el is False:
                return False

        return True

    def any(self):
        if self._dtype is not bool:
            raise TypeError('cannot apply .any on a non-boolean column')

        for el in self:
            if el is True:
                return True

        return False

    def count(self) -> int:
        ret = 0
        for el in self:
            if not isnull(el):
                ret += 1

        return ret

    def fillna(self, value: Any) -> 'Column':
        ret = [value if isnull(j) else j for j in self]
        return Column(ret)

    def filter(self, func) -> 'Column':
        if not callable(func):
            raise TypeError('you can only filter using functions')

        return Column([j for j in self if func(j)])

    def dropna(self) -> 'Column':
        return self.filter(lambda x: not(isnull(x)))

    # TODO: values could be a column
    # TODO: test if values contains None - there should be special treatment
    # for that
    def isin(self, values: Union[set, List[Any]]) -> 'Column':
        if not isinstance(values, set):
            values = set(values)

        ret = [j in values if not isnull(j) else None for j in self]
        return Column(ret)

    def notnull(self) -> 'Column':
        return self.apply(lambda x: not(isnull(x)))

    def sum(self):
        if self._dtype not in (int, float):
            raise TypeError('cannot apply .sum on a non-numeric column')

        return sum(self._data)

    def max(self):
        if self._dtype not in (int, float):
            raise TypeError('cannot apply .max on a non-numeric column')

        maxval = -math.inf
        for el in self:
            if not(isnull(el)) and el > maxval:
                maxval = el

        return maxval

    def min(self):
        if self._dtype not in (int, float):
            raise TypeError('cannot apply .min on a non-numeric column')

        minval = +math.inf
        for el in self:
            if not(isnull(el)) and el < minval:
                minval = el

        return minval

    def mean(self):
        if self._dtype not in (int, float):
            raise TypeError('cannot apply .mean on a non-numeric column')

        return self.sum() / self.count()

    def quantile(self):
        # TODO: include all the relevant algos
        raise NotImplementedError

    def unique(self):
        """includes nulls"""
        # nan issue https://github.com/numpy/numpy/issues/9358
        # can be sped up, if we know no values are nulls (but not when dtype
        # is float, we have to be careful there)
        if self._dtype is not float and self._null_array is None:
            return list(set(self))

        fin = self.dropna().tolist()
        if len(fin) < len(self):
            fin += [math.nan] if self._dtype is float else [None]
        return list(set(fin))

    def nunique(self) -> int:
        return len(self.unique())

    # TODO: fails: upd.Column([1,2,3,None,2,1]).sort().tail(1)
    def sort(self, ascending=True, nulls_first=False) -> 'Column':
        reverse = not(ascending)
        # TODO: inefficient, it will always copy the array twice,
        # even if there are no nulls - check it contains any
        # also, .dropna copies already, so we can sort in place?
        ret = sorted(self.dropna(), reverse=reverse)
        nnulls = len(self) - len(ret)  # how many nulls are there?
        if nulls_first:
            return Column(nnulls * [None] + ret)
        else:
            return Column(ret + nnulls * [None])

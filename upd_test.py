import upd

import unittest
import math

# TODO: test +-inf everywhere (especially sums etc.)


class TestColumnInit(unittest.TestCase):
    def test_basic_init(self):
        samples = [[1, 2, 3], [4, 5, 6],
                   list(range(1000)), [1, None, 2, None], []]

        for ds in samples:
            s = upd.Column(ds)
            self.assertEqual(len(s), len(ds))

            for j, el in enumerate(s):
                self.assertEqual(el, ds[j])

    def test_type_inference(self):
        tt = [
            [[1, 2, 3], int],
            [[1, 2.0, 3], float],
            [[1, math.nan, 3], float],
            [[1, 2, None], int],
            [[1, 2, None, 3.0], float],
            [[True, False, True], bool],
            [[True, None, False], bool],
            [[], None],
            [[None, None], type(None)],
        ]
        for el in tt:
            c = upd.Column(el[0])
            self.assertEqual(c.dtype, el[1])


class TestColumnApply(unittest.TestCase):
    def test_apply(self):
        s = upd.Column([1, 2, 3])
        s = s.apply(lambda x: x**2 - 3)

        self.assertEqual(s.tolist(), [-2, 1, 6])


class TestColumnCopy(unittest.TestCase):
    def test_copy(self):
        s = upd.Column([1, 2, 3])
        sc = s.copy()
        self.assertEqual(s.tolist(), sc.tolist())
        sc[0] = 10

        self.assertNotEqual(s.tolist(), sc.tolist())


class TestHeadTail(unittest.TestCase):
    def test_head(self):
        pass

    def test_tail(self):
        pass


class TestFunctions(unittest.TestCase):
    def test_count(self):
        pass

    def test_isnull(self):
        pass

    # TODO: test assertRaises in both all/any for non-bool columns
    def test_all(self):
        pass

    def test_any(self):
        pass

    def test_fillna(self):
        pass

    def test_filter(self):
        pass

    def test_dropna(self):
        tt = [
            [[1, 2, 3, None, 4], [1, 2, 3, 4]],
            [[1, math.nan, 4], [1, 4]],
        ]
        for el in tt:
            self.assertEqual(upd.Column(el[0]).dropna().tolist(), el[1])

    def test_isin(self):
        pass

    def test_notnull(self):
        pass

    def test_sum(self):
        pass

    def test_max(self):
        pass

    def test_min(self):
        pass

    def test_mean(self):
        pass

    def test_quantile(self):
        pass

    def test_unique(self):
        pass

    def test_nunique(self):
        tt = [
            [[1, 2, 3], 3],
            [[1, 2, 3, None], 4],
            [[1, 2, 3, 3, 2, 1], 3],
            [[1, 2, None, math.nan], 3],  #  float nan issue in `array`
        ]
        for el in tt:
            self.assertEqual(upd.Column(el[0]).nunique(), el[1])

    def test_sort(self):
        pass

    def test_value_counts(self):
        pass


if __name__ == '__main__':
    unittest.main()

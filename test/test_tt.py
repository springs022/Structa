import unittest

from tt import UnreachableTT, CostTT


def _stats():
    return {
        "lookups": 0,
        "hits": 0,
        "stores": 0,
        "store_updates": 0,
        "evictions": 0,
    }


class UnreachableTTTests(unittest.TestCase):
    def test_exact_remain_hits(self):
        tt = UnreachableTT(1024, _stats())
        tt.store(0x123456789ABCDEF, 5)
        self.assertTrue(tt.hit(0x123456789ABCDEF, 5, margin=0))
        self.assertTrue(tt.hit(0x123456789ABCDEF, 5, margin=3))

    def test_unknown_key_misses(self):
        tt = UnreachableTT(1024, _stats())
        tt.store(1, 5)
        self.assertFalse(tt.hit(2, 5, margin=0))

    def test_margin_semantics_match_previous_implementation(self):
        # 旧実装：delta == 0 または delta > margin のときにヒット
        for failed_remain in range(0, 8):
            for remain in range(0, 8):
                for margin in range(0, 6):
                    tt = UnreachableTT(1024, _stats())
                    tt.store(0xABCD, failed_remain)
                    delta = failed_remain - remain
                    expected = (delta == 0) or (delta > margin)
                    with self.subTest(
                        failed_remain=failed_remain, remain=remain, margin=margin
                    ):
                        self.assertEqual(
                            expected, tt.hit(0xABCD, remain, margin)
                        )

    def test_store_keeps_larger_remain(self):
        tt = UnreachableTT(1024, _stats())
        tt.store(7, 3)
        tt.store(7, 9)
        self.assertTrue(tt.hit(7, 9, margin=0))
        tt.store(7, 2)
        self.assertTrue(tt.hit(7, 9, margin=0))

    def test_max_key_value_is_accepted(self):
        tt = UnreachableTT(1024, _stats())
        h = (1 << 64) - 1
        tt.store(h, 4)
        self.assertTrue(tt.hit(h, 4, margin=0))

    def test_size_is_power_of_two_and_bounded(self):
        tt = UnreachableTT(5000, _stats())
        self.assertEqual(4096, tt.size)
        self.assertEqual(0, tt.size & (tt.size - 1))


class CostTTTests(unittest.TestCase):
    def test_round_trip(self):
        tt = CostTT(1024, {"lookups": 0, "hits": 0})
        self.assertIsNone(tt.get(42))
        tt.put(42, 3, 7)
        self.assertEqual((3, 7), tt.get(42))

    def test_large_values_are_clamped_but_still_exceed_budget(self):
        tt = CostTT(1024, {"lookups": 0, "hits": 0})
        tt.put(1, 1000, 1000)
        need_s, need_g = tt.get(1)
        self.assertGreater(need_s, 127)
        self.assertGreater(need_g, 127)

    def test_zero_costs_are_distinguished_from_empty(self):
        tt = CostTT(1024, {"lookups": 0, "hits": 0})
        tt.put(99, 0, 0)
        self.assertEqual((0, 0), tt.get(99))


if __name__ == "__main__":
    unittest.main()

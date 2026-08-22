import unittest

from io_utils import format_elapsed_time


class ElapsedTimeFormatTests(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual("0時間00分00秒", format_elapsed_time(0))
        self.assertEqual("0時間00分59秒", format_elapsed_time(59.9))

    def test_minutes_and_hours(self):
        self.assertEqual("0時間01分00秒", format_elapsed_time(60))
        self.assertEqual("1時間05分05秒", format_elapsed_time(3905))

    def test_negative_value_is_clamped(self):
        self.assertEqual("0時間00分00秒", format_elapsed_time(-1))


if __name__ == "__main__":
    unittest.main()

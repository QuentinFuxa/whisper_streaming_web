import unittest
from src.whisper.timestaped_words import TimeStampedSegment, TimeStampedSequence

class TestTimeStampedSegment(unittest.TestCase):
    def setUp(self):
        self.segment = TimeStampedSegment(0.0, 1.0, "Hello")
    
    def test_initialization(self):
        self.assertEqual(str(self.segment), "Hello (0.000 - 1.000)")
        self.assertEqual(repr(self.segment), "(0.000, 1.000, 'Hello')")
    
    def test_shift(self):
        self.segment.shift(1.0)
        self.assertEqual(repr(self.segment), "(1.000, 2.000, 'Hello')")
    
    def test_equality(self):
        other = TimeStampedSegment(0.0, 1.0, "Hello")
        self.assertEqual(self.segment, other)
        self.assertEqual(self.segment, "Hello")
    
    def test_addition(self):
        # Test number addition (shift)
        shifted = self.segment + 1.0
        self.assertEqual(repr(shifted), "(1.000, 2.000, 'Hello')")
        
        # Test string concatenation
        concatenated = self.segment + " World"
        self.assertEqual(repr(concatenated), "(0.000, 1.000, 'Hello World')")
def test_comparison_with_numbers(self):
    segment = TimeStampedSegment(1.0, 2.0, "Hello")
    # Test less than
    self.assertTrue(segment < 3.0)
    self.assertFalse(segment < 1.5)
    # Test greater than
    self.assertTrue(segment > 0.5)
    self.assertFalse(segment > 1.5)

def test_comparison_with_segments(self):
    seg1 = TimeStampedSegment(1.0, 2.0, "Hello")
    seg2 = TimeStampedSegment(2.0, 3.0, "World")
    seg3 = TimeStampedSegment(1.5, 2.5, "Overlap")
    
    # Test less than
    self.assertTrue(seg1 < seg2)
    self.assertFalse(seg1 < seg3)
    # Test greater than
    self.assertTrue(seg2 > seg1)
    self.assertFalse(seg3 > seg1)

def test_comparison_invalid_types(self):
    segment = TimeStampedSegment(1.0, 2.0, "Hello")
    # Test less than
    with self.assertRaises(TypeError):
        segment < "invalid"
    # Test greater than
    with self.assertRaises(TypeError):
        segment > "invalid"
        
class TestTimeStampedSequence(unittest.TestCase):
    def setUp(self):
        self.sequence = TimeStampedSequence([
            TimeStampedSegment(0.0, 1.0, "Hello"),
            TimeStampedSegment(1.0, 2.0, "World")
        ])
    
    def test_initialization(self):
        self.assertEqual(str(self.sequence), "[Hello World (0.000 - 2.000)]")
    
    def test_from_tuples(self):
        seq = TimeStampedSequence.from_tuples([(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
        self.assertEqual(str(seq), "[Hello World (0.000 - 2.000)]")
    
    def test_concatenate(self):
        result = self.sequence.concatenate()
        self.assertEqual(repr(result), "(0.000, 2.000, 'Hello World')")
    
    def test_get_text(self):
        self.assertEqual(self.sequence.get_text(), "Hello World")
        self.assertEqual(self.sequence.get_text(sep="-"), "Hello-World")
    
    def test_shift(self):
        self.sequence.shift(1.0)
        self.assertEqual(str(self.sequence), "[Hello World (1.000 - 3.000)]")
    
    def test_long_sequence(self):
        long_seq = TimeStampedSequence([
            TimeStampedSegment(i, i+1, f"Word{i}") 
            for i in range(10)
        ])
        self.assertIn("…", str(long_seq))
        self.assertIn("10 words", str(long_seq))

if __name__ == '__main__':
    unittest.main()
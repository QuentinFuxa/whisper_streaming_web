from typing import List

import logging
logger = logging.getLogger(__name__)





class TimeStampedSegment(str):
    """
    Represents a segment of text with start and end timestamps.

    Attributes:
        start (float): The start time of the segment.
        end (float): The end time of the segment.
        attr (dict): Additional attributes for the segment.
    """
    def __new__(cls, start: float, end: float, text: str):
        instance = super().__new__(cls, text)
        instance.start = start
        instance.end = end
        return instance

    def __str__(self):
        return f'{super().__str__()} ({self.start:.3f} - {self.end:.3f})'
    
    def __repr__(self):
        return f'({self.start:.3f}, {self.end:.3f}, {super().__repr__()})'
    
    def shift(self, shift: float) -> None:
        """
        Shifts the segment by a given amount of time.

        Args:
            shift (float): The amount of time to shift the segment.

        Returns:
            TimeStampedSegment: A new segment shifted by the given amount of time.

        Example:
            >>> segment = TimeStampedSegment(0.0, 1.0, "Hello")
            >>> segment.shift(1.0)
            >>> segment
            (1.000, 2.000, 'Hello')
        """
        self.start += shift
        self.end += shift
    
    def __eq__(self, other):
        if isinstance(other, TimeStampedSegment):
            return (
                self.start == other.start and 
                self.end == other.end and 
                super().__eq__(other))
        elif isinstance(other, str):
            return super().__eq__(other)
        else:
            return False
        
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.end < other
        elif isinstance(other, TimeStampedSegment):
            return self.end < other.start
        else:
            raise TypeError(f"unsupported operand type(s) for <: '{type(self).__name__}' and '{type(other).__name__}'")

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return other < self.start 
        elif isinstance(other, TimeStampedSegment):
            return other.end < self.start
        else:
            raise TypeError(f"unsupported operand type(s) for >: '{type(self).__name__}' and '{type(other).__name__}'")


    
    def __add__(self, other) -> 'TimeStampedSegment':
        """
        Concatenates the segment with another segment or a string.
        If the other object is a number, it will shift the segment by that amount of time.

        """
        if isinstance(other, (int, float)):
            return TimeStampedSegment(self.start + other, self.end +other, super().__str__())
        elif isinstance(other, str):
            return TimeStampedSegment(self.start, self.end, super().__str__() + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")




class TimeStampedSequence(list):
    """
    Represents a collection of TimeStampedSegment instances.
    """
    def __init__(self, segments: List[TimeStampedSegment] = None,sep=None):
        segments = segments if segments is not None else []
        super().__init__(segments)
        
        if sep is None:
            self.infer_sep()

    def from_tuples(tuples: List[tuple]) -> 'TimeStampedSequence':
        """
        Creates a TimeStampedSequence from a list of tuples.

        Args:
            tuples (List[tuple]): A list of tuples where each tuple contains a start time, end time, and text.

        Returns:
            TimeStampedSequence: A new TimeStampedSequence instance.

        Example:
        >>> sequence = TimeStampedSequence.from_tuples([(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
        >>> str(sequence)
        '[Hello World (0.000 - 2.000)]'
        """
        return TimeStampedSequence([TimeStampedSegment(*t) for t in tuples])


    def infer_sep(self) -> None:

        n_tailing_spaces = 0
        for word in self:
            if word.endswith(' '):
                n_tailing_spaces += 1


        if n_tailing_spaces / len(self) > 0.5:
            sep=""
        else:
            sep= " "

        logger.debug(f"{n_tailing_spaces} words of {len(self)} have tailing spaces I will use '{sep}' to join them")
        
        self.sep = sep

    def concatenate(self) -> TimeStampedSegment:
        """
        Concatenates all segments in the sequence into a single segment.

        Returns:
            TimeStampedSegment: A single segment containing all text from the sequence.

        Example:
            >>> sequence = TimeStampedSequence.from_tuples([(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
            >>> sequence.sep==" "
            True
            >>> sequence.concatenate()
            (0.000, 2.000, 'Hello World')
            >>> sequence = TimeStampedSequence.from_tuples([(0.0, 1.0, "Hello "), (1.0, 2.0, "World! ")])
            >>> sequence.sep==""
            True
            >>> sequence.concatenate()
            (0.000, 2.000, 'Hello World! ')
        """
        if len(self) == 0:
            return TimeStampedSegment()
        
        

        return TimeStampedSegment(self[0].start, self[-1].end, self.sep.join(segment for segment in self))
    
    def get_text(self,sep=None) -> str:
        """
        Returns the text of all segments in the sequence.

        Returns:
            str: The text of all segments in the sequence.

        Example:
            >>> sequence = TimeStampedSequence.from_tuples([(0.0, 1.0, "Hello"), (1.0, 2.0, "World")])
            >>> sequence.get_text()
            'Hello World'
            >>> sequence.get_text(sep="_")
            'Hello_World'
        """
        if sep is None:
            sep = self.sep

        
        return sep.join(segment for segment in self)
    
    def shift(self, shift: float) -> None:
        """
        Shifts all segments in the sequence by a given amount of time.

        Args:
            shift (float): The amount of time to shift the segments.

        Returns:
            TimeStampedSequence: A new sequence with all segments shifted by the given amount of time.

        Example:
            >>> sequence = TimeStampedSequence([TimeStampedSegment(0.0, 1.0, "Hello"), TimeStampedSegment(1.0, 2.0, "World")])
            >>> sequence.shift(1.0)
            >>> sequence
            [(1.000, 2.000, 'Hello'), (2.000, 3.000, 'World')]
        """
        for segment in self:
            segment.shift(shift)
    
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):

        if len(self) == 0:
            return "[]"
        if len(self) <= 5:
            return f"[{self.get_text()} ({self[0].start:.3f} - {self[-1].end:.3f})]"

        return "[{first}…{last} ({n} words {start:.3f} - {end:.3f})]".format(
            n= len(self),
            start= self[0].start,
            end= self[-1].end,
            first= self.sep.join((self[0],self[1])),
            last= self.sep.join((self[-2],self[-1]))
        )







if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod(verbose=True)
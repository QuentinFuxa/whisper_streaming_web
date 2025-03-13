from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple
from itertools import groupby

import logging
logger = logging.getLogger(__name__)

@dataclass
class TimedText:
    start: Optional[float]
    end: Optional[float]
    text: Optional[str] = ''
    speaker: Optional[int] = -1
    probability: Optional[float] = None

    def shift(self, offset: float) -> None:
        self.start += offset
        self.end += offset

@dataclass
class ASRToken(TimedText):
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, self.probability)

@dataclass
class Sentence(TimedText):
    pass

@dataclass
class Transcript(TimedText):
    pass

@dataclass
class SpeakerSegment(TimedText):
    pass


class TimedList(list):
    """
    Represents a collection of TimedText instances.
    """
    def __init__(self, words: List[TimedText] = None,sep:Optional[str]=None):
        if words is None:
            words = []

        for word in words:
            if not isinstance(word, TimedText):
                raise ValueError(f"Expect a TimedText instance, got {type(word)}")

        super().__init__(words)
        
        self.sep= sep


    def infer_sep(self) -> None:
        if len(self) == 0:
            return " "

        n_tailing_spaces = 0
        for word in self:
            if word.text.endswith(' ') or word.text.startswith(' '):
                n_tailing_spaces += 1


        if n_tailing_spaces / len(self) > 0.8:
            sep=""
        else:
            sep= " "
        
        logger.debug(f"{n_tailing_spaces} words of {len(self)} have tailing spaces I will use '{sep}' to join them")
        self.sep = sep

    def concatenate(self, sep: Optional[str] = None, offset: float = 0) -> TimedText:
        """
        Concatenates all segments in the sequence into a single segment.

        Returns:
            TimedText: A single segment containing all text from the sequence.

        Example:
            >>> sequence = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "World")])
            >>> sequence.concatenate() == TimedText(start=0.0, end=2.0, text='Hello World')
            True
            >>> sequence.sep==" "
            True
        """

        if len(self) == 0:
            return TimedText(None, None)
        

        text = self.get_text(sep)
        probability = sum(word.probability for word in self if word.probability) / len(self)
        if probability==0:
            probability=None

        speaker= self[0].speaker
        if not all(word.speaker==speaker for word in self):
            logger.warning(f"Not all words in the sequence are not from the same speaker: {self}")
            speaker=-1

        start = offset + self[0].start
        end = offset + self[-1].end

        return TimedText(start, end, text, probability=probability, speaker=speaker)

    
    def get_text(self,sep: Optional[str] = None) -> str:
        """
        Returns the text of all segments in the sequence.

        Returns:
            str: The text of all segments in the sequence.

        Example:
            >>> sequence = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "World")])
            >>> sequence.get_text()
            'Hello World'
        """

        if len(self) == 0:
            return ""

        if sep is None:
            if self.sep is None:
                self.infer_sep()
            sep = self.sep
        elif self.sep is None:
            self.sep = sep
        
        return sep.join(word.text for word in self)
    
    def shift(self, shift: float) -> None:
        """
        Shifts all segments in the sequence by a given amount of time.

        Args:
            shift (float): The amount of time to shift the segments.

        Returns:
            TimedList: A new sequence with all segments shifted by the given amount of time.

        Example:
            >>> sequence = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "World")])
            >>> sequence.shift(1.0)
            >>> sequence == [TimedText(start=1.0, end=2.0, text='Hello'), TimedText(start=2.0, end=3.0, text='World')]
            True
        """
        for segment in self:
            segment.shift(shift)

    def extend(self, other: List[TimedText] | 'TimedList'):
        """
        Extend the sequence with another sequence of TimedText instances.
        """
        assert isinstance(other, TimedList), "Expect a TimedList to extend"
        
        if len(other) == 0:
            return
        elif len(self) == 0:
            super().extend(other)
            self.sep = other.sep
            return
        else:
            if (self.sep != other.sep):
                if self.sep is None:
                    self.sep = other.sep
                elif other.sep is not None:
                    logger.warning(f"Two TimedLists with different separators are being extended: '{self.sep}' != '{other.sep}' use ' ' in future")
                    self.sep = ' '

            super().extend(other)
        
        

    
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):
        """
        Returns a string representation of the TimedList.
        Example:
            >>> tokens = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "all"), TimedText(2.0, 3.0, "Together!")])
            >>> print(tokens)
            [Hello all Together! (0.000 - 3.000)]
        """

        if len(self) == 0:
            return "[]"

        else:
            combined_text = self.concatenate()
            return f"[{combined_text.text} ({combined_text.start:.3f} - {combined_text.end:.3f})]"



    def __getitem__(self, index: int) -> TimedText:
        if isinstance(index, slice):
            return TimedList(super().__getitem__(index), sep=self.sep)
        else:
            return super().__getitem__(index)

    def __add__(self, other: List[TimedText] | 'TimedList') -> 'TimedList':
        """
        Add two TimedLists together.
        """
        assert isinstance(other, TimedList), "Expect a TimedList to add"
        return TimedList(super().__add__(other), sep=self.sep)

    def split_at(self, time: float, edge_word_goes_left: bool) -> Tuple['TimedList', 'TimedList']:
        """
        Split the sequence at a given time. 
        Assuming time is in the middle of a word, the flag edge_word_goes_left decides which side the word goes to.
        
        Returns a tuple of two TimedLists.

        Example:
            >>> greeting = TimedList([TimedText(0.0, 1.0, "Hello"), TimedText(1.0, 2.0, "all"), TimedText(2.0, 3.0, "Together!")])
            >>> before,after= greeting.split_at(1.5, edge_word_goes_left=True)
            >>> before.get_text()
            'Hello all'
            >>> before,after= greeting.split_at(1.5, edge_word_goes_left=False)
            >>> before.get_text()
            'Hello'
        """
        # word at i goes to the right list
        if edge_word_goes_left:
            # find the first word that starts after the time
            for i, word in enumerate(self):
                if word.start >= time:
                    break
            else:
                return self, TimedList([])

           
            
            
        else: # edge_word_goes_right
            # find the first word that ends after the time
            # word at i goes to the right list
            for i, word in enumerate(self):
                if word.end >= time:
                    break
            else:
                return self, TimedList([])

        return self[:i], self[i:]

        


    def split_by_speaker(self) -> List['TimedList']:
        """
        Split the sequence by speaker.
        Returns a list of TimedLists, each containing tokens from a single speaker.

        Example:
            >>> tokens = TimedList([TimedText(0.0, 1.0, "Hello", speaker=0),TimedText(1.0,2.0,"Everybody.",speaker=0), TimedText(2.0, 3.0, "Bonjour!", speaker=1)])
            >>> split_by_speaker = tokens.split_by_speaker()
            >>> len(split_by_speaker)
            2
            >>> split_by_speaker[0].get_text()
            'Hello Everybody.'
            >>> split_by_speaker[1].get_text()
            'Bonjour!'
        """
        return [TimedList(list(word_iterator), sep=self.sep) for _,word_iterator in groupby(self, key=lambda x: x.speaker)]
        



    def split_to_sentences(self, sentence_splitter: Optional[Callable[[str], List[str]] | Callable[[List[str]], List[str]]] = None) -> List['TimedList']:
            """
            Converts a list of tokens to a list of Sentence objects using the provided
            sentence tokenizer.
            """
            if len(self) == 0:
                return TimedList([])


            split_by_speaker = self.split_by_speaker()

            if len(split_by_speaker) > 1:
                logger.debug(f"Sentence splitting found {len(split_by_speaker)} speakers. Try to split last speaker's utterance")


            if sentence_splitter is None:
                logger.warning("No sentence splitter provided, concatenating tokens")
                return split_by_speaker

            
            last_speaker_utterance = split_by_speaker.pop(-1)
   
            last_speaker_sentences = run_sentence_splitter(last_speaker_utterance, sentence_splitter)

            return split_by_speaker + last_speaker_sentences



def run_sentence_splitter(input_sentences: 'TimedList',sentence_splitter: Callable[[str], List[str]] | Callable[[List[str]], List[str]]) -> List['TimedList']:
    """
    Run the sentence splitter on the transcript.
    """
    
    if len(input_sentences) == 0:
        return []
    
    assert isinstance(input_sentences, TimedList), "Expect a TimedList to run the sentence splitter"

    full_text = input_sentences.get_text()

    try:
        # most tokenizers (e.g., MosesSentenceSplitter) expect a list input.
        sentence_texts = sentence_splitter([full_text])
    except Exception as e:   
        try:
            sentence_texts = sentence_splitter(full_text)
        except Exception as e2:
            raise ValueError("Tokenization failed") from e

    # Match output of sentence splitter to the input tokens
    sentences= []
    token_index = 0

    for sent_text in sentence_texts:
        sent_text = sent_text.strip()
        if not sent_text:
            continue
        sent_tokens = TimedList([],sep=input_sentences.sep)

        accumulated = ""
        # Accumulate tokens until roughly matching the length of the sentence text.
        while token_index < len(input_sentences) and len(accumulated) < len(sent_text):
            token = input_sentences[token_index]
            accumulated = (accumulated + input_sentences.sep + token.text).strip() if accumulated else token.text
            sent_tokens.append(token)
            token_index += 1
        if sent_tokens:
            sentences.append(sent_tokens)
    return sentences



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
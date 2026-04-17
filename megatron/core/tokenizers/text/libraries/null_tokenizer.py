# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict


class NullTokenizer:
    """
    Synthetic tokenizer for performance benchmarking and debugging

    Args:
        vocab_size: vocabulary size for embedding
    """

    def __init__(self, vocab_size):
        """ """
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def text_to_ids(self, text):
        """Converts text to ids."""
        return [int(x) for x in text.split(' ')]

    def text_to_tokens(self, text):
        """Converts text to tokens."""
        return text.split(' ')

    def ids_to_text(self, ids):
        """Converts ids to text."""
        text = [str(x) for x in ids]
        return ' '.join(text)

    def tokens_to_text(self, tokens):
        """Converts tokens to text."""
        return ' '.join(tokens)

    def tokens_to_ids(self, tokens):
        """Converts tokens to ids."""
        return [int(x) for x in tokens]

    def ids_to_tokens(self, ids):
        """Converts ids to tokens."""
        return [str(x) for x in ids]

    def add_special_tokens(self, special_tokens=None):
        """Null tokenizer does not maintain a mutable special-token table."""
        return None

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Returns offsets."""
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def unique_identifiers(self) -> OrderedDict:
        """Property required for use with megatron-core datasets."""
        return OrderedDict({"class": f"{type(self).__module__}.{type(self).__qualname__}"})

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        """ """
        raise NotImplementedError

    @property
    def inv_vocab(self):
        """ """
        raise NotImplementedError

    @property
    def cls(self):
        """Returns cls token."""
        return -1

    @property
    def sep(self):
        """Returns sep token."""
        return -1

    @property
    def pad(self):
        """Returns pad token."""
        return 0

    @property
    def pad_id(self):
        """Returns pad token id."""
        return self.pad

    @property
    def bos(self):
        """Returns bos token."""
        return None

    @property
    def bos_id(self):
        """Returns bos token id."""
        return self.bos

    @property
    def eos(self):
        """Returns eos token."""
        return self.eod

    @property
    def eos_id(self):
        """Returns eos token id."""
        return self.eod

    @property
    def unk(self):
        """Returns unk token."""
        return self.pad

    @property
    def unk_id(self):
        """Returns unk token id."""
        return self.unk

    @property
    def mask(self):
        """Returns mask token."""
        return -1

    @property
    def eod(self):
        """Returns eod token."""
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        """ """
        return None

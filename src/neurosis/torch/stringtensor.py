from typing import Literal, Optional, TypeAlias

from torch import ByteTensor, Tensor

StringEncoding: TypeAlias = Literal["ascii", "utf-8", "utf-16-le", "utf-32-le"]


class StringTensor(Tensor):
    def __new__(cls, data, encoding: StringEncoding = "utf-8", pad_to: int = -1, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, data, encoding: StringEncoding = "utf-8", pad_to: int = -1):
        string = str(data)  # make sure it's a string
        self.encoding = encoding  # save the encoding
        string_bytes = [x for x in string.encode(encoding)]  # convert to list of bytes
        byte_len = len(string_bytes)  # get the byte count

        if byte_len < pad_to:
            # null-pad the string to the desired length
            string_bytes += bytes([0 for _ in range(pad_to - len(string_bytes))])
        self.data = ByteTensor(string_bytes)  # convert to tensor of bytes

    def __repr__(self) -> str:
        as_string = str(self)
        str_len = len(as_string)
        return f"StringTensor('{as_string}', len={str_len}, nbytes={len(self)}, encoding={self.encoding}, dtype=torch.uint8)"

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self, encoding: Optional[str] = None):
        return bytes(self.data).decode(encoding or self.encoding).split("\x00", maxsplit=1)[0]

    def tensor(self) -> Tensor:
        return self.data


class StringArray:
    def __init__(
        self,
        strings: list[str],
        encoding: StringEncoding = "utf-8",
        max_len: int = -1,
    ):
        self.encoding = encoding
        if not isinstance(strings, list):
            strings = [strings]
        # encode the strings
        encoded = [x.encode(encoding) for x in strings]
        # find max length for padding
        self.max_len = max([len(x) for x in encoded] + [max_len])
        # build the string tensors with the encoding and padding
        self.strings = [StringTensor(x, encoding=encoding, pad_to=self.max_len) for x in strings]

    def __repr__(self) -> str:
        return f"StringArray({self.strings}, encoding={self.encoding}, max_len={self.max_len})"

    def __getitem__(self, idx: int) -> StringTensor:
        return self.strings[idx]

    def to_list(self) -> list[str]:
        return [x.to_string(encoding=self.encoding) for x in self.strings]

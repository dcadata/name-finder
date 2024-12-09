from enum import Enum


class _GenerationBase:
    def __init__(self, after: int, before: int) -> None:
        self.after = after
        self.before = before

    @property
    def years_as_dict(self) -> dict[str, int]:
        return dict(after=self.after, before=self.before)

    @property
    def after_as_dict(self) -> dict[str, int]:
        return dict(after=self.after)

    @property
    def before_as_dict(self) -> dict[str, int]:
        return dict(before=self.before)

    @property
    def years_as_tuple(self) -> tuple[int, int]:
        return self.after, self.before


class Generation:
    GenZ: _GenerationBase = _GenerationBase(1997, 2012)
    Millennial: _GenerationBase = _GenerationBase(1981, 1996)
    GenX: _GenerationBase = _GenerationBase(1965, 1980)
    BoomerII: _GenerationBase = _GenerationBase(1955, 1964)
    BoomerI: _GenerationBase = _GenerationBase(1946, 1954)
    PostWar: _GenerationBase = _GenerationBase(1928, 1945)
    WWII: _GenerationBase = _GenerationBase(1922, 1927)

    @classmethod
    def combine(cls, *args: _GenerationBase) -> _GenerationBase:
        return _GenerationBase(args[0].after, args[-1].before)


class _GenderBound:
    Feminine: tuple[float, float] = (0, .1)
    Masculine: tuple[float, float] = (.9, 1)
    Neutral: tuple[float, float] = (.3, 1 - .3)
    NeutralBroad: tuple[float, float] = (.2, 1 - .2)
    NeutralBroadest: tuple[float, float] = (.1, 1 - .1)


class _GenderInd(str, Enum):
    Feminine: str = 'f'
    Masculine: str = 'm'
    Neutral: str = 'x'


class Gender:
    Bound = _GenderBound
    Ind = _GenderInd


class _SsaSexInd(str, Enum):
    Female: str = 'f'
    Male: str = 'm'


class _SsaUnisexInd:
    All: str = 'all'
    Total: str = 'total'


class _SsaSexSuffix(str, Enum):
    Female: str = '_f'
    Male: str = '_m'


class SsaSex:
    Ind = _SsaSexInd
    Unisex = _SsaUnisexInd
    Suffix = _SsaSexSuffix
    HueOrderAndPalette = dict(hue_order=Ind, palette=('red', 'blue'))

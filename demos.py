from abc import abstractmethod
from enum import Enum


class _GenerationBase:
    def as_dict(self) -> dict[str, int]:
        return dict(after=self.after, before=self.before)

    def as_tuple(self) -> tuple[int, int]:
        return self.after, self.before

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def after(self) -> int:
        pass

    @property
    @abstractmethod
    def before(self) -> int:
        pass


class _GenerationGenZ(_GenerationBase):
    after: int = 1997
    before: int = 2012


class _GenerationMillennial(_GenerationBase):
    after: int = 1981
    before: int = 1996


class _GenerationGenX(_GenerationBase):
    after: int = 1965
    before: int = 1980


class _GenerationBoomerII(_GenerationBase):
    after: int = 1955
    before: int = 1964


class _GenerationBoomerI(_GenerationBase):
    after: int = 1946
    before: int = 1954


class _GenerationPostWar(_GenerationBase):
    after: int = 1928
    before: int = 1945


class _GenerationWWII(_GenerationBase):
    after: int = 1922
    before: int = 1927


class Generation:
    GenZ: _GenerationBase = _GenerationGenZ()
    Millennial: _GenerationBase = _GenerationMillennial()
    GenX: _GenerationBase = _GenerationGenX()
    BoomerII: _GenerationBase = _GenerationBoomerII()
    BoomerI: _GenerationBase = _GenerationBoomerI()
    PostWar: _GenerationBase = _GenerationPostWar()
    WWII: _GenerationBase = _GenerationWWII()


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


class _SsaSexPalette(str, Enum):
    Female: str = 'red'
    Male: str = 'blue'


class SsaSex:
    Ind = _SsaSexInd
    Unisex = _SsaUnisexInd
    Suffix = _SsaSexSuffix
    Palette = _SsaSexPalette

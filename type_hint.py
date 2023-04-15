# 型ヒント
from typing import (
    Any,
    NewType, # >= python 3.5.2
    TypeVar,
    Optional, # Optional[int] is same as Union[int, None]
    Union,
    AnyStr,
    Generator,
    Text, # >= Python 3.5.2
    Annotated, # >= Python 3.9
    TypedDict, # >= Python 3.8
    Final, # >= Python 3.8
    ClassVar, # >= Python 3.5.3
    Literal, # >= Python 3.8
    Callable, # 非推奨 Python 3.9.x
    NoReturn, # >= python 3.6.2
    # Self, # 自分自身のオブジェクト]
    IO, # 非推奨 Python 3.8
    TextIO, # 非推奨 Python 3.8
    BinaryIO, # 非推奨 Python 3.8
    Pattern, # 非推奨 Python 3.9
    Match, # 非推奨 Python 3.9
)

# ジェネリック版の派生クラス
from typing import (
    Generic, # ジェネリック型のための抽象基底クラス
    NamedTuple, 
    Hashable, 
    List, # 非推奨 Python 3.9 same as list -> recommend Iterable, Sequence
    Dict, # 非推奨 Python 3.9 same as dict -> recommend Mapping
    Tuple, # 非推奨 Python 3.9
    Set, # 非推奨 Python 3.9 same as set
    FrozenSet, # 非推奨 Python 3.9 same as frozenset
    DefaultDict, # 非推奨 Python 3.9 >= Python 3.5.2 same as collections.defaultdict
    OrderedDict, # 非推奨 Python 3.9 >= Python 3.7.2 same as collections.OrderedDict
    ChainMap, # 非推奨 Python 3.9 >= Python 3.5.4 same as collections.ChainMap
    Counter, # 非推奨 Python 3.9 >= Python 3.5.4 same as collections.Counter
    Deque, # 非推奨 Python 3.9 >= Python 3.5.4 same as collections.Deque
    Reversible, # 非推奨 Python 3.9 >= Python 3.5.4 same as collections.Reversible
)

# ジェネリク版の基底クラス
from typing import (
    AbstractSet, # 非推奨 Python 3.9 same as collections.abc.Set
    ByteString, # 非推奨 Python 3.9 same as collection.abc.ByteString
    Collection, # 非推奨 Python 3.9 same as collections.abc.Collection
    Container, # 非推奨 Python 3.9 same as collections.abc.Container
    ItemsView, # 非推奨 Python 3.9 same as collections.abc.ItemsView
    KeysView, # 非推奨 Python 3.9 same as collections.abc.KeysView
    Mapping, # 非推奨 Python 3.9 same as collections.abc.Mapping
    MappingView, # 非推奨 Python 3.9 same as collections.abc.MappingView
    MutableMapping, # 非推奨 Python 3.9 same as collections.abc.MutableMapping
    MutableSequence, # 非推奨 Python 3.9 same as collections.abc.MutableSequence
    MutableSet, # 非推奨 Python 3.9 same as collections.abc.MutableSet
    Sequence, # 非推奨 Python 3.9 same as collections.abc.Sequence
    ValuesView, # 非推奨 Python 3.9 same as collections.abc.ValuesView
    Iterable, # 非推奨 Python 3.9 same as collections.abc.Iterable
    Iterator, # 非推奨 Python 3.9 same as collections.abc.Iterator
    Sized, # same as collections.abc.Sized
)

# 非同期型
from typing import (
    Coroutine, # >= Python 3.5.3 same as collections.abc.Coroutine
    AsyncGenerator, # 非推奨 Python 3.9 >= Python 3.6.1 same as collections.abc.AsyncGenerator
    AsyncIterable, #  非推奨 Python 3.9 >= Python 3.5.2 same as collections.abc.AsyncIterable
    AsyncIterator, #  非推奨 Python 3.9 >= Python 3.5.2 same as collections.abc.AsyncIterator
    Awaitable, # 非推奨 Python 3.9 >= Python 3.5.2 same as collections.abc.Awaitable
)

# コンテキストマネージャ型
from typing import (
    ContextManager, # 非推奨 Python 3.9 >= Python 3.5.4 same as contextlib.AbstractContextManager
    AsyncContextManager, # 非推奨 Python 3.9 >= Python 3.5.4 same as contextlib.AbstractAsyncContextManager
)

# プロトコル型の具体クラス
from typing import (
    Protocol, # >= Python 3.8
    runtime_checkable, # >= Python 3.8
)

# プロトコル型の抽象クラス
from typing import (
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex, # >= Python 3.8
    SupportsInt,
    SupportsRound,
)

# 関数とデコレータ
from typing import (
    cast,
    overload,
    final, # >= Python 3.8
    no_type_check,
    no_type_check_decorator,
    # type_check_only,
)

# その他
from typing import (
    get_type_hints,
    get_args,
    get_origin, # >= Python 3.8
    ForwardRef, # >= Python 3.7.4
)

# 定数
from typing import (
    TYPE_CHECKING
)
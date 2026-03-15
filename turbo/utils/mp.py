from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")

class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, item: T) -> None:
        event = msgpack.packb(self.encoder(item), use_bin_type=True)
        self.socket.send(event, copy=False)

    def shutdown(self) -> None:
        self.socket.close()
        self.context.term()

class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, item: T) -> None:
        event = msgpack.packb(self.encoder(item), use_bin_type=True)
        self.socket.send(event, copy=False)

    def shutdown(self) -> None:
        self.socket.close()
        self.context.term()

class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv(copy=False)
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw_bytes: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw_bytes, raw=False))

    def empty(self) -> bool:
        return not self.socket.poll(timeout=0)

    def shutdown(self) -> None:
        self.socket.close()
        self.context.term()

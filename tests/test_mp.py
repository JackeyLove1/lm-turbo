import socket
import time

from turbo.utils.mp import ZmqAsyncPushQueue, ZmqPullQueue, ZmqPushQueue


def _free_tcp_addr() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
    return f"tcp://{host}:{port}"


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition was not met before timeout")


def test_zmq_push_queue_put_and_pull_queue_get():
    addr = _free_tcp_addr()
    pull_queue = ZmqPullQueue(addr=addr, create=True, decoder=lambda data: data["value"])
    push_queue = ZmqPushQueue(
        addr=addr,
        create=False,
        encoder=lambda value: {"value": value},
    )

    try:
        assert pull_queue.empty()

        push_queue.put("payload")

        _wait_until(lambda: not pull_queue.empty())
        assert pull_queue.get() == "payload"
        assert pull_queue.empty()
    finally:
        push_queue.shutdown()
        pull_queue.shutdown()


def test_zmq_pull_queue_get_raw_and_decode():
    addr = _free_tcp_addr()
    pull_queue = ZmqPullQueue(addr=addr, create=True, decoder=lambda data: data["value"] * 2)
    push_queue = ZmqPushQueue(
        addr=addr,
        create=False,
        encoder=lambda value: {"value": value},
    )

    try:
        push_queue.put(21)

        _wait_until(lambda: not pull_queue.empty())
        raw_message = pull_queue.get_raw()

        assert isinstance(raw_message, bytes)
        assert pull_queue.decode(raw_message) == 42
        assert pull_queue.empty()
    finally:
        push_queue.shutdown()
        pull_queue.shutdown()


def test_zmq_async_push_queue_can_send_to_pull_queue():
    addr = _free_tcp_addr()
    pull_queue = ZmqPullQueue(addr=addr, create=True, decoder=lambda data: data["value"])
    push_queue = ZmqAsyncPushQueue(
        addr=addr,
        create=False,
        encoder=lambda value: {"value": value},
    )

    try:
        push_queue.put({"nested": "value"})

        _wait_until(lambda: not pull_queue.empty())
        assert pull_queue.get() == {"nested": "value"}
    finally:
        push_queue.shutdown()
        pull_queue.shutdown()

#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Unix-socket LJ client for one RPQA run.

The framing is the single-structure path in ``drivers/py/driver.py`` of
afiorent/i-pi ``4e86f5fd``. Each POSDATA is one force evaluation, which is
how that client is called for a bead during dynamics and during the
``rpqa`` local optimisation. The call that reaches the budget is evaluated
and counted, then the socket is closed without a reply.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from lj_potential import evaluate
from meter import Meter


def main() -> None:
    ipi_root = os.environ["IPI_ROOT"]
    sys.path.insert(0, os.path.join(ipi_root, "drivers", "py"))
    sys.path.insert(0, ipi_root)
    from driver import HDRLEN, Message, recv_data, send_data

    address = os.environ["RPQA_ADDRESS"]
    prefix = os.environ.get("RPQA_SOCKET_PREFIX", "/tmp/ipi_")
    budget = int(os.environ["RPQA_BUDGET"])
    meter = Meter(os.environ["RPQA_METER"])
    import socket

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(prefix + address)
    def recv_exact(n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = sock.recv(n - len(buf))
            if not part:
                return b""
            buf += part
        return buf

    cell = np.zeros((3, 3), float)
    icell = np.zeros((3, 3), float)
    pos = np.zeros(0, float)
    ready = False
    have = False
    pot = 0.0
    force = np.zeros(0, float)
    vir = np.zeros((3, 3), float)
    nat = 0
    stop_after_reply = False
    while True:
        header = recv_exact(HDRLEN)
        if header == b"":
            return
        if header == Message("STATUS"):
            if not ready:
                sock.sendall(Message("NEEDINIT"))
            elif have:
                sock.sendall(Message("HAVEDATA"))
            else:
                sock.sendall(Message("READY"))
        elif header == Message("INIT"):
            recv_data(sock, np.int32())
            initlen = recv_data(sock, np.int32())
            recv_data(sock, np.empty(initlen, dtype="S1"))
            ready = True
        elif header == Message("POSDATA"):
            if budget > 0 and meter.at_budget(budget):
                return
            cell = recv_data(sock, cell)
            icell = recv_data(sock, icell)
            nat = int(recv_data(sock, np.int32()))
            if len(pos) == 0:
                pos = np.zeros((nat, 3), float)
                force = np.zeros((nat, 3), float)
            pos = recv_data(sock, pos)
            mark = time.perf_counter_ns()
            pot, force, vir = evaluate(cell, pos)
            under = meter.charge(time.perf_counter_ns() - mark, budget)
            have = True
            if budget > 0 and not under:
                # The evaluation is counted. Reply so the server is not left
                # in a 60 s select, then stop before the next geometry.
                stop_after_reply = True
        elif header == Message("GETFORCE"):
            sock.sendall(Message("FORCEREADY"))
            send_data(sock, np.float64(pot))
            send_data(sock, np.int32(nat))
            send_data(sock, force)
            send_data(sock, vir)
            send_data(sock, np.int32(0))
            have = False
            if stop_after_reply:
                return
        elif header == Message("EXIT"):
            return


if __name__ == "__main__":
    main()

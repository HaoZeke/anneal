#!/usr/bin/env python3
"""Regenerate the OpenCPMD pbc3 references in CppCore/tests/data/d2cpmd/.

The mapping is subroutine pbc3 in src/pbc_utils.mod.F90 of OpenCPMD at
commit 062582b7cfd832d36f88f504cd08e4ead42eb404, the tree cpmdc builds
against. subroutine vdw in src/vdw_utils.mod.F90 calls it (through
subroutine pbc) with m = 1 and then, for ALL DFT-D2, the Grimme branch.
nxvdw defaults to 0 in subroutine empvdwin, so that one image is the
whole sum. The script compiles those subroutines, plus subroutine rot
which pbc3 calls, and evaluates the Grimme pair on the image.

The cells are latgen's (subroutine latgen in src/latgen_utils.mod.F90),
apbc is the one subroutine setsc in src/setsc_utils.mod.F90 stores, and
v1crys, v2crys and v3crys are the ones subroutine cry in the same file
stores for symmetry 12. Every code latgen builds and pbc3 handles has a
cell here: 1 and 8 fold per axis (labels 5 and 15), 2 is face centred
cubic (label 10), 4 and 14 with alpha and beta at 90 degrees share the
shear branch (label 25), 12 is the monoclinic branch (label 20, which no
other code reaches), and 3, 5, 6 and 7 are the general branch (label 99).
Symmetry 12 describes the same shape as 14 with alpha and beta at 90
degrees, so a box cannot name it.

Each file also records the Grimme energy of the Euclidean nearest image
of the same pair and of the nearest integer fractional image, from the
same compiled kernel, so a test can require the mappings to differ. Lengths in the files are bohr times
0.529177210859, CPMD's own bohr.

Usage: regen_d2_cpmd_pbc3.py --src PATH/TO/CPMD [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path

COMMIT = "062582b7cfd832d36f88f504cd08e4ead42eb404"
CPMD_BOHR = 0.529177210859  # 1 / fbohr in OpenCPMD's cnst.mod.F90

# Two oxygen atoms. The second sits at a fractional combination of the
# lattice vectors, including shifts of more than one cell. BLYP s6.
CASES = (
    {
        "name": "pbc3_hexagonal",
        "ibrav": 4,
        "celldm": (6.0, 0.0, 10.0 / 6.0, 0.0, 0.0, 0.0),
        # Clear of a half period, where pbc3's dint flips. This image is not
        # the Euclidean one and not the nearest integer one.
        "fractional": (1.6643902557724974, -1.6638963493351189, 1.753272330076137),
    },
    {
        "name": "pbc3_fcc",
        "ibrav": 2,
        "celldm": (8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        # Face centred cubic pbc3 matches the Euclidean image and not the
        # nearest integer one. The fractional part is several cells out.
        "fractional": (3.402548652475856, 3.2766108679397057, 2.3230221790953633),
    },
    {
        "name": "pbc3_monoclinic",
        "ibrav": 14,
        "celldm": (5.0, 6.5 / 5.0, 8.0 / 5.0, 0.0, 0.0, math.cos(math.radians(70.0))),
        "fractional": (1.365075927403026, 1.482151284990993, 0.682173191634714),
    },
    {
        "name": "pbc3_cubic",
        "ibrav": 1,
        "celldm": (7.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "fractional": (2.37, -1.61, 3.18),
    },
    {
        "name": "pbc3_orthorhombic",
        "ibrav": 8,
        "celldm": (6.0, 1.2, 1.5, 0.0, 0.0, 0.0),
        "fractional": (1.71, -2.36, 0.83),
    },
    {
        "name": "pbc3_bcc",
        "ibrav": 3,
        "celldm": (8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "fractional": (1.43, -0.77, 2.31),
    },
    {
        "name": "pbc3_rhombohedral",
        "ibrav": 5,
        "celldm": (7.0, 0.0, 0.0, 0.3, 0.0, 0.0),
        "fractional": (0.83, -1.41, 1.67),
    },
    {
        "name": "pbc3_tetragonal",
        "ibrav": 6,
        "celldm": (6.0, 0.0, 1.4, 0.0, 0.0, 0.0),
        "fractional": (1.62, 2.43, -1.28),
    },
    {
        "name": "pbc3_body_centred_tetragonal",
        "ibrav": 7,
        "celldm": (7.0, 0.0, 1.3, 0.0, 0.0, 0.0),
        "fractional": (1.37, -0.66, 0.91),
    },
    {
        "name": "pbc3_monoclinic12",
        "ibrav": 12,
        "celldm": (5.0, 6.5 / 5.0, 8.0 / 5.0, math.cos(math.radians(70.0)), 0.0, 0.0),
        "fractional": (1.365075927403026, 1.482151284990993, 0.682173191634714),
    },
    {
        # a1 - a2 is shorter than a1, so the two shortest images are
        # swapped in sort2's order and v1crys and v3crys change places.
        "name": "pbc3_monoclinic12_skew",
        "ibrav": 12,
        "celldm": (5.0, 6.0 / 5.0, 7.0 / 5.0, 0.8, 0.0, 0.0),
        "fractional": (-1.3975, 0.5394, 1.4722),
    },
)


def extract_subroutine(text: str, name: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    begin = re.compile(rf"^\s*SUBROUTINE\s+{name}\b", re.IGNORECASE)
    end = re.compile(rf"^\s*END\s+SUBROUTINE\s+{name}\b", re.IGNORECASE)
    for i, line in enumerate(lines):
        if begin.search(line):
            start = i
            break
    if start is None:
        raise SystemExit(f"missing subroutine {name}")
    for j in range(start + 1, len(lines)):
        if end.search(lines[j]):
            return "".join(lines[start : j + 1])
    raise SystemExit(f"missing END SUBROUTINE {name}")


def extract_parameter(text: str, name: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.search(rf"PARAMETER\s*::\s*{name}\b", line):
            start = i
            break
    if start is None:
        raise SystemExit(f"missing parameter {name}")
    if "/)" in lines[start] or not lines[start].rstrip().endswith("&"):
        return lines[start]
    blob = [lines[start]]
    for line in lines[start + 1 :]:
        blob.append(line)
        if "/)" in line:
            return "".join(blob)
    raise SystemExit(f"parameter {name} does not close")


def require_lines(text: str, needles: tuple[str, ...], where: str) -> None:
    for needle in needles:
        if needle not in text:
            raise SystemExit(f"{where} no longer contains {needle}")


def latgen(
    ibrav: int, celldm: tuple[float, ...]
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Lattice rows, apbc and v1crys..v3crys, from latgen, setsc and cry."""
    a = celldm[0]
    no_v = [[0.0] * 4 for _ in range(3)]
    no_ap = [0.0, 0.0, 0.0, 0.0]
    sr3 = math.sqrt(3.0)
    if ibrav == 1:
        return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], [a / 2.0, 0.0, 0.0, 0.0], no_v
    if ibrav == 2:
        term = a / 2.0
        h = [[-term, 0.0, term], [0.0, term, term], [-term, term, 0.0]]
        ap = [a * math.sqrt(2.0) / 4.0, 0.0, 0.0, 0.0]
        return h, ap, no_v
    if ibrav == 3:
        t = a / 2.0
        return [[t, t, t], [-t, t, t], [-t, -t, t]], no_ap, no_v
    if ibrav == 4:
        c = a * celldm[2]
        h = [[a, 0.0, 0.0], [-a / 2.0, a * sr3 / 2.0, 0.0], [0.0, 0.0, c]]
        ap = [a / 2.0, a * sr3 / 4.0, c / 2.0, sr3 / 3.0]
        return h, ap, no_v
    if ibrav == 5:
        term1 = math.sqrt(1.0 + 2.0 * celldm[3])
        term2 = math.sqrt(1.0 - celldm[3])
        a2y = math.sqrt(2.0) * a * term2 / sr3
        a2z = a * term1 / sr3
        a1x = a * term2 / math.sqrt(2.0)
        a1y = -a1x / sr3
        h = [[a1x, a1y, a2z], [0.0, a2y, a2z], [-a1x, a1y, a2z]]
        return h, no_ap, no_v
    if ibrav == 6:
        return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a * celldm[2]]], no_ap, no_v
    if ibrav == 7:
        x = a / 2.0
        z = celldm[2] * a / 2.0
        return [[x, -x, z], [x, x, z], [-x, -x, z]], no_ap, no_v
    if ibrav == 8:
        b = a * celldm[1]
        c = a * celldm[2]
        return (
            [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]],
            [a / 2.0, b / 2.0, c / 2.0, 0.0],
            no_v,
        )
    if ibrav == 12:
        sinval = math.sqrt(1.0 - celldm[3] ** 2)
        a2x = a * celldm[1] * celldm[3]
        a2y = a * celldm[1] * sinval
        c = a * celldm[2]
        h = [[a, 0.0, 0.0], [a2x, a2y, 0.0], [0.0, 0.0, c]]
        # setsc: bbbb, cccc, xl(kk) and the sort2 order, then cry.
        xl = [((a - k * a2x) ** 2 + (k * a2y) ** 2, k) for k in range(4)]
        xl.sort(key=lambda item: item[0])  # sort2 is stable for n = 4
        ap = [math.sqrt(xl[0][0]) / 2.0, a * celldm[1] / 2.0, c / 2.0, math.sqrt(xl[1][0]) / 2.0]

        def direction(x: float, y: float) -> list[float]:
            aux = math.sqrt(x * x + y * y)
            return [x, y, x / aux, y / aux]

        k1 = xl[0][1]
        k3 = xl[1][1]
        v = [
            direction(a - k1 * a2x, -k1 * a2y),
            direction(a2x, a2y),
            direction(a - k3 * a2x, -k3 * a2y),
        ]
        return h, ap, v
    if ibrav == 14:
        b = a * celldm[1]
        c = a * celldm[2]
        # alpha = beta = 90 degrees, so celldm(4) and celldm(5) are 0 and
        # setsc keeps the shear branch instead of the general one.
        if abs(celldm[3]) > 1.0e-10 or abs(celldm[4]) > 1.0e-10:
            raise SystemExit("ibrav 14 reference must keep alpha and beta at 90 degrees")
        cg = celldm[5]
        sg = math.sqrt(1.0 - cg * cg)
        h = [[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [0.0, 0.0, c]]
        ap = [a / 2.0, (b * sg) / 2.0, c / 2.0, -cg / sg]
        return h, ap, no_v
    raise SystemExit(f"ibrav {ibrav} is not one of the reference cells")


def cart(s: tuple[float, float, float], h: list[list[float]]) -> list[float]:
    return [sum(s[k] * h[k][a] for k in range(3)) for a in range(3)]


def inverse(h: list[list[float]]) -> list[list[float]]:
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    det = a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)
    inv = 1.0 / det
    return [
        [(e * j - f * i) * inv, (c * i - b * j) * inv, (b * f - c * e) * inv],
        [(f * g - d * j) * inv, (a * j - c * g) * inv, (c * d - a * f) * inv],
        [(d * i - e * g) * inv, (b * g - a * i) * inv, (a * e - b * d) * inv],
    ]


def wrap_axis(x: float, length: float) -> float:
    y = math.fmod(x, length)
    return y - length * math.trunc(y / (0.5 * length))


def euclidean_image(r: list[float], h: list[list[float]]) -> list[float]:
    """The image the old nearest-vector search returned, for the control."""
    hinv = inverse(h)
    s = [wrap_axis(sum(r[a] * hinv[a][k] for a in range(3)), 1.0) for k in range(3)]
    best = cart(tuple(s), h)
    best2 = sum(v * v for v in best)
    for nx in (-1, 0, 1):
        for ny in (-1, 0, 1):
            for nz in (-1, 0, 1):
                cand = cart((s[0] - nx, s[1] - ny, s[2] - nz), h)
                r2 = sum(v * v for v in cand)
                if r2 + 1.0e-12 < best2:
                    best2 = r2
                    best = cand
    return best


def nint(x: float) -> float:
    return math.floor(abs(x) + 0.5) * (1.0 if x >= 0.0 else -1.0)


def nearest_integer_image(r: list[float], h: list[list[float]]) -> list[float]:
    """The image the general branch (label 99) of pbc3 returns."""
    hinv = inverse(h)
    s = [sum(r[a] * hinv[a][k] for a in range(3)) for k in range(3)]
    return cart(tuple(v - nint(v) for v in s), h)


def fortran_source(pbc3: str, rot: str, tables: str) -> str:
    return f"""
module kinds
  implicit none
  integer, parameter :: real_8 = selected_real_kind(14, 200)
end module

module metr
  use kinds
  implicit none
  type metr_t
    real(real_8) :: ht(3,3) = 0.0_real_8
    real(real_8) :: htm1(3,3) = 0.0_real_8
  end type
  type(metr_t) :: metr_com
end module

module bc
  use kinds
  implicit none
  type bc_t
    real(real_8) :: v1crys(4) = 0.0_real_8
    real(real_8) :: v2crys(4) = 0.0_real_8
    real(real_8) :: v3crys(4) = 0.0_real_8
  end type
  type(bc_t) :: bc_com
end module

module pbc_extract
  use kinds
  use metr
  use bc
  implicit none
{tables}
  real(real_8), parameter :: fbohr = 1._real_8/0.529177210859_real_8
  real(real_8), parameter :: a6grim = 20.0_real_8
  real(real_8), parameter :: vdweps = 1.0e-2_real_8
  real(real_8), parameter :: s6blyp = 1.20_real_8
  real(real_8), parameter :: reps = 1.0e-2_real_8
contains
{pbc3}
{rot}
  subroutine set_htm1(ht, htm1)
    real(real_8), intent(in) :: ht(3,3)
    real(real_8), intent(out) :: htm1(3,3)
    real(real_8) :: d
    d = ht(1,1)*(ht(2,2)*ht(3,3) - ht(2,3)*ht(3,2)) &
      - ht(1,2)*(ht(2,1)*ht(3,3) - ht(2,3)*ht(3,1)) &
      + ht(1,3)*(ht(2,1)*ht(3,2) - ht(2,2)*ht(3,1))
    htm1(1,1) = (ht(2,2)*ht(3,3) - ht(2,3)*ht(3,2))/d
    htm1(1,2) = (ht(1,3)*ht(3,2) - ht(1,2)*ht(3,3))/d
    htm1(1,3) = (ht(1,2)*ht(2,3) - ht(1,3)*ht(2,2))/d
    htm1(2,1) = (ht(2,3)*ht(3,1) - ht(2,1)*ht(3,3))/d
    htm1(2,2) = (ht(1,1)*ht(3,3) - ht(1,3)*ht(3,1))/d
    htm1(2,3) = (ht(1,3)*ht(2,1) - ht(1,1)*ht(2,3))/d
    htm1(3,1) = (ht(2,1)*ht(3,2) - ht(2,2)*ht(3,1))/d
    htm1(3,2) = (ht(1,2)*ht(3,1) - ht(1,1)*ht(3,2))/d
    htm1(3,3) = (ht(1,1)*ht(2,2) - ht(1,2)*ht(2,1))/d
  end subroutine set_htm1

  subroutine grimme(n, z, pos, do_pbc, ibrav, ap, energy, force)
    integer, intent(in) :: n, z(:), ibrav
    real(real_8), intent(in) :: pos(:,:), ap(4)
    logical, intent(in) :: do_pbc
    real(real_8), intent(out) :: energy, force(:,:)
    integer :: l, m, a
    real(real_8) :: dx, dy, dz, x2, y2, z2, rlp(3), rlm2, rlm1, rlm6
    real(real_8) :: rpow, rexp, rphi, ffac, vdwst, vdwrm
    energy = 0.0_real_8
    force(:,1:n) = 0.0_real_8
    do l = 1, n
      do m = 1, l
        dx = pos(1,l) - pos(1,m)
        dy = pos(2,l) - pos(2,m)
        dz = pos(3,l) - pos(3,m)
        if (do_pbc) then
          call pbc3(dx, dy, dz, x2, y2, z2, 1, ap, ibrav)
        else
          x2 = dx
          y2 = dy
          z2 = dz
        end if
        rlp(1) = x2
        rlp(2) = y2
        rlp(3) = z2
        rlm2 = rlp(1)*rlp(1) + rlp(2)*rlp(2) + rlp(3)*rlp(3)
        rlm1 = sqrt(rlm2)
        if (rlm1 < reps) cycle
        if (z(l) > 86 .or. z(m) > 86) cycle
        vdwst = sqrt(vdwc6(z(l))*vdwc6(z(m)))*convfac
        vdwrm = (vdwr0(z(l)) + vdwr0(z(m)))*fbohr
        rlm6 = rlm2*rlm2*rlm2
        rpow = a6grim*(rlm1/vdwrm - 1.0_real_8)
        if (abs(rpow) < 1.0_real_8/vdweps) then
          rexp = 1.0_real_8 + exp(-rpow)
          rphi = -s6blyp*vdwst/(rlm6*rexp)
          ffac = s6blyp*vdwst/(rlm6*rexp*rlm1) &
               *(6.0_real_8/rlm1 - a6grim*(rexp - 1.0_real_8)/(vdwrm*rexp))
          energy = energy + rphi
          do a = 1, 3
            force(a,l) = force(a,l) - ffac*rlp(a)
            force(a,m) = force(a,m) + ffac*rlp(a)
          end do
        end if
      end do
    end do
  end subroutine grimme
end module

program pbc3_drive
  use kinds
  use metr
  use pbc_extract
  implicit none
  character(len=16) :: mode
  integer :: ibrav, n, i, z(8)
  real(real_8) :: ap(4), pos(3,8), energy, force(3,8)
  logical :: do_pbc
  read(*,*) mode
  read(*,*) ibrav
  read(*,*) ap
  read(*,*) bc_com%v1crys
  read(*,*) bc_com%v2crys
  read(*,*) bc_com%v3crys
  do i = 1, 3
    read(*,*) metr_com%ht(i,1), metr_com%ht(i,2), metr_com%ht(i,3)
  end do
  call set_htm1(metr_com%ht, metr_com%htm1)
  read(*,*) n
  if (n < 1 .or. n > 8) stop 2
  do i = 1, n
    read(*,*) z(i), pos(1,i), pos(2,i), pos(3,i)
  end do
  do_pbc = mode == 'pbc'
  call grimme(n, z, pos, do_pbc, ibrav, ap, energy, force)
  write(*,'(A,1X,ES25.17)') 'energy', energy
  do i = 1, n
    write(*,'(A,3(1X,ES25.17))') 'force', force(1,i), force(2,i), force(3,i)
  end do
end program
"""


def compile_driver(src: Path, work: Path) -> Path:
    pbc_text = (src / "src/pbc_utils.mod.F90").read_text()
    vdw_text = (src / "src/vdw_utils.mod.F90").read_text()
    vdwin_text = (src / "src/vdwin_utils.mod.F90").read_text()
    cnst_text = (src / "src/cnst.mod.F90").read_text()
    require_lines(
        pbc_text,
        ("SUBROUTINE pbc3", "go to (5,10,99,25,99,99,99,15,99,99,99,20,99,25) ibrav"),
        "pbc_utils.mod.F90",
    )
    require_lines(
        vdw_text,
        (
            "CALL pbc(xlm_,ylm_,zlm_,xlm,ylm,zlm,1,parm%apbc,parm%ibrav)",
            "rphi=-s6grim*vdwst(n)/(rlm6*rexp)",
            "ffac=s6grim*vdwst(n)/(rlm6*rexp*rlm1)",
            "reps = 1.0e-2_real_8",
        ),
        "vdw_utils.mod.F90",
    )
    require_lines(
        vdwin_text,
        (
            "vdwi%nxvdw=0",
            "empvdwr%vdweps=1.0e-2_real_8",
            "a6grim=20.0_real_8",
            'IF (str == "BLYP" ) THEN',
            "set_s6 = 1.20_real_8",
        ),
        "vdwin_utils.mod.F90",
    )
    require_lines(cnst_text, ("fbohr=1._real_8/0.529177210859_real_8",), "cnst.mod.F90")
    tables = (
        extract_parameter(vdwin_text, "vdwc6")
        + extract_parameter(vdwin_text, "vdwr0")
        + extract_parameter(vdwin_text, "convfac")
    )
    code = fortran_source(
        extract_subroutine(pbc_text, "pbc3"),
        extract_subroutine(pbc_text, "rot"),
        tables,
    )
    path = work / "pbc3_drive.f90"
    path.write_text(code)
    exe = work / "pbc3_drive"
    subprocess.run(
        ["gfortran", "-cpp", "-O2", "-J", str(work), "-o", str(exe), str(path), "-lopenblas"],
        check=True,
    )
    return exe


def run_case(
    exe: Path,
    mode: str,
    ibrav: int,
    ap: list[float],
    vcrys: list[list[float]],
    h: list[list[float]],
    pos: list[list[float]],
) -> dict:
    lines = [mode, str(ibrav), " ".join(f"{v:.16e}" for v in ap)]
    for v in vcrys:
        lines.append(" ".join(f"{x:.16e}" for x in v))
    for row in h:
        lines.append(" ".join(f"{v:.16e}" for v in row))
    lines.append(str(len(pos)))
    for p in pos:
        lines.append("8 " + " ".join(f"{v:.16e}" for v in p))
    out = subprocess.run(
        [str(exe)],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    energy = None
    forces = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "energy":
            energy = float(parts[1])
        elif parts[0] == "force":
            forces.append([float(v) for v in parts[1:4]])
    if energy is None or len(forces) != len(pos):
        raise SystemExit(f"driver output was not a pair result:\n{out.stdout}\n{out.stderr}")
    return {"energy": energy, "forces": forces}


def angstrom(rows: list[list[float]]) -> list[list[float]]:
    return [[v * CPMD_BOHR for v in row] for row in rows]


def fmt(v: float) -> str:
    return format(v, ".17g")


def write_case(
    out_dir: Path, case: dict, result: dict, eucl: dict, near: dict, compiler: str, h_bohr, ap, vcrys, pos_bohr
) -> None:
    h_ang = angstrom(h_bohr)
    pos_ang = angstrom(pos_bohr)
    differs = abs(result["energy"] - eucl["energy"]) > 1.0e-8
    body = [
        f"# {case['name']}: OpenCPMD {COMMIT}",
        "# subroutine pbc3 in src/pbc_utils.mod.F90, called from subroutine vdw",
        "# in src/vdw_utils.mod.F90 (m = 1, Grimme branch, nxvdw = 0).",
        f"# compiler: {compiler}",
        f"# ibrav {case['ibrav']}",
        "# celldm " + " ".join(fmt(v) for v in case["celldm"]),
        "# apbc " + " ".join(fmt(v) for v in ap),
        "# v1crys v2crys v3crys " + " ".join(fmt(x) for v in vcrys for x in v),
        "# fractional " + " ".join(fmt(v) for v in case["fractional"]),
        "# positions in bohr, atom 1 at the origin:",
        "# " + " ".join(fmt(v) for v in pos_bohr[1]),
        "# euclidean and nearest are the Grimme energies of the Euclidean",
        "# nearest image and of the nearest integer fractional image, from",
        "# the same compiled kernel, not from pbc3.",
        f"version {COMMIT}",
        f"ibrav {case['ibrav']}",
        f"differs {1 if differs else 0}",
        "box " + " ".join(fmt(v) for row in h_ang for v in row),
        "atom 8 " + " ".join(fmt(v) for v in pos_ang[0]),
        "atom 8 " + " ".join(fmt(v) for v in pos_ang[1]),
        f"energy {fmt(result['energy'])}",
        f"euclidean {fmt(eucl['energy'])}",
        f"nearest {fmt(near['energy'])}",
    ]
    for force in result["forces"]:
        body.append("force " + " ".join(fmt(v) for v in force))
    path = out_dir / f"{case['name']}.txt"
    path.write_text("\n".join(body) + "\n")
    print(f"wrote {path} energy {result['energy']:.8e} differs {int(differs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="OpenCPMD checkout")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("CppCore/tests/data/d2cpmd"),
        help="directory for the reference files",
    )
    args = parser.parse_args()
    src = args.src.resolve()
    commit = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise SystemExit(f"{src} is {commit}, expected {COMMIT}")
    compiler = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0].strip()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pbc3-") as tmp:
        exe = compile_driver(src, Path(tmp))
        saw = False
        for case in CASES:
            h, ap, vcrys = latgen(case["ibrav"], case["celldm"])
            second = cart(case["fractional"], h)
            pos = [[0.0, 0.0, 0.0], second]
            result = run_case(exe, "pbc", case["ibrav"], ap, vcrys, h, pos)
            image = euclidean_image(second, h)
            eucl = run_case(exe, "raw", case["ibrav"], ap, vcrys, h, [[0.0, 0.0, 0.0], image])
            wrapped = nearest_integer_image(second, h)
            near = run_case(exe, "raw", case["ibrav"], ap, vcrys, h, [[0.0, 0.0, 0.0], wrapped])
            if case["ibrav"] == 12 and abs(result["energy"] - near["energy"]) <= 1.0e-8:
                raise SystemExit(f"{case['name']} does not tell symmetry 12 from the general branch")
            differs = abs(result["energy"] - eucl["energy"]) > 1.0e-8
            saw = saw or differs
            write_case(args.out, case, result, eucl, near, compiler, h, ap, vcrys, pos)
        if not saw:
            raise SystemExit("no reference cell distinguishes pbc3 from the Euclidean image")


if __name__ == "__main__":
    main()

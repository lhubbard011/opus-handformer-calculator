"""Plain-Python Game of Life, written separately from the weights: the expected byte sequence."""


def step(rows: list[str]) -> list[str]:
    """One generation; cells outside the grid are dead."""
    h, w = len(rows), len(rows[0])
    alive = lambda r, c: 0 <= r < h and 0 <= c < w and rows[r][c] == "#"  # noqa: E731
    out = []
    for r in range(h):
        line = ""
        for c in range(w):
            n = sum(alive(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
            line += "#" if n == 3 or (n == 2 and rows[r][c] == "#") else "."
        out.append(line)
    return out


def output(text: str) -> str:
    """What the model must write after the prompt '\\n<text>=' (text = 'N:row/row/...')."""
    count, grid = text.split(":", 1)
    rows, out = grid.split("/"), ""
    for k in range(int(count) - 1, -1, -1):
        rows = step(rows)
        out += f"{k}:" + "/".join(rows) + "="
    return out + "\n"

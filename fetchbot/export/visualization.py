"""Composite video frames: 3D chase view (+ robot camera inset) | live 2D map panel."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fetchbot.mapping.occupancy import FREE, OCCUPIED

if TYPE_CHECKING:
    from fetchbot.evaluation.episode import FetchEpisode

LABEL_RGB = {
    "red": (220, 30, 30),
    "green": (30, 170, 40),
    "blue": (30, 70, 230),
    "yellow": (230, 200, 0),
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw, text: str, size: int, max_w: float, min_size: int = 9
) -> ImageFont.ImageFont:
    """Largest default font <= ``size`` whose rendering of ``text`` is at most ``max_w`` px wide."""
    font = _font(size)
    while size > min_size and draw.textlength(text, font=font) > max_w:
        size -= 1
        font = _font(size)
    return font


def _wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    size: int,
    max_w: float,
    max_lines: int,
    min_size: int = 10,
) -> tuple[ImageFont.ImageFont, list[str]]:
    """Word-wrap ``text`` into at most ``max_lines`` lines of width <= ``max_w``.

    Shrinks the font down to ``min_size`` if needed; if it still does not fit, the
    last line is cut with an ellipsis (never drawn past the edge)."""
    words = text.split()
    while True:
        font = _font(size)
        lines, cur = [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if not cur or draw.textlength(cand, font=font) <= max_w:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        fits = all(draw.textlength(ln, font=font) <= max_w for ln in lines)
        if (len(lines) <= max_lines and fits) or size <= min_size:
            break
        size -= 1
    if len(lines) > max_lines or not fits:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "...", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    return font, lines


def _place_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    flip_dx: float | None = None,
    halo: tuple[int, int, int] | None = (250, 250, 250),
) -> None:
    """Draw ``text`` at (x, y) but keep it fully inside ``box`` (x0, y0, x1, y1).

    If the label would run past the right edge and ``flip_dx`` is given, it is moved
    to the other side of its anchor (x - flip_dx - width); otherwise it is shifted
    left. The vertical position is clamped the same way, using the font's real
    ascent + descent so letters like g/p/y are never cut. A light halo behind the
    text keeps it readable on dark map cells.
    """
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w = right - left
    if x + w > x1 - 2:
        x = (x - flip_dx - w) if flip_dx is not None else (x1 - 2 - w)
    x = min(max(x, x0 + 2), x1 - 2 - w)
    y = min(max(y, y0 + 1 - top), y1 - 1 - bottom)
    if halo is not None:
        draw.rectangle((x + left - 1, y + top - 1, x + right + 1, y + bottom + 1), fill=halo)
    draw.text((x, y), text, fill=fill, font=font)


class MapPanelRenderer:
    """Draws the robot's *belief*: occupancy grid, frontiers, path, detections, zones."""

    COLORS = {
        "unknown": (118, 120, 130),
        "free": (240, 240, 238),
        "free_unseen": (200, 204, 222),
        "occ": (25, 25, 30),
        "frontier": (0, 190, 215),
        "path": (220, 0, 200),
        "robot": (20, 110, 255),
    }

    TEXT_H = 128  # px below the map for the status text (command, program, state)

    def __init__(self, width: int = 480, height: int = 480) -> None:
        self.width, self.height = width, height
        self.font = _font(14)
        self.small = _font(12)

    def render(self, ep: FetchEpisode) -> Image.Image:
        g = ep.grid
        states = g.states()
        rgb = np.empty((*states.shape, 3), dtype=np.uint8)
        rgb[:] = self.COLORS["unknown"]
        rgb[states == FREE] = self.COLORS["free_unseen"]
        rgb[(states == FREE) & ep.coverage.seen] = self.COLORS["free"]
        rgb[states == OCCUPIED] = self.COLORS["occ"]
        if ep.explorer.frontier_mask is not None:
            rgb[ep.explorer.frontier_mask] = self.COLORS["frontier"]
        scale = min(self.width / g.nx, (self.height - self.TEXT_H) / g.ny)
        mw, mh = int(g.nx * scale), int(g.ny * scale)
        img = Image.new("RGB", (self.width, self.height), (250, 250, 250))
        grid_img = Image.fromarray(rgb[::-1]).resize((mw, mh), Image.NEAREST)
        ox, oy = (self.width - mw) // 2, 4
        img.paste(grid_img, (ox, oy))
        d = ImageDraw.Draw(img)

        def px(x: float, y: float) -> tuple[float, float]:
            col = (x - g.origin[0]) / g.resolution
            row = (y - g.origin[1]) / g.resolution
            return ox + col * scale, oy + mh - row * scale

        m_per_px = g.resolution / scale
        map_box = (ox, oy, ox + mw, oy + mh)
        zone_labels: list[tuple[float, float, str]] = []
        zone_r = {z.name: z.radius for z in ep.spec.zones}
        for name, pose in ep.ctx.destinations.items():
            r = zone_r.get(name, 0.8) / m_per_px
            cx, cy = px(pose.x, pose.y)
            is_dest = name == ep.command.destination
            d.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                outline=(150, 60, 170) if is_dest else (120, 120, 120),
                width=3 if is_dest else 1,
            )
            # Zone labels are drawn after the path/detections (see below) so they stay on top.
            zone_labels.append((cx - r, cy + r, name))

        nav = ep.navigator
        if nav.path and nav.active:
            pts = [px(x, y) for x, y in nav.path[nav._path_idx :]]
            if len(pts) > 1:
                d.line(pts, fill=self.COLORS["path"], width=2)
        if nav.goal is not None and nav.active:
            gx, gy = px(nav.goal.x, nav.goal.y)
            d.line((gx - 5, gy - 5, gx + 5, gy + 5), fill=(200, 0, 0), width=2)
            d.line((gx - 5, gy + 5, gx + 5, gy - 5), fill=(200, 0, 0), width=2)

        for oid, det in ep.semantic_map.items():
            if det.map_pose is None:
                continue
            cx, cy = px(det.map_pose.x, det.map_pose.y)
            color = LABEL_RGB.get(det.label.split()[0], (0, 0, 0))
            r = 6 if oid == ep.ctx.target_id else 4
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(0, 0, 0))
            shape = ep.semantic_map.shape_of(oid)
            text = f"{det.label} {shape}" if shape else det.label
            if oid == ep.ctx.target_id:
                text += " (target)"
            _place_label(d, cx + 7, cy - 7, text, self.small, map_box, color, flip_dx=7)
        for lx, ly, name in zone_labels:
            _place_label(d, lx, ly, name, self.small, map_box, (90, 30, 110), halo=None)

        pose = ep.robot.get_pose()
        cx, cy = px(pose.x, pose.y)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        hl, hw = ep.robot.half_length, ep.robot.half_width
        corners = [
            px(pose.x + a * c - b * s, pose.y + a * s + b * c)
            for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
        ]
        d.polygon(corners, fill=self.COLORS["robot"], outline=(0, 0, 0))
        hx, hy = px(pose.x + hl * c, pose.y + hl * s)
        d.line((cx, cy, hx, hy), fill=(0, 0, 0), width=2)
        if ep.robot.is_holding:
            d.rectangle((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255))

        ty = oy + mh + 5
        x0, max_w = 8, self.width - 16
        state = ep.manager.current.upper()
        if ep.ctx.failure:
            state += " (failed)"
        n = len(ep.program.steps)
        k = min(ep.step_index, n - 1)
        if ep.finished:
            ok = sum(r.success for r in ep.records)
            prog = f"Program done: {ok}/{n} steps succeeded"
        else:
            prog = f"Step {k + 1}/{n}: {ep.program.steps[k].describe()}"
            if ep.ctx.surveying:
                prog += "  [survey]"
        # The command can be long: wrap it onto at most 2 lines (shrinking the font
        # if needed) so that nothing is clipped at the panel edge.
        cmd_font, cmd_lines = _wrap(d, f'Command: "{ep.program.raw_text}"', 14, max_w, 2)
        lines = [(ln, cmd_font) for ln in cmd_lines]
        for text in (
            prog,
            f"State: {state}    sim t = {ep.world.sim_time:5.1f} s",
            f"Distance {ep.odom.distance:5.1f} m   replans {nav.blocked_replans}"
            f"   map known {100 * g.known_fraction():3.0f}%",
        ):
            lines.append((text, _fit_font(d, text, 14, max_w)))
        step_h = (self.height - ty - 2) / max(len(lines), 1)
        for i, (line, font) in enumerate(lines):
            d.text((x0, ty + i * min(19, step_h)), line, fill=(20, 20, 20), font=font)
        return img


class CompositeFrameRenderer:
    """Left: chase-camera render with robot-camera inset. Right: :class:`MapPanelRenderer`."""

    def __init__(self, view_size: tuple[int, int] = (800, 480), panel_width: int = 480) -> None:
        self.view_w, self.view_h = view_size
        self.panel = MapPanelRenderer(panel_width, self.view_h)
        self.font = _font(13)
        self.render_seconds = 0.0
        self.frames_rendered = 0

    def chase_eye(self, ep: FetchEpisode):
        pose = ep.robot.get_pose()
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        eye = (pose.x - 3.4 * c, pose.y - 3.4 * s, 3.6)
        target = (pose.x + 1.2 * c, pose.y + 1.2 * s, 0.0)
        return eye, target

    def render(self, ep: FetchEpisode) -> np.ndarray:
        import time

        t0 = time.perf_counter()
        eye, target = self.chase_eye(ep)
        view, _ = ep.world.render(eye, target, self.view_w, self.view_h, 60.0)
        self.render_seconds += time.perf_counter() - t0
        self.frames_rendered += 1
        left = Image.fromarray(view)
        cam = ep.last_camera_rgb
        if cam is not None:
            inset = Image.fromarray(cam).resize(
                (cam.shape[1] * 3 // 2, cam.shape[0] * 3 // 2), Image.NEAREST
            )
            di = ImageDraw.Draw(inset)
            k = 1.5
            for det in ep.last_detections:
                x0, y0, x1, y1 = det.bbox_xyxy
                color = LABEL_RGB.get(det.label.split()[0], (255, 0, 255))
                di.rectangle(
                    (x0 * k - 2, y0 * k - 2, x1 * k + 2, y1 * k + 2), outline=color, width=2
                )
                text = f"{det.label} {det.shape}" if det.shape else det.label
                _place_label(
                    di,
                    x0 * k,
                    y0 * k - 16,
                    text,
                    self.font,
                    (0, 0, inset.width, inset.height),
                    color,
                    halo=None,
                )
            left.paste(inset, (self.view_w - inset.width - 8, 8))
            dl = ImageDraw.Draw(left)
            dl.rectangle(
                (self.view_w - inset.width - 9, 7, self.view_w - 7, inset.height + 8),
                outline=(20, 20, 20),
                width=1,
            )
            dl.text(
                (self.view_w - inset.width - 4, inset.height + 10),
                "robot RGB-D camera",
                fill=(20, 20, 20),
                font=self.font,
            )
        dl = ImageDraw.Draw(left)
        dl.text(
            (10, 8),
            f"PyBullet (CPU tiny renderer)  seed {ep.seed}",
            fill=(20, 20, 20),
            font=self.font,
        )
        right = self.panel.render(ep)
        frame = Image.new("RGB", (self.view_w + right.width, self.view_h))
        frame.paste(left, (0, 0))
        frame.paste(right, (self.view_w, 0))
        return np.asarray(frame)

# interactive_voronoi.py
import sys
import random
import math
import pygame
import numpy as np

from shapely.geometry import box, Point, MultiPoint, Polygon
from shapely.ops import voronoi_diagram

# Fallback flag will be set if voronoi_diagram isn't available.
USE_SHAPELY_VOR = True

# Try to ensure voronoi_diagram exists.
try:
    # this will raise if not present in this shapely version
    _ = voronoi_diagram
except Exception:
    USE_SHAPELY_VOR = False

# If shapely voronoi not available, we'll later use scipy + boundary points (hack fallback)
try:
    from scipy.spatial import Voronoi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

if not USE_SHAPELY_VOR and not HAVE_SCIPY:
    raise RuntimeError("Your environment needs either shapely >=2.0 (preferred) or scipy. Install them with: pip install shapely scipy")

# ---------- Config ----------
WIDTH, HEIGHT = 1000, 700
POINT_RADIUS = 6
HIT_RADIUS = 10
BG_COLOR = (30, 30, 30)
LINE_COLOR = (40, 40, 40)
FPS = 120

# ---------- Helpers ----------
def rand_color():
    # nice saturated pastel-ish colors
    h = random.random()
    s = 0.6 + 0.3 * random.random()
    l = 0.5 + 0.1 * random.random()
    # convert HSL to RGB
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))

def screen_poly_from_shapely(poly: Polygon):
    # convert shapely polygon coordinates to pygame-int tuple list
    coords = list(poly.exterior.coords)
    pts = [(int(round(x)), int(round(y))) for x, y in coords]
    return pts

def nearest_point_index(points, x, y):
    best_i = None
    best_d2 = float("inf")
    for i, (px, py, col) in enumerate(points):
        dx = px - x
        dy = py - y
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i, best_d2

# ---------- Voronoi computation functions ----------
def bounded_voronoi_shapely(points_xy, bounds):
    """
    Compute Voronoi polygons clipped to bounds using shapely.ops.voronoi_diagram.
    Returns list of (polygon, site_index) where polygon is a shapely Polygon clipped to bounds.
    """
    if len(points_xy) == 0:
        return []

    mp = MultiPoint([Point(x, y) for x, y in points_xy])
    # Ask shapely to compute the voronoi diagram within a sufficiently large envelope.
    # We pass envelope == bounding box so voronoi_diagram produces regions covering that area.
    envelope = box(bounds[0], bounds[1], bounds[2], bounds[3])
    try:
        v = voronoi_diagram(mp, envelope=envelope)
    except TypeError:
        # Some shapely variants may expect geometry only and will auto-bounds; still proceed
        v = voronoi_diagram(mp)

    # v is a geometrycollection of polygons (Voronoi cells)
    polys = []
    # Map each polygon to the nearest site (centroid mapping). This is robust.
    for geom in v.geoms:
        # intersect/cap with the bounds rectangle to be safe
        clipped = geom.intersection(envelope)
        if clipped.is_empty:
            continue
        # sometimes result could be MultiPolygon; we take the polygonal union as single poly if so
        if clipped.geom_type == "MultiPolygon":
            clipped = Polygon()
            for g in geom.geoms:
                clipped = clipped.union(g).intersection(envelope)
        polys.append(clipped)

    # Now assign each polygon to nearest site (by polygon centroid or representative point)
    assigned = []
    for poly in polys:
        rep = poly.representative_point()  # robust interior point
        rx, ry = rep.x, rep.y
        # find nearest original site
        best_i, d2 = nearest_point_index([(x, y, None) for (x, y) in points_xy], rx, ry)
        if best_i is not None:
            assigned.append((poly, best_i))
    # There can be duplicates if voronoi_diagram produced multiple polygons per site; merge them
    merged = {}
    for poly, idx in assigned:
        if idx not in merged:
            merged[idx] = poly
        else:
            merged[idx] = merged[idx].union(poly)
    result = [(merged[i], i) for i in sorted(merged.keys())]
    return result

def bounded_voronoi_scipy(points_xy, bounds):
    """
    Fallback: approximate bounded Voronoi by adding points around the bounding box corners/edges
    to make regions finite, compute Voronoi with scipy, then clip polygons to the bounding box.
    Not as accurate near edges, but works.
    """
    if len(points_xy) == 0:
        return []
    minx, miny, maxx, maxy = bounds
    margin = max(maxx - minx, maxy - miny) * 2.0  # far away helper points
    # create helper points around box (8 or more)
    helpers = [
        (minx - margin, miny - margin), (minx - margin/2, miny - margin),
        (maxx + margin, miny - margin), (maxx + margin, miny - margin/2),
        (maxx + margin, maxy + margin), (maxx + margin/2, maxy + margin),
        (minx - margin, maxy + margin), (minx - margin, maxy + margin/2),
    ]
    all_pts = np.array(points_xy + helpers)
    vor = Voronoi(all_pts)
    minx, miny, maxx, maxy = bounds
    bbox = box(minx, miny, maxx, maxy)
    polys_assigned = []
    for i in range(len(points_xy)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if not region:
            continue
        if -1 in region:
            # region infinite: build from ridge lines
            # collect finite vertices + construct approximate polygon via convex hull of reachable finite vertices
            finite_verts = [vor.vertices[v] for v in region if v != -1]
            if len(finite_verts) < 3:
                continue
            poly = Polygon(finite_verts).convex_hull.intersection(bbox)
            if not poly.is_empty and poly.area > 0:
                polys_assigned.append((poly, i))
        else:
            coords = [tuple(vor.vertices[v]) for v in region]
            poly = Polygon(coords).intersection(bbox)
            if not poly.is_empty and poly.area > 0:
                polys_assigned.append((poly, i))
    # merge if multiple parts per site
    merged = {}
    for poly, idx in polys_assigned:
        if idx not in merged:
            merged[idx] = poly
        else:
            merged[idx] = merged[idx].union(poly)
    return [(merged[i], i) for i in sorted(merged.keys())]

# ---------- Pygame app ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Interactive Bounded Voronoi (pygame + shapely)")
    clock = pygame.time.Clock()
    running = True

    # store points as tuples (x, y, color)
    points = []
    dragging_idx = None
    drag_offset = (0, 0)

    # precompute bounds for shapely functions: (minx, miny, maxx, maxy)
    bounds = (0.0, 0.0, float(WIDTH), float(HEIGHT))
    use_shapely = USE_SHAPELY_VOR

    if not use_shapely:
        print("Warning: shapely.ops.voronoi_diagram not available. Using scipy fallback (less precise near edges).")
    else:
        print("Using shapely.ops.voronoi_diagram for bounded Voronoi (recommended).")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:  # left click: either start dragging or create new point
                    idx, d2 = nearest_point_index(points, mx, my)
                    if idx is not None and d2 <= HIT_RADIUS*HIT_RADIUS:
                        dragging_idx = idx
                        px, py, _ = points[idx]
                        drag_offset = (px - mx, py - my)
                    else:
                        # add new point
                        col = rand_color()
                        points.append([float(mx), float(my), col])
                elif event.button == 3:  # right click: remove a point if close
                    idx, d2 = nearest_point_index(points, mx, my)
                    if idx is not None and d2 <= HIT_RADIUS*HIT_RADIUS:
                        points.pop(idx)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_idx = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging_idx is not None:
                    mx, my = event.pos
                    offx, offy = drag_offset
                    points[dragging_idx][0] = float(mx + offx)
                    points[dragging_idx][1] = float(my + offy)

        # --- compute voronoi polygons each frame ---
        pts_xy = [(p[0], p[1]) for p in points]
        polys = []
        if use_shapely:
            try:
                polys = bounded_voronoi_shapely(pts_xy, bounds)
            except Exception as e:
                # fallback if something goes wrong
                print("Shapely voronoi failed:", e)
                if HAVE_SCIPY:
                    polys = bounded_voronoi_scipy(pts_xy, bounds)
        else:
            if HAVE_SCIPY:
                polys = bounded_voronoi_scipy(pts_xy, bounds)

        # Clear screen
        screen.fill(BG_COLOR)

        # Draw Voronoi polygons
        # build list such that we skip sites that have no polygon (possible if too few points)
        for poly, site_idx in polys:
            # get site color
            if site_idx < 0 or site_idx >= len(points):
                continue
            color = points[site_idx][2]
            try:
                pts = screen_poly_from_shapely(poly)
                if len(pts) >= 3:
                    pygame.gfxdraw_filled_polygon = getattr(pygame, "gfxdraw", None)  # silence linter
                    # use pygame.draw.polygon
                    pygame.draw.polygon(screen, color, pts)
                    # subtle border
                    pygame.draw.polygon(screen, LINE_COLOR, pts, 1)
            except Exception:
                pass

        # Draw points on top
        for i, (x, y, col) in enumerate(points):
            ix, iy = int(round(x)), int(round(y))
            pygame.draw.circle(screen, (255, 255, 255), (ix, iy), POINT_RADIUS+2)  # white border for visibility
            pygame.draw.circle(screen, col, (ix, iy), POINT_RADIUS)
            # optionally draw index
            # font = pygame.font.SysFont(None, 18)
            # screen.blit(font.render(str(i), True, (200,200,200)), (ix+8, iy-8))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

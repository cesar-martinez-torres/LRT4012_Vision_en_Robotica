"""Generate a seeded maze and coordinate visual commands for the Burger."""

from __future__ import annotations

import math
import json
import os
import random
from collections import deque
from pathlib import Path

from controller import Supervisor


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ICON_FIELDS = {
    "forward": "forwardIconTranslation",
    "left": "leftIconTranslation",
    "right": "rightIconTranslation",
    "stop": "stopIconTranslation",
}
CELL_SIZE = 0.42
WALL_THICKNESS = 0.025
WALL_HEIGHT = 0.13
SIGN_DISTANCE = 0.72
SIGN_HEIGHT = 0.27


def require_node(supervisor: Supervisor, def_name: str):
    node = supervisor.getFromDef(def_name)
    if node is None:
        raise RuntimeError(f"Required DEF node was not found: {def_name}")
    return node


def cell_center(cell: tuple[int, int], rows: int, cols: int) -> tuple[float, float]:
    column, row = cell
    x = (column - (cols - 1) / 2.0) * CELL_SIZE
    y = (row - (rows - 1) / 2.0) * CELL_SIZE
    return x, y


def grid_neighbors(cell: tuple[int, int], rows: int, cols: int):
    column, row = cell
    for dx, dy in DIRECTIONS:
        neighbor = (column + dx, row + dy)
        if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
            yield neighbor


def generate_maze(rows: int, cols: int, rng: random.Random):
    adjacency = {
        (column, row): set() for row in range(rows) for column in range(cols)
    }
    initial = (rng.randrange(cols), rng.randrange(rows))
    visited = {initial}
    stack = [initial]
    while stack:
        current = stack[-1]
        candidates = [
            neighbor
            for neighbor in grid_neighbors(current, rows, cols)
            if neighbor not in visited
        ]
        if not candidates:
            stack.pop()
            continue
        rng.shuffle(candidates)
        neighbor = candidates[0]
        adjacency[current].add(neighbor)
        adjacency[neighbor].add(current)
        visited.add(neighbor)
        stack.append(neighbor)
    return adjacency


def farthest_path(adjacency, start: tuple[int, int]):
    parent = {start: None}
    distance = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            distance[neighbor] = distance[current] + 1
            queue.append(neighbor)
    goal = max(distance, key=lambda cell: (distance[cell], cell[1], cell[0]))
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return goal, path


def path_to_commands(path: list[tuple[int, int]], initial_heading: int = 0):
    heading = initial_heading
    commands = []
    for current, following in zip(path, path[1:]):
        delta = (following[0] - current[0], following[1] - current[1])
        target_heading = DIRECTIONS.index(delta)
        turn = (target_heading - heading) % 4
        if turn == 1:
            commands.append("left")
        elif turn == 2:
            commands.extend(("right", "right"))
        elif turn == 3:
            commands.append("right")
        commands.append("forward")
        heading = target_heading
    commands.append("stop")
    return commands


def import_wall(children, x: float, y: float, size_x: float, size_y: float):
    node = f"""
Solid {{
  translation {x:.6f} {y:.6f} {WALL_HEIGHT / 2.0:.6f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.20 0.27 0.34
        roughness 0.92
        metalness 0
      }}
      geometry Box {{ size {size_x:.6f} {size_y:.6f} {WALL_HEIGHT:.6f} }}
    }}
  ]
  name "maze wall"
  boundingObject Box {{ size {size_x:.6f} {size_y:.6f} {WALL_HEIGHT:.6f} }}
}}
"""
    children.importMFNodeFromString(-1, node)


def import_marker(children, x: float, y: float, color: tuple[float, float, float]):
    red, green, blue = color
    size = CELL_SIZE * 0.64
    node = f"""
Transform {{
  translation {x:.6f} {y:.6f} 0.003
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {red:.3f} {green:.3f} {blue:.3f}
        roughness 1
        metalness 0
        emissiveColor {red * 0.18:.3f} {green * 0.18:.3f} {blue * 0.18:.3f}
      }}
      geometry Box {{ size {size:.6f} {size:.6f} 0.006 }}
    }}
  ]
}}
"""
    children.importMFNodeFromString(-1, node)


def build_track(children, adjacency, rows: int, cols: int, start, goal):
    while children.getCount() > 0:
        children.removeMF(children.getCount() - 1)

    for row in range(rows):
        for column in range(cols):
            cell = (column, row)
            x, y = cell_center(cell, rows, cols)
            if column == 0:
                import_wall(children, x - CELL_SIZE / 2, y, WALL_THICKNESS, CELL_SIZE + WALL_THICKNESS)
            if row == 0:
                import_wall(children, x, y - CELL_SIZE / 2, CELL_SIZE + WALL_THICKNESS, WALL_THICKNESS)
            if column == cols - 1:
                import_wall(children, x + CELL_SIZE / 2, y, WALL_THICKNESS, CELL_SIZE + WALL_THICKNESS)
            elif (column + 1, row) not in adjacency[cell]:
                import_wall(children, x + CELL_SIZE / 2, y, WALL_THICKNESS, CELL_SIZE + WALL_THICKNESS)
            if row == rows - 1:
                import_wall(children, x, y + CELL_SIZE / 2, CELL_SIZE + WALL_THICKNESS, WALL_THICKNESS)
            elif (column, row + 1) not in adjacency[cell]:
                import_wall(children, x, y + CELL_SIZE / 2, CELL_SIZE + WALL_THICKNESS, WALL_THICKNESS)

    start_x, start_y = cell_center(start, rows, cols)
    goal_x, goal_y = cell_center(goal, rows, cols)
    import_marker(children, start_x, start_y, (0.12, 0.38, 0.72))
    import_marker(children, goal_x, goal_y, (0.10, 0.62, 0.24))


def reset_robot(robot_node, start: tuple[int, int], rows: int, cols: int):
    x, y = cell_center(start, rows, cols)
    robot_node.getField("translation").setSFVec3f([x, y, 0.0])
    robot_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])
    robot_node.resetPhysics()


def place_sign(robot_node, sign_node, command: str):
    position = robot_node.getPosition()
    orientation = robot_node.getOrientation()
    forward_x, forward_y = orientation[0], orientation[3]
    sign_node.getField("translation").setSFVec3f(
        [
            position[0] + SIGN_DISTANCE * forward_x,
            position[1] + SIGN_DISTANCE * forward_y,
            SIGN_HEIGHT,
        ]
    )
    yaw = math.atan2(forward_y, forward_x)
    sign_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, yaw])
    for field_name in ICON_FIELDS.values():
        sign_node.getField(field_name).setSFVec3f([-0.030, 0.0, 5.0])
    sign_node.getField(ICON_FIELDS[command]).setSFVec3f([-0.030, 0.0, 0.0])


def main() -> None:
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("status receiver")
    receiver.enable(time_step)
    emitter = supervisor.getDevice("command emitter")
    robot_node = require_node(supervisor, "BURGER")
    sign_node = require_node(supervisor, "SIGN")
    track_root = require_node(supervisor, "TRACK_ROOT")
    children = track_root.getField("children")

    seed = int(os.environ.get("WEBOTS_TRACK_SEED", "4012"))
    rows = int(os.environ.get("WEBOTS_TRACK_ROWS", "4"))
    cols = int(os.environ.get("WEBOTS_TRACK_COLS", "4"))
    if rows < 3 or cols < 3:
        raise ValueError("The random track requires at least 3 rows and 3 columns.")
    batch = os.environ.get("WEBOTS_TURTLEBOT_BATCH", "0") == "1"
    goal_tolerance = float(os.environ.get("WEBOTS_GOAL_TOLERANCE", "0.04"))
    rng = random.Random(seed)
    adjacency = generate_maze(rows, cols, rng)
    corners = ((0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1))
    start = corners[rng.randrange(len(corners))]
    goal, path = farthest_path(adjacency, start)
    route = path_to_commands(path)

    build_track(children, adjacency, rows, cols, start, goal)
    reset_robot(robot_node, start, rows, cols)
    supervisor.step(time_step)

    route_index = 0
    completed = []
    measurements = []
    cell_index = 0
    heading = 0

    def finish(status, reason, code):
        emitter.send(b"finish" if status == "completed" else b"abort")
        if batch:
            output = Path(__file__).resolve().parents[2] / "data" / "turtlebot_route"
            output.mkdir(parents=True, exist_ok=True)
            (output / f"motion_summary_{seed}.json").write_text(
                json.dumps({"seed": seed, "rows": rows, "cols": cols,
                            "status": status, "reason": reason,
                            "expected_commands": route, "completed_commands": completed,
                            "measurements": measurements}, indent=2), encoding="utf-8")
        # Deliver the final acknowledgement and console messages before shutdown.
        for _ in range(3):
            if supervisor.step(time_step) == -1:
                break
        if batch:
            supervisor.simulationQuit(code)
    place_sign(robot_node, sign_node, route[route_index])
    print(
        f"scenario_seed={seed} grid={rows}x{cols} start={start} goal={goal} "
        f"path_cells={len(path)} route_commands={len(route)}",
        flush=True,
    )
    print(f"path={path}", flush=True)
    print(f"route={route}", flush=True)
    print(f"route_step=1/{len(route)} sign={route[0]}", flush=True)

    finished = False
    while supervisor.step(time_step) != -1:
        if finished:
            continue
        while receiver.getQueueLength() > 0:
            message = receiver.getString()
            receiver.nextPacket()
            if message.startswith("failed:"):
                print(f"scenario_status=failed reason={message}", flush=True)
                finish("failed", message, 3)
                return
            if not message.startswith("done:"):
                continue
            command = message.split(":", 1)[1]
            expected = route[route_index]
            completed.append(command)
            print(f"reported={command} expected={expected}", flush=True)
            if command != expected:
                print(
                    f"scenario_status=failed reason=wrong_command step={route_index + 1}",
                    flush=True,
                )
                finish("failed", "wrong_command", 1)
                return

            if command == "forward":
                cell_index += 1
            elif command == "left":
                heading = (heading + 1) % 4
            elif command == "right":
                heading = (heading - 1) % 4
            target_x, target_y = cell_center(path[cell_index], rows, cols)
            position = robot_node.getPosition()
            orientation = robot_node.getOrientation()
            yaw = math.atan2(orientation[3], orientation[0])
            yaw_error = math.atan2(math.sin(yaw - heading * math.pi / 2),
                                   math.cos(yaw - heading * math.pi / 2))
            position_error = math.hypot(position[0] - target_x, position[1] - target_y)
            measurements.append({"command": command, "position_error_m": position_error,
                                 "heading_error_deg": math.degrees(yaw_error)})
            print(f"motion_check position_error={position_error:.4f} "
                  f"heading_error_deg={math.degrees(yaw_error):.3f}", flush=True)
            if position_error > goal_tolerance or abs(yaw_error) > math.radians(2.0):
                print("scenario_status=failed reason=motion_error", flush=True)
                finish("failed", "motion_error", 2)
                return

            route_index += 1
            if route_index >= len(route):
                position = robot_node.getPosition()
                goal_x, goal_y = cell_center(goal, rows, cols)
                goal_error = math.hypot(position[0] - goal_x, position[1] - goal_y)
                success = goal_error <= goal_tolerance
                status = "completed" if success else "failed"
                print(
                    f"scenario_status={status} goal_error={goal_error:.3f} "
                    f"tolerance={goal_tolerance:.3f} commands={completed}",
                    flush=True,
                )
                finished = True
                finish(status, "goal_reached" if success else "goal_error", 0 if success else 2)
                return

            place_sign(robot_node, sign_node, route[route_index])
            emitter.send(b"next")
            print(
                f"route_step={route_index + 1}/{len(route)} sign={route[route_index]}",
                flush=True,
            )


if __name__ == "__main__":
    main()

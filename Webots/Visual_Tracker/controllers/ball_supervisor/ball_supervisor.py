"""Movimiento circular de la pelota roja."""

from math import cos, sin

from controller import Supervisor


RADIUS = 1.0          # radio de la trayectoria [m]
ANGULAR_SPEED = 0.06  # velocidad angular [rad/s]
BALL_HEIGHT = 0.10    # altura del centro de la pelota [m]


supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
ball_translation = supervisor.getFromDef("BALL").getField("translation")

while supervisor.step(timestep) != -1:
    angle = ANGULAR_SPEED * supervisor.getTime()
    x = RADIUS * sin(angle)
    y = RADIUS * cos(angle)
    ball_translation.setSFVec3f([x, y, BALL_HEIGHT])

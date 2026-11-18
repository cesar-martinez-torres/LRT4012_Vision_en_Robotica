"""Three synchronized views on one native Webots Display device."""

import cv2
import numpy as np

WIDTH, HEIGHT = 960, 244
IMAGE_WIDTH, IMAGE_HEIGHT, HEADER = 320, 192, 24


def compose_views(bgr, mask, target, speed, angular, inference_ms, state):
    """Display labels without changing the input image or integer class mask."""
    overlay = bgr.copy()
    overlay[mask == 1] = (0.55 * overlay[mask == 1] + 0.45 * np.array([40, 220, 40])).astype(np.uint8)
    if target is not None:
        point = (round(target[2]), round(target[3]))
        cv2.line(overlay, (IMAGE_WIDTH // 2, IMAGE_HEIGHT - 1), point, (0, 0, 230), 1)
        cv2.circle(overlay, point, 5, (0, 0, 255), -1)
    else:
        cv2.putText(overlay, 'SIN REFERENCIA VISUAL', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    visible_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    canvas = np.full((HEIGHT, WIDTH, 3), 28, np.uint8)
    for index, (title, image) in enumerate(zip(
            ('CAMARA RGB', 'MASCARA PREDICHA', 'CARRIL Y PUNTO OBJETIVO'),
            (bgr, visible_mask, overlay))):
        left = index * IMAGE_WIDTH
        # Labels below the images remain visible beside Webots' native RGB overlay.
        canvas[:IMAGE_HEIGHT, left:left + IMAGE_WIDTH] = image
        cv2.putText(canvas, title, (left + 10, IMAGE_HEIGHT + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (245, 245, 245), 1, cv2.LINE_AA)
    footer = f'{state} | v={speed:.3f} m/s | giro={angular:.3f} rad/s | ONNX={inference_ms:.1f} ms'
    cv2.putText(canvas, footer, (10, HEIGHT - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Blanco: carril | Negro: fondo', (650, HEIGHT - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 225), 1, cv2.LINE_AA)
    return canvas, overlay


class VisionDashboard:
    def __init__(self, display):
        self.display = display
        if (display.getWidth(), display.getHeight()) != (WIDTH, HEIGHT):
            raise ValueError('vision dashboard Display must be 960 x 244 pixels')
        self.last = None
        self.update(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8),
                    np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8),
                    None, 0, 0, 0, 'ESPERANDO INICIO')

    def update(self, bgr, mask, target, speed, angular, inference_ms, state='SEGUIMIENTO'):
        self.last = (bgr, mask, target, inference_ms)
        canvas, overlay = compose_views(bgr, mask, target, speed, angular, inference_ms, state)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        reference = self.display.imageNew(rgb.tobytes(), self.display.RGB, WIDTH, HEIGHT)
        try:
            self.display.imagePaste(reference, 0, 0, False)
        finally:
            # Delete each temporary image after copying it to the persistent display.
            self.display.imageDelete(reference)
        return canvas, overlay

    def stop(self, state):
        bgr, mask, target, elapsed = self.last
        self.update(bgr, mask, target, 0, 0, elapsed, state)

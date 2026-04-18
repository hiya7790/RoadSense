"""
simulation/dashboard.py — Pygame Interactive Dashboard for VisionSuspend
Displays live camera feed, classification label, suspension dial, and confidence bar.
"""

import sys
import os
import math
import cv2
import numpy as np
import pygame
import argparse
from tensorflow.keras.models import load_model

CLASSES = ["smooth", "gravel", "pothole", "wet"]
SUSPENSION_MAP = {
    "smooth": ("Soft", (100, 220, 100)),
    "gravel": ("Medium", (255, 180, 50)),
    "pothole": ("Firm", (220, 80, 80)),
    "wet": ("Adaptive", (100, 150, 255)),
}
IMG_SIZE = (224, 224)

# Dashboard layout
WIN_W, WIN_H = 900, 540
FEED_W, FEED_H = 540, 405
PANEL_X = 560
DIAL_CX, DIAL_CY = PANEL_X + 155, 200
DIAL_R = 90


def preprocess(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb / 255.0, axis=0)


def draw_dial(surface, label, confidence):
    suspension_label, color = SUSPENSION_MAP.get(label, ("Unknown", (200, 200, 200)))

    # Dial background
    pygame.draw.circle(surface, (40, 40, 40), (DIAL_CX, DIAL_CY), DIAL_R + 5)
    pygame.draw.circle(surface, (20, 20, 20), (DIAL_CX, DIAL_CY), DIAL_R)

    # Tick marks
    for i in range(4):
        angle = math.radians(-150 + i * 100)
        x1 = DIAL_CX + int((DIAL_R - 8) * math.cos(angle))
        y1 = DIAL_CY - int((DIAL_R - 8) * math.sin(angle))
        x2 = DIAL_CX + int(DIAL_R * math.cos(angle))
        y2 = DIAL_CY - int(DIAL_R * math.sin(angle))
        pygame.draw.line(surface, (180, 180, 180), (x1, y1), (x2, y2), 2)

    # Needle: maps class index to angle
    class_idx = CLASSES.index(label) if label in CLASSES else 0
    needle_angle = math.radians(-150 + class_idx * 100)
    nx = DIAL_CX + int((DIAL_R - 20) * math.cos(needle_angle))
    ny = DIAL_CY - int((DIAL_R - 20) * math.sin(needle_angle))
    pygame.draw.line(surface, color, (DIAL_CX, DIAL_CY), (nx, ny), 4)
    pygame.draw.circle(surface, (255, 255, 255), (DIAL_CX, DIAL_CY), 8)

    return suspension_label, color


def run_dashboard(model_path, camera_id=0):
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("VisionSuspend — Dashboard")
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont("Arial", 22, bold=True)
    font_md = pygame.font.SysFont("Arial", 17)
    font_sm = pygame.font.SysFont("Arial", 14)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        pygame.quit()
        sys.exit()

    label = "smooth"
    confidence = 0.0
    print("Dashboard running. Close window to quit.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        ret, frame = cap.read()
        if ret:
            preds = model.predict(preprocess(frame), verbose=0)[0]
            idx = int(np.argmax(preds))
            label = CLASSES[idx]
            confidence = float(preds[idx])

        screen.fill((15, 15, 20))

        # --- Live feed ---
        if ret:
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (FEED_W, FEED_H)), cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
            screen.blit(surf, (10, 10))
        pygame.draw.rect(screen, (80, 80, 80), (10, 10, FEED_W, FEED_H), 2)

        # --- Panel background ---
        pygame.draw.rect(screen, (25, 25, 35), (PANEL_X, 0, WIN_W - PANEL_X, WIN_H))

        # Title
        title = font_lg.render("VisionSuspend", True, (220, 220, 255))
        screen.blit(title, (PANEL_X + 10, 12))

        # Road label
        _, color = SUSPENSION_MAP.get(label, ("Unknown", (200, 200, 200)))
        road_txt = font_md.render(f"Road:  {label.upper()}", True, color)
        screen.blit(road_txt, (PANEL_X + 10, 50))

        # Suspension dial
        suspension_label, susp_color = draw_dial(screen, label, confidence)
        dial_title = font_sm.render("Suspension Setting", True, (180, 180, 180))
        screen.blit(dial_title, (PANEL_X + 90, 100))
        susp_txt = font_lg.render(suspension_label, True, susp_color)
        screen.blit(susp_txt, (PANEL_X + 110, 295))

        # Confidence bar
        bar_y = 340
        bar_w_total = WIN_W - PANEL_X - 20
        conf_label = font_sm.render(f"Confidence: {confidence*100:.1f}%", True, (200, 200, 200))
        screen.blit(conf_label, (PANEL_X + 10, bar_y - 20))
        pygame.draw.rect(screen, (50, 50, 50), (PANEL_X + 10, bar_y, bar_w_total, 18), border_radius=6)
        filled_w = int(bar_w_total * confidence)
        if filled_w > 0:
            pygame.draw.rect(screen, susp_color, (PANEL_X + 10, bar_y, filled_w, 18), border_radius=6)

        # Class legend
        legend_y = 390
        for i, cls in enumerate(CLASSES):
            _, c = SUSPENSION_MAP[cls]
            pygame.draw.rect(screen, c, (PANEL_X + 10, legend_y + i * 22, 12, 12))
            legend_txt = font_sm.render(f"{cls.capitalize()} → {SUSPENSION_MAP[cls][0]}", True, (180, 180, 180))
            screen.blit(legend_txt, (PANEL_X + 28, legend_y + i * 22 - 2))

        pygame.display.flip()
        clock.tick(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSuspend Pygame Dashboard")
    parser.add_argument("--model", type=str, default="models/saved/mobilenetv2_best.h5")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_dashboard(args.model, args.camera)

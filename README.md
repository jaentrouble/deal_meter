# LostArk Deal Analyzer

- Final goal: Analyzing per-skill damage

---

## Phase 1 - Boss HP recognizer

- Reads boss HP
- Multi-digit recognizer

### 1. Video labeler

- Cut HP-bar area of the video
  - Cut not only number, but whole HP bar -> For augmentation
- Save as image
- Label Current HP
- No need for Max HP? - Fixed anyway, no need to recognize

### 2. Model Training

- Model
  - CNN + RNN
  - Mobile Net? -> Need speed (Need to recognize every frames)

---

## Phase 2 - Time recognizer

- Time format is fixed
- Should not need RNN

### 1. Video labeler

- Reuse Video labeler from Phase 1

### 2. Model Training

- Model
  - CNN
    - No need for RNN, predict only 4 numbers

---
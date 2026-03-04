# PDP_IOD
## Beyond Prompt Degradation: Prototype-guided Dual-pool Prompting for Incremental Object Detection

<p align="center">
  <img src="overall.png" width="90%">
</p>

> Official implementation of **PDP**  
> Prototype-guided Dual-pool Prompting for Incremental Object Detection

---

## 🔥 Introduction

Incremental Object Detection (IOD) aims to continuously learn new object categories without forgetting previously learned ones.

Recent prompt-based IOD methods are replay-free and parameter-efficient. However, they suffer from two critical issues:

- **Prompt Coupling**: Task-general and task-specific prompts interfere with each other.
- **Prompt Drift**: Inconsistent supervision causes old categories to degrade.

To address these challenges, we propose **PDP**, a prompt-decoupled continual detection framework that explicitly separates transferable and task-specific knowledge while maintaining supervision consistency across incremental steps.

---

## 🧠 Key Contributions

### 1️⃣ Dual-Pool Prompt Decoupling
- **Shared Prompt Pool**: captures task-general transferable knowledge
- **Private Prompt Pool**: learns task-specific discriminative features
- Explicit decoupling mitigates prompt interference

### 2️⃣ Prototypical Pseudo-Label Generation (PPG)
- Dynamically maintains class prototype space
- Filters teacher-generated pseudo labels using prototype similarity
- Ensures supervision consistency during incremental training

---

## 📊 Results

| Dataset | Setting | Performance Gain |
|----------|----------|----------------|
| MS-COCO | 21+19+20+20 | +9.2% AP |
| PASCAL VOC | 19+1 | +3.3% AP |

PDP achieves state-of-the-art performance while remaining replay-free and parameter-efficient.

---

## 🏗️ Installation

```bash

git clone [https://github.com/yourname/PDP_IOD.git](https://github.com/yourname/PDP_IOD.git)
cd PDP_IOD
conda create -n pdp python=3.8 -y
conda activate pdp
pip install -r requirements.txt

## 🚀 Usage
###  🏋️ Training

```bash

bash run.sh

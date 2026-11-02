# PyTorch Multi-Class Classification

A hands-on PyTorch project that builds, trains, and evaluates a neural network for **multi-class classification** on synthetic 2D "blob" data.

> [!IMPORTANT]
> **This project is part of the lesson _"PyTorch Neural Network Classification"_ from the course [_PyTorch for Deep Learning Bootcamp_](https://github.com/mrdbourke/pytorch-deep-learning) by Daniel Bourke.**
>
> You can also find the link to this project in my practice notebooks repository:
> **[teodorp11/pytorch-practice-notebooks](https://github.com/teodorp11/pytorch-practice-notebooks.git)**

---

## 📚 About the Lesson

| | |
|---|---|
| **Course** | PyTorch for Deep Learning Bootcamp |
| **Instructor** | Daniel Bourke |
| **Lesson** | PyTorch Neural Network Classification |
| **Topic** | Multi-class classification with PyTorch |

---

## 🧠 What This Notebook Covers

The notebook [`multiclass_classification.ipynb`](multiclass_classification.ipynb) walks through the full PyTorch workflow:

1. **Creating data** – Generates 1,000 samples with 2 features and 4 classes using `sklearn.datasets.make_blobs`.

   ✨ **Personal contribution:** Out of curiosity, I extended the lesson by comparing **low (`0.5`), medium (`1.5`), and high (`5`)** `cluster_std` values to see how cluster spread changes the plotted data and how hard the classification becomes.

2. **Visualizing data** – Plots each dataset variant with Matplotlib.

3. **Preparing data** – Converts NumPy arrays to PyTorch tensors and splits them into train/test sets (80/20).

4. **Device-agnostic code** – Runs on CUDA if it's available, otherwise falls back to CPU.

5. **Building a model** – Defines a custom `BlobModel` (`nn.Module`) with configurable hidden units.

6. **Loss & optimizer** – Uses `nn.CrossEntropyLoss` and `torch.optim.SGD`.

7. **Logits → probabilities → labels** – Applies `torch.softmax(dim=1)` and then `argmax`.

8. **Training & testing loop** – Trains for 100 epochs and tracks loss and accuracy.

9. **Evaluation** – Makes predictions in `torch.inference_mode()` and computes accuracy with:
   - a custom `accuracy_fn`
   - `torchmetrics.Accuracy(task="multiclass")`

10. **Decision boundaries** – Plots train/test decision boundaries with `plot_decision_boundary` from the course's `helper_functions.py`.

---

## 🛠️ Tech Stack

- [Python 3](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [TorchMetrics](https://lightning.ai/docs/torchmetrics/)
- [Requests](https://requests.readthedocs.io/)
- [Jupyter](https://jupyter.org/)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/teodorp11/pytorch-multiclass-classification.git
cd pytorch-multiclass-classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch scikit-learn matplotlib torchmetrics requests jupyter
```

> [!TIP]
> If you have an NVIDIA GPU, get the right CUDA build of PyTorch from the [official install selector](https://pytorch.org/get-started/locally/).

### 4. Launch the notebook

```bash
jupyter notebook multiclass_classification.ipynb
```

Or open it in VS Code with the Jupyter extension.

> [!NOTE]
> The notebook downloads `helper_functions.py` from the [course repository](https://github.com/mrdbourke/pytorch-deep-learning) on first run. That file is listed in `.gitignore`, so it isn't committed.

---

## 📁 Project Structure

```
pytorch-multiclass-classification/
├── multiclass_classification.ipynb   # Main notebook
├── README.md                         # Project documentation
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
```

---

## 🙏 Acknowledgements

- **Daniel Bourke** for the excellent [_PyTorch for Deep Learning Bootcamp_](https://github.com/mrdbourke/pytorch-deep-learning) course and the [Learn PyTorch](https://www.learnpytorch.io/) materials.
- `helper_functions.py` comes from [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning).

---

## 🔗 Related

📓 More practice notebooks: [teodorp11/pytorch-practice-notebooks](https://github.com/teodorp11/pytorch-practice-notebooks.git)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤖 Disclaimer

The `README.md` and `.gitignore` files in this repository were created with the help of **Claude Opus 5.5**.
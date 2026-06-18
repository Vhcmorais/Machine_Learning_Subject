#!/usr/bin/env python3
"""
Trabalho 07 - Aproximacao Funcional com MLP (Multilayer Perceptron)
--------------------------------------------------------------------

Arquitetura:
    1 entrada -> H neuronios ocultos (tanh) -> 1 saida (linear)

Pontos amostrados (fornecidos no trabalho):
    x = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    t = [-.9602, -.5770, -.0729, .3771, .6405, .6600,
          .4609,  .1336, -.2013, -.4344, -.5000]
"""

import json
import base64
import io
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dados amostrados
# ---------------------------------------------------------------------------
X_DATA = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
T_DATA = np.array([-.9602, -.5770, -.0729, .3771, .6405, .6600,
                    .4609, .1336, -.2013, -.4344, -.5000])


# ---------------------------------------------------------------------------
# MLP do zero (1 camada oculta)
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, hidden_neurons, seed=42):
        rng = np.random.default_rng(seed)
        self.H = hidden_neurons
        # pesos iniciais pequenos e aleatorios (uniforme em [-0.5, 0.5])
        self.W1 = rng.uniform(-0.5, 0.5, size=(1, hidden_neurons))
        self.b1 = rng.uniform(-0.5, 0.5, size=(1, hidden_neurons))
        self.W2 = rng.uniform(-0.5, 0.5, size=(hidden_neurons, 1))
        self.b2 = rng.uniform(-0.5, 0.5, size=(1, 1))
        # acumuladores de momento
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        # X: (N, 1)
        self.z1 = X @ self.W1 + self.b1          # (N, H)
        self.a1 = np.tanh(self.z1)                # ativacao oculta: tanh
        self.z2 = self.a1 @ self.W2 + self.b2      # (N, 1)
        self.y = self.z2                           # saida linear
        return self.y

    def backward(self, X, T, lr, momentum):
        N = X.shape[0]

        # gradiente da MSE em relacao a saida (ativacao linear -> derivada = 1)
        err = self.y - T
        dz2 = (2.0 / N) * err

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)              # derivada da tanh

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # gradiente descendente com momento
        self.vW2 = momentum * self.vW2 - lr * dW2
        self.vb2 = momentum * self.vb2 - lr * db2
        self.vW1 = momentum * self.vW1 - lr * dW1
        self.vb1 = momentum * self.vb1 - lr * db1

        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

    def predict(self, X):
        return self.forward(X)


def zscore_normalize(x, mean, std):
    return (x - mean) / std


def train_mlp(hidden_neurons, lr, epochs, momentum):
    x_mean = X_DATA.mean()
    x_std = X_DATA.std()

    X = zscore_normalize(X_DATA, x_mean, x_std).reshape(-1, 1)
    T = T_DATA.reshape(-1, 1)

    net = MLP(hidden_neurons)
    mse_history = []

    for _ in range(epochs):
        y_pred = net.forward(X)
        mse_history.append(float(np.mean((y_pred - T) ** 2)))
        net.backward(X, T, lr, momentum)

    final_mse = float(np.mean((net.predict(X) - T) ** 2))

    # curva continua (mais densa) para visualizar a aproximacao
    x_dense = np.linspace(-0.2, 1.2, 400)
    x_dense_norm = zscore_normalize(x_dense, x_mean, x_std).reshape(-1, 1)
    y_dense = net.predict(x_dense_norm).flatten()

    return {
        "mse_history": mse_history,
        "final_mse": final_mse,
        "x_dense": x_dense,
        "y_dense": y_dense,
    }


# ---------------------------------------------------------------------------
# Geracao dos graficos (matplotlib -> PNG em base64)
# ---------------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_samples():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X_DATA, T_DATA, color="#5b3fa0", s=45, zorder=3, label="Pontos amostrados")
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_title("Pontos amostrados")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.grid(alpha=0.25)
    return fig_to_base64(fig)


def plot_fit(x_dense, y_dense):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X_DATA, T_DATA, color="#5b3fa0", s=45, zorder=3, label="Pontos amostrados")
    ax.plot(x_dense, y_dense, color="#e8385d", linewidth=2.5, label="Saida da MLP")
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_title("Aproximacao da MLP")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    return fig_to_base64(fig)


def plot_mse(mse_history):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mse_history, color="#e8385d", linewidth=1.8)
    ax.set_title("Convergencia do erro (MSE)")
    ax.set_xlabel("epoca")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.25)
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Servidor HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open("index.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._set_cors()
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/train":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                params = json.loads(body) if body else {}
                hidden = int(params.get("hidden_neurons", 10))
                lr = float(params.get("learning_rate", 0.05))
                epochs = int(params.get("epochs", 3000))
                momentum = float(params.get("momentum", 0.5))

                hidden = max(1, min(hidden, 200))
                epochs = max(10, min(epochs, 50000))

                result = train_mlp(hidden, lr, epochs, momentum)

                response = {
                    "final_mse": result["final_mse"],
                    "epochs": epochs,
                    "hidden_neurons": hidden,
                    "img_samples": plot_samples(),
                    "img_fit": plot_fit(result["x_dense"], result["y_dense"]),
                    "img_mse": plot_mse(result["mse_history"]),
                }

                payload = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._set_cors()
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._set_cors()
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silencia o log padrao do http.server


def run(port=8765):
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Servidor rodando em http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
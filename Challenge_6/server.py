from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import csv

# ── Adaline ──────────────────────────────────────────────────────────────────
class Adaline:
    def __init__(self, learning_rate=0.01, epochs=1000, tol=1e-6):
        self.lr = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.weights = None
        self.bias = None
        self.errors = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        # ── normalização z-score (evita overflow) ──
        self.x_mean = X.mean(axis=0)
        self.x_std  = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1  # evita divisão por zero
        self.y_mean = y.mean()
        self.y_std  = y.std() or 1.0

        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.errors  = []

        for _ in range(self.epochs):
            output = Xn @ self.weights + self.bias
            error  = yn - output
            mse    = np.mean(error ** 2)

            if not np.isfinite(mse):  # interrompe se divergiu
                break

            self.errors.append(mse)
            self.weights += self.lr * (Xn.T @ error) / n_samples
            self.bias    += self.lr * error.mean()

            if mse < self.tol:
                break

        # ── desnormaliza para escala original ──
        self.weights = self.weights * (self.y_std / self.x_std)
        self.bias    = self.y_mean - (self.x_mean * self.weights).sum()
        return self

    def predict(self, X):
        return np.array(X) @ self.weights + self.bias


# ── Plotting ──────────────────────────────────────────────────────────────────
DARK_BG  = "#0d0f1a"
PANEL    = "#13172a"
ACCENT1  = "#00f5c4"   # teal
ACCENT2  = "#ff5fa0"   # pink
ACCENT3  = "#7c6fff"   # purple
GRID_CLR = "#1e2340"
TEXT_CLR = "#c8d0f0"

def styled_fig(rows=1, cols=2, figsize=(14, 5)):
    fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
    for spine in ['bottom','top','left','right']:
        pass
    return fig

def make_charts(X_raw, y_raw, model):
    X = np.array(X_raw, dtype=float).reshape(-1, 1)
    y = np.array(y_raw, dtype=float)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    margin = (x_max - x_min) * 0.1 or 1
    x_line = np.linspace(x_min - margin, x_max + margin, 300).reshape(-1, 1)
    y_adaline = model.predict(x_line)

    # Standard OLS for comparison
    coeffs = np.polyfit(X[:, 0], y, 1)
    y_ols = np.polyval(coeffs, x_line[:, 0])

    fig = styled_fig(figsize=(15, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.08, right=0.97, top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_edgecolor(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.6, linestyle='--')

    # ── Chart 1: OLS (standard) ──
    ax1.scatter(X[:, 0], y, color=ACCENT2, s=55, zorder=5, alpha=0.9, edgecolors='white', linewidths=0.4)
    ax1.plot(x_line[:, 0], y_ols, color=ACCENT1, linewidth=2.2)
    ax1.set_title("Regressão Linear Padrão (OLS)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("X"); ax1.set_ylabel("y")
    p = mpatches.Patch(color=ACCENT1, label=f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
    ax1.legend(handles=[p], facecolor=DARK_BG, edgecolor=GRID_CLR,
               labelcolor=TEXT_CLR, fontsize=8)

    # ── Chart 2: ADALINE ──
    ax2.scatter(X[:, 0], y, color=ACCENT2, s=55, zorder=5, alpha=0.9, edgecolors='white', linewidths=0.4)
    ax2.plot(x_line[:, 0], y_adaline, color=ACCENT3, linewidth=2.2)
    ax2.set_title("Regressão Linear — ADALINE", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("X"); ax2.set_ylabel("y")
    w = model.weights[0]; b = model.bias
    p2 = mpatches.Patch(color=ACCENT3, label=f"y = {w:.4f}x + {b:.4f}")
    ax2.legend(handles=[p2], facecolor=DARK_BG, edgecolor=GRID_CLR,
               labelcolor=TEXT_CLR, fontsize=8)

    # ── Chart 3: Convergência ──
    ax3.plot(model.errors, color=ACCENT1, linewidth=1.8, alpha=0.85)
    ax3.fill_between(range(len(model.errors)), model.errors, alpha=0.12, color=ACCENT1)
    ax3.set_title("Curva de Convergência do ADALINE (MSE por época)", fontsize=11, fontweight='bold', pad=10)
    ax3.set_xlabel("Épocas"); ax3.set_ylabel("MSE")
    ax3.set_yscale('log')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # silence

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors(); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)
        try:
            data = json.loads(body)
            X = data['X']; y = data['y']
            lr     = float(data.get('lr', 0.01))
            epochs = int(data.get('epochs', 1000))

            model = Adaline(learning_rate=lr, epochs=epochs)
            model.fit([[v] for v in X], y)

            img = make_charts(X, y, model)
            result = {
                'image': img,
                'weight': model.weights[0],
                'bias': model.bias,
                'final_mse': model.errors[-1] if model.errors else None,
                'epochs_run': len(model.errors),
            }
            self._respond(200, result)
        except Exception as e:
            self._respond(400, {'error': str(e)})

    def _respond(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')


if __name__ == '__main__':
    port = 8765
    print(f"🚀  Servidor ADALINE rodando em http://localhost:{port}")
    HTTPServer(('', port), Handler).serve_forever()

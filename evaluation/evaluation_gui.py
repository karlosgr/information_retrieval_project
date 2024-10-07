# evaluation_gui.py

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)
from PyQt5.QtCore import Qt


class EvaluationGUI(QWidget):
    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Evaluación del Sistema de Recomendación")
        self.setGeometry(
            100, 100, 500, 400
        )  # Aumentar tamaño para acomodar más métricas

        # Widgets
        self.k_label = QLabel("Valor de N (Número de recomendaciones):")
        self.k_input = QLineEdit("10")  # Valor por defecto
        self.evaluate_button = QPushButton("Evaluar")
        self.evaluate_button.clicked.connect(self.evaluate)

        self.results_label = QLabel("Resultados:")
        self.results_table = QTableWidget(8, 2)  # Actualizar a 8 filas
        self.results_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionMode(QTableWidget.NoSelection)

        # Definir todas las métricas
        self.metrics = [
            "Precision@k",
            "Recall@k",
            "F1-Score@k",
            "NDCG@k",
            "MAP@k",
            "MRR@k",
            "Hit Rate@k",
            "Coverage",
        ]

        # Layouts
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.k_label)
        input_layout.addWidget(self.k_input)
        input_layout.addWidget(self.evaluate_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.results_label)
        main_layout.addWidget(self.results_table)

        self.setLayout(main_layout)

        # Aplicar estilos (opcional)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #27ae60;
                color: #ecf0f1;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QTableWidget {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: none;
                padding: 5px;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: #ecf0f1;
                font-weight: bold;
            }
        """
        )

    def evaluate(self):
        k_text = self.k_input.text()
        try:
            k = int(k_text)
            if k <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self, "Error", "Por favor, introduce un valor numérico positivo para N."
            )
            return

        # Actualizar N en el evaluador
        self.evaluator.k = k

        # Ejecutar evaluación
        try:
            results = self.evaluator.evaluate()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Ocurrió un error durante la evaluación:\n{str(e)}"
            )
            return

        # Mostrar resultados en la tabla
        self.results_table.setRowCount(
            len(self.metrics)
        )  # Asegurar el número correcto de filas
        for i, metric in enumerate(self.metrics):
            metric_item = QTableWidgetItem(metric)
            metric_item.setTextAlignment(Qt.AlignCenter)
            value = results.get(metric, None)
            if value is not None:
                value_item = QTableWidgetItem(f"{value:.4f}")
            else:
                value_item = QTableWidgetItem("N/A")
            value_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 0, metric_item)
            self.results_table.setItem(i, 1, value_item)

        # Ajustar el tamaño de las columnas
        self.results_table.resizeColumnsToContents()


if __name__ == "__main__":
    # Ejemplo de uso:
    # Aquí debes reemplazar `recommender` y `ratings` con tus propios objetos.
    # from your_recommender_module import Recommender
    # ratings = pd.read_csv("path_to_ratings.csv")
    # recommender = Recommender()
    # evaluator = Evaluator(recommender, ratings, k=10)

    # Por ahora, crearemos un Evaluator de ejemplo con implementaciones vacías.
    class DummyRecommender:
        def update_data(self, train_data):
            pass

        def recommend(self, user_id, top_n=10):
            return []

    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [101, 102, 103, 104, 105, 106],
            "rating": [5, 4, 3, 2, 4, 5],
        }
    )

    recommender = DummyRecommender()
    evaluator = Evaluator(recommender, ratings, k=10)

    app = QApplication(sys.argv)
    gui = EvaluationGUI(evaluator)
    gui.show()
    sys.exit(app.exec_())

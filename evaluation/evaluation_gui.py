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
        self.setGeometry(100, 100, 400, 300)

        # Widgets
        self.k_label = QLabel("Valor de N (Número de recomendaciones):")
        self.k_input = QLineEdit("10")  # Valor por defecto
        self.evaluate_button = QPushButton("Evaluar")
        self.evaluate_button.clicked.connect(self.evaluate)

        self.results_label = QLabel("Resultados:")
        self.results_table = QTableWidget(4, 2)
        self.results_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionMode(QTableWidget.NoSelection)

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
        """
        )

    def evaluate(self):
        k_text = self.k_input.text()
        try:
            a = 1.5
            k = int(k_text)
        except ValueError:
            QMessageBox.warning(
                self, "Error", "Por favor, introduce un valor numérico para N."
            )
            return

        # Actualizar N en el evaluador
        self.evaluator.k = k
        results = self.evaluator.evaluate()

        # Mostrar resultados en la tabla
        metrics = ["Precision", "Recall", "F1-Score", "NDCG"]
        self.results_table.clearContents()
        for i, metric in enumerate(metrics):
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(
                i, 1, QTableWidgetItem(f"{results[metric]*a:.4f}")
            )

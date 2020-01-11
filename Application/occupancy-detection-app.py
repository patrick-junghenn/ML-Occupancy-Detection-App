import sys
import pandas as pd
import numpy as np
#pyqt5
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox,QRadioButton, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPlainTextEdit, QSlider, QDialog, QVBoxLayout, QSizePolicy, QMessageBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import (pyqtSlot, pyqtSignal)
from mlxtend.plotting import plot_decision_regions
from matplotlib.colors import ListedColormap
#plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
#modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from pydotplus import graph_from_dot_data
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
# Libraries to display decision tree
import webbrowser
import random
import warnings
warnings.filterwarnings("ignore")

font_size_window = 'font-size:15px;'


class RandomForest(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        #box1
        self.groupBox1 = QGroupBox('Features')
        self.groupBox1Layout = QGridLayout()  # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.split_size_label = QLabel('Train-Test Split Size:')
        self.split_size_label.adjustSize()
        self.split_size_edit = QLineEdit(self)
        self.split_size_edit.setText("30")
        self.btnExecute = QPushButton("Run")
        self.btnExecute.clicked.connect(self.update)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.split_size_label, 3, 0)
        self.groupBox1Layout.addWidget(self.split_size_edit, 3, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 4, 0)
        #box2
        self.groupBox2 = QGroupBox('Model Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.results_label = QLabel('Results:')
        self.results_label.adjustSize()
        self.results_output = QPlainTextEdit()
        self.accuracy_label = QLabel('Accuracy:')
        self.accuracy_output = QLineEdit()
        self.groupBox2Layout.addWidget(self.results_label)
        self.groupBox2Layout.addWidget(self.results_output)
        self.groupBox2Layout.addWidget(self.accuracy_label)
        self.groupBox2Layout.addWidget(self.accuracy_output)

        #confusion matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        #ROC curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        # Feature Importance
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()
        #box3
        self.groupBoxG3 = QGroupBox('Features Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)
        self.groupBox3 = QGroupBox('Hyperparameters')
        self.groupBox3Layout = QGridLayout()  # Grid
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.nsamps_label = QLabel('Number of Estimators:')
        self.nsamps_label.adjustSize()
        self.nsamps = QLineEdit(self)
        self.nsamps.setText("100")
        self.maxD_lab = QLabel('Max Depth:')
        self.maxD_lab.adjustSize()
        self.maxD = QLineEdit(self)
        self.maxD.setText("12")
        self.min_s_lab = QLabel('Minimum Samples Split')
        self.min_s_lab.adjustSize()
        self.min_s = QLineEdit(self)
        self.min_s.setText("2")
        self.min_s_ll = QLabel('Minimum Samples Leaf')
        self.min_s_ll.adjustSize()
        self.min_s_l = QLineEdit(self)
        self.min_s_l.setText("1")
        self.groupBox3Layout.addWidget(self.maxD_lab, 0, 0)
        self.groupBox3Layout.addWidget(self.maxD, 0, 1)
        self.groupBox3Layout.addWidget(self.nsamps_label, 1, 0)
        self.groupBox3Layout.addWidget(self.nsamps, 1, 1)
        self.groupBox3Layout.addWidget(self.min_s_lab, 2, 0)
        self.groupBox3Layout.addWidget(self.min_s, 2, 1)
        self.groupBox3Layout.addWidget(self.min_s_ll, 3, 0)
        self.groupBox3Layout.addWidget(self.min_s_l, 3, 1)
        #arrange
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 2)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 1)
        self.layout.addWidget(self.groupBoxG3, 1, 2)
        self.layout.addWidget(self.groupBox3, 1, 0)

        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.new_occ_data = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[0]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[1]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[2]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[3]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[4]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)
        split = float(self.split_size_edit.text())
        n_ests = int(self.nsamps.text())
        max_d = int(self.maxD.text())
        min_split = int(self.min_s.text())
        min_samp_l = int(self.min_s_l.text())

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.results_output.clear()
        self.results_output.setUndoRedoEnabled(False)

        split = split / 100

        X_rf = self.new_occ_data
        y_rf = occ_data['Occupancy']
        class_le = LabelEncoder()
        y_rf = class_le.fit_transform(y_rf)
        X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=split, random_state=100)
        self.rf_clf = RandomForestClassifier(n_estimators=n_ests, random_state=100, max_depth=max_d,
                                             min_samples_split=min_split, min_samples_leaf= min_samp_l)
        self.rf_clf.fit(X_train, y_train)
        rf_pred = self.rf_clf.predict(X_test)
        rf_pred_score = self.rf_clf.predict_proba(X_test)
        #diognostics
        conf_matrix = confusion_matrix(y_test, rf_pred)
        self.class_report = classification_report(y_test, rf_pred)
        self.results_output.appendPlainText(self.class_report)
        self.rf_acc = accuracy_score(y_test, rf_pred)
        self.accuracy_output.setText(str(self.rf_acc))

        # Confusion Matrix
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('rainbow', 14))
        self.ax1.set_yticklabels(states)
        self.ax1.set_xticklabels(states, rotation=90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        for i in range(len(states)):
            for j in range(len(states)):
                rf_pred_score = self.rf_clf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #ROC Curve
        fpr, tpr, _ = roc_curve(y_test, rf_pred)
        auc = roc_auc_score(y_test, rf_pred)
        lw = 2
        self.ax2.plot(fpr, tpr, color='blue',
                      lw=lw, label='ROC curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='magenta', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Random Forest')
        self.ax2.legend(loc="lower right")
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #Feature Importances
        importances = self.rf_clf.feature_importances_
        f_importances = pd.Series(importances, self.new_occ_data.columns)
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
        X_feats = f_importances.index
        contrib = list(f_importances)
        self.ax3.barh(X_feats, contrib)
        self.ax3.set_aspect('auto')
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()
        plt.tight_layout()
        plt.show()

class DecisionTree(QMainWindow):
    '''decision tree model class'''
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.Title = "Decision Tree Classifier"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        #box1
        self.groupBox1 = QGroupBox('Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.split_size_label = QLabel('Train-Test Split Size:')
        self.split_size_label.adjustSize()
        self.split_size_edit = QLineEdit(self)
        self.split_size_edit.setText("30")
        self.btnExecute = QPushButton("Run")
        self.btnExecute.clicked.connect(self.update)
        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.split_size_label, 3, 0)
        self.groupBox1Layout.addWidget(self.split_size_edit, 3, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 4, 0)
        self.groupBox1Layout.addWidget(self.btnDTFigure, 4, 1)
        #groupbox 2
        self.groupBox2 = QGroupBox('Model Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.results_label = QLabel('Results:')
        self.results_label.adjustSize()
        self.results_output = QPlainTextEdit()
        self.accuracy_label = QLabel('Accuracy:')
        self.accuracy_output = QLineEdit()
        self.groupBox2Layout.addWidget(self.results_label)
        self.groupBox2Layout.addWidget(self.results_output)
        self.groupBox2Layout.addWidget(self.accuracy_label)
        self.groupBox2Layout.addWidget(self.accuracy_output)
        #groupbox 3
        self.groupBox3 = QGroupBox('Hyperparameters')
        self.groupBox3Layout = QGridLayout()  # Grid
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.maxD_lab = QLabel('Max Depth')
        self.maxD_lab.adjustSize()
        self.maxD = QLineEdit(self)
        self.maxD.setText('12')
        self.max_leaf_lab = QLabel('Max Leaf Nodes')
        self.max_leaf_lab.adjustSize()
        self.max_leaf = QLineEdit(self)
        self.max_leaf.setText("5")
        self.min_samps_lab = QLabel('Minimum Sample Leafs')
        self.min_samps_lab.adjustSize()
        self.min_samps = QLineEdit(self)
        self.min_samps.setText('1')
        self.groupBox3Layout.addWidget(self.maxD_lab, 0, 0)
        self.groupBox3Layout.addWidget(self.maxD, 0, 1)
        self.groupBox3Layout.addWidget(self.max_leaf_lab, 1, 0)
        self.groupBox3Layout.addWidget(self.max_leaf, 1, 1)
        self.groupBox3Layout.addWidget(self.min_samps_lab, 2, 0)
        self.groupBox3Layout.addWidget(self.min_samps, 2, 1)

        # Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        # ROC Curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        # feature importance
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()
        self.groupBoxG3 = QGroupBox('Features Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)
        # arrange
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 2)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 1)
        self.layout.addWidget(self.groupBoxG3, 1, 2)
        self.layout.addWidget(self.groupBox3, 1, 0)

        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.new_occ_data = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[0]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[1]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[2]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[3]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[4]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)
        split = float(self.split_size_edit.text())
        depth = float(self.maxD.text())
        max_leaf = int(self.max_leaf.text())
        min_samps = int(self.min_samps.text())

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.results_output.clear()
        self.results_output.setUndoRedoEnabled(False)

        split = split / 100

        X_ = self.new_occ_data
        y_ = occ_data[target]
        class_le = LabelEncoder()
        #encode y1
        y_ = class_le.fit_transform(y_)
        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=split, random_state=42)
        self.dt_clf = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=depth,
                                             min_samples_leaf=min_samps, max_leaf_nodes=max_leaf)
        self.dt_clf.fit(X_train, y_train)
        dt_pred = self.dt_clf.predict(X_test)
        #diognosttics
        conf_matrix = confusion_matrix(y_test, dt_pred)
        self.report = classification_report(y_test, dt_pred)
        self.results_output.appendPlainText(self.report)
        self.dt_acc = accuracy_score(y_test, dt_pred)
        self.accuracy_output.setText(str(self.dt_acc))
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('rainbow', 14))
        self.ax1.set_yticklabels(states)
        self.ax1.set_xticklabels(states, rotation=90)

        for i in range(len(states)):
            for j in range(len(states)):
                dt_pred_score = self.dt_clf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #ROC Curve
        fpr, tpr, _ = roc_curve(y_test, dt_pred)
        auc = roc_auc_score(y_test, dt_pred)
        lw = 2
        self.ax2.plot(fpr, tpr, color='blue',
                      lw=lw, label='ROC curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='magenta', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Decision Tree')
        self.ax2.legend(loc="lower right")
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        # feature importances
        importances = self.dt_clf.feature_importances_
        f_importances = pd.Series(importances, self.new_occ_data.columns)
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances.plot(x='Feature', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
        X_feats = f_importances.index
        contrib = list(f_importances)
        self.ax3.barh(X_feats, contrib)
        self.ax3.set_aspect('auto')
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

    def view_tree(self):
        dot_data = export_graphviz(self.dt_clf, filled=True, rounded=True, class_names=states,
                                   feature_names=self.new_occ_data.columns, out_file=None)
        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_plot.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')

class LogReg(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(LogReg, self).__init__()
        self.Title = "Logistic Regression Classifier"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        #box1
        self.groupBox1 = QGroupBox('Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.split_size_label = QLabel('Train-Test Split Size:')
        self.split_size_label.adjustSize()
        self.split_size_edit = QLineEdit(self)
        self.split_size_edit.setText("30")
        self.btnExecute = QPushButton("Run")
        self.btnExecute.clicked.connect(self.update)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.split_size_label, 3, 0)
        self.groupBox1Layout.addWidget(self.split_size_edit, 3, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 4, 0)
        #box2
        self.groupBox2 = QGroupBox('Model Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.results_label = QLabel('Results')
        self.results_label.adjustSize()
        self.results_output = QPlainTextEdit()
        self.accuracy_label = QLabel('Accuracy')
        self.accuracy_output = QLineEdit()
        self.groupBox2Layout.addWidget(self.results_label)
        self.groupBox2Layout.addWidget(self.results_output)
        self.groupBox2Layout.addWidget(self.accuracy_label)
        self.groupBox2Layout.addWidget(self.accuracy_output)

        # Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        # ROC Curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 2)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 1)

        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.new_occ_data = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[0]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[1]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[2]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[3]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[4]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)

        split = float(self.split_size_edit.text())
        self.ax1.clear()
        self.ax2.clear()
        self.results_output.clear()
        self.results_output.setUndoRedoEnabled(False)

        split = split / 100

        X1 = self.new_occ_data
        y1 = occ_data[target]

        class_le = LabelEncoder()
        # fit and transform the class
        y1 = class_le.fit_transform(y1)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=split, random_state=100)
        self.lr_clf = LogisticRegression()
        self.lr_clf.fit(X_train, y_train)
        lr_pred = self.lr_clf.predict(X_test)
        # Diognostics
        conf_matrix = confusion_matrix(y_test, lr_pred)
        self.report = classification_report(y_test, lr_pred)
        self.results_output.appendPlainText(self.report)
        self.lr_acc = accuracy_score(y_test, lr_pred)
        self.accuracy_output.setText(str(self.lr_acc))

        # Confusion Matrix
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('rainbow', 14))
        self.ax1.set_yticklabels(states)
        self.ax1.set_xticklabels(states, rotation=90)

        for i in range(len(states)):
            for j in range(len(states)):
                lr_pred_score = self.lr_clf.predict(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #ROC curve
        fpr, tpr, _ = roc_curve(y_test, lr_pred)
        auc = roc_auc_score(y_test, lr_pred)
        lw = 2
        self.ax2.plot(fpr, tpr, color='blue',
                      lw=lw, label='ROC curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='magenta', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Logistic Regression')
        self.ax2.legend(loc="lower right")
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class kNearestNeighbors(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(kNearestNeighbors, self).__init__()
        self.Title = "K-Nearest Neighbors Classifier"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        #box1
        self.groupBox1 = QGroupBox('Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        #split size
        self.split_size_label = QLabel('Train-Test Split Size:')
        self.split_size_label.adjustSize()
        self.split_size_edit = QLineEdit(self)
        self.split_size_edit.setText("30")
        #execute button
        self.btnExecute = QPushButton("Run")
        self.btnExecute.clicked.connect(self.update)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.split_size_label, 3, 0)
        self.groupBox1Layout.addWidget(self.split_size_edit, 3, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 4, 0)
        #box2
        self.groupBox2 = QGroupBox('Model Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.results_label = QLabel('Results:')
        self.results_label.adjustSize()
        self.results_output = QPlainTextEdit()
        self.accuracy_label = QLabel('Accuracy:')
        self.accuracy_output = QLineEdit()
        self.groupBox2Layout.addWidget(self.results_label)
        self.groupBox2Layout.addWidget(self.results_output)
        self.groupBox2Layout.addWidget(self.accuracy_label)
        self.groupBox2Layout.addWidget(self.accuracy_output)
        #box3
        self.groupBox3 = QGroupBox('Hyperparameters')
        self.groupBox3Layout = QGridLayout()  # Grid
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.k_label = QLabel('K:')
        self.k_label.adjustSize()
        self.k = QLineEdit(self)
        self.k.setText("4")
        self.groupBox3Layout.addWidget(self.k_label, 0, 0)
        self.groupBox3Layout.addWidget(self.k, 1, 0)

        # Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        # ROC Curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 2)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 1)
        self.layout.addWidget(self.groupBox3, 1, 0)
        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.new_occ_data = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[0]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[1]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[2]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[3]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[4]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)
        split = float(self.split_size_edit.text())
        kNeighbors = int(self.k.text())

        self.ax1.clear()
        self.ax2.clear()
        self.results_output.clear()
        self.results_output.setUndoRedoEnabled(False)

        split = split / 100
        X1 = self.new_occ_data
        y1 = occ_data[target]

        class_le = LabelEncoder()
        # fit and transform the class
        y1 = class_le.fit_transform(y1)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=split, random_state=100)
        self.knn_clf = KNeighborsClassifier(n_neighbors=kNeighbors)
        self.knn_clf.fit(X_train, y_train)
        knn_pred = self.knn_clf.predict(X_test)

        #Diognostics
        conf_matrix = confusion_matrix(y_test, knn_pred)
        self.report = classification_report(y_test, knn_pred)
        self.results_output.appendPlainText(self.report)
        self.knn_acc = accuracy_score(y_test, knn_pred)
        self.accuracy_output.setText(str(self.knn_acc))

        # Confusion Matrix
        target_classes = ['Vacant', 'Occupied']
        self.ax1.set_xlabel('Predicted')
        self.ax1.set_ylabel('True')
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('rainbow', 14))
        self.ax1.set_yticklabels(target_classes)
        self.ax1.set_xticklabels(target_classes, rotation=90)

        for i in range(len(target_classes)):
            for j in range(len(target_classes)):
                knn_pred_score = self.knn_clf.predict(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, knn_pred)
        auc = roc_auc_score(y_test, knn_pred_score)
        lw = 2
        self.ax2.plot(fpr, tpr, color='blue',
                      lw=lw, label='ROC curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='magenta', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive')
        self.ax2.set_ylabel('True Positive')
        self.ax2.set_title('ROC Curve K-Nearest Neighbors')
        self.ax2.legend(loc="lower right")
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class XGboost(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(XGboost, self).__init__()
        self.Title = "XGBoost Classifier"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        #box1
        self.groupBox1 = QGroupBox('Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.split_size_label = QLabel('Test Data Size:')
        self.split_size_label.adjustSize()
        self.split_size_edit = QLineEdit(self)
        self.split_size_edit.setText("30")
        self.btnExecute = QPushButton("Run")
        self.btnExecute.clicked.connect(self.update)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.split_size_label, 3, 0)
        self.groupBox1Layout.addWidget(self.split_size_edit, 3, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 4, 0)
        #box2
        self.groupBox2 = QGroupBox('Model Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.results_label = QLabel('Results:')
        self.results_label.adjustSize()
        self.results_output = QPlainTextEdit()
        self.accuracy_label = QLabel('Accuracy:')
        self.accuracy_output = QLineEdit()
        self.groupBox2Layout.addWidget(self.results_label)
        self.groupBox2Layout.addWidget(self.results_output)
        self.groupBox2Layout.addWidget(self.accuracy_label)
        self.groupBox2Layout.addWidget(self.accuracy_output)
        #box3
        self.groupBox3 = QGroupBox('Hyperparameters')
        self.groupBox3Layout = QGridLayout()  # Grid
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.bdepth_label = QLabel('Max Depth:')
        self.bdepth_label.adjustSize()
        self.bdepth = QLineEdit(self)
        self.bdepth.setText('12')
        self.l_rate_label = QLabel('Learning Rate')
        self.l_rate = QLineEdit(self)
        self.l_rate.setText('0.3')
        self.cw_label = QLabel('Minimum Child Weight')
        self.cw = QLineEdit(self)
        self.cw.setText('5')
        self.subsamp_label = QLabel('Sub Sample Size')
        self.subsamp = QLineEdit(self)
        self.subsamp.setText('0.5')
        self.ests_label = QLabel('Number of Estimators')
        self.ests = QLineEdit(self)
        self.ests.setText('100')
        self.groupBox3Layout.addWidget(self.bdepth_label, 0, 0)
        self.groupBox3Layout.addWidget(self.bdepth, 0, 1)
        self.groupBox3Layout.addWidget(self.l_rate_label, 1, 0)
        self.groupBox3Layout.addWidget(self.l_rate, 1, 1)
        self.groupBox3Layout.addWidget(self.cw_label, 2, 0)
        self.groupBox3Layout.addWidget(self.cw, 2, 1)
        self.groupBox3Layout.addWidget(self.subsamp_label, 3, 0)
        self.groupBox3Layout.addWidget(self.subsamp, 3, 1)
        self.groupBox3Layout.addWidget(self.ests_label, 4, 0)
        self.groupBox3Layout.addWidget(self.ests, 4, 1)

        #confusion matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        #ROC Curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        # Feature Importance
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()
        self.groupBoxG3 = QGroupBox('Features Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 2)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 1)
        self.layout.addWidget(self.groupBoxG3, 1, 2)
        self.layout.addWidget(self.groupBox3, 1, 0)

        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.new_occ_data = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[0]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[1]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[2]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[3]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[4]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)

        split = float(self.split_size_edit.text())
        n_ests = int(self.ests.text())
        depth = int(self.bdepth.text())
        learn_rate = float(self.l_rate.text())
        child_weight = int(self.cw.text())
        sub_sample = float(self.subsamp.text())

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.results_output.clear()
        self.results_output.setUndoRedoEnabled(False)

        split = split / 100

        X1 = self.new_occ_data
        y1 = occ_data[target]

        class_le = LabelEncoder()
        # encode y1
        y1 = class_le.fit_transform(y1)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=split, random_state=100)
        self.xgb_clf = XGBClassifier(max_depth=depth, learning_rate=learn_rate, min_child_weight=child_weight, subsample= sub_sample, n_estimators=n_ests)
        self.xgb_clf.fit(X_train, y_train)
        xgb_pred = self.xgb_clf.predict(X_test)
        #diognostics
        conf_matrix = confusion_matrix(y_test, xgb_pred)
        self.report = classification_report(y_test, xgb_pred)
        self.results_output.appendPlainText(self.report)
        self.xgb_acc = accuracy_score(y_test, xgb_pred)
        self.accuracy_output.setText(str(self.xgb_acc))

        #confusion matrix
        target_classes = ['Vacant','Occupied']
        self.ax1.set_xlabel('Predicted')
        self.ax1.set_ylabel('True')
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('rainbow', 14))
        self.ax1.set_yticklabels(target_classes)
        self.ax1.set_xticklabels(target_classes, rotation=45)
        for i in range(len(target_classes)):
            for j in range(len(target_classes)):
                xg_score = self.xgb_clf.predict(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # ROC
        fpr, tpr, _ = roc_curve(y_test, xgb_pred)
        auc = roc_auc_score(y_test, xgb_pred)
        lw = 2
        self.ax2.plot(fpr, tpr, color='blue',
                      lw=lw, label='ROC curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='magenta', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive')
        self.ax2.set_ylabel('True Positive')
        self.ax2.set_title('ROC Curve XGBoost')
        self.ax2.legend(loc="lower right")
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        # feature imporatnce
        importances = self.xgb_clf.feature_importances_
        f_importances = pd.Series(importances, self.new_occ_data.columns)
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
        X_feats = f_importances.index
        contrib = list(f_importances)
        self.ax3.barh(X_feats, contrib)
        self.ax3.set_aspect('auto')
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()
        plt.tight_layout()
        plt.show()

class adaBoost(QMainWindow):
    '''adaBoost model class'''
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(adaBoost, self).__init__()
        self.Title = "AdaBoost Classifier"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        #box1
        self.groupBox1 = QGroupBox('Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.split_size_label = QLabel('Train-Test Split Size:')
        self.split_size_label.adjustSize()
        self.split_size_edit = QLineEdit(self)
        self.split_size_edit.setText("30")
        self.btnExecute = QPushButton("Run")
        self.btnExecute.clicked.connect(self.update)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.split_size_label, 3, 0)
        self.groupBox1Layout.addWidget(self.split_size_edit, 3, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 4, 0)
        #box2
        self.groupBox2 = QGroupBox('Model Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.results_label = QLabel('Results:')
        self.results_label.adjustSize()
        self.results_output = QPlainTextEdit()
        self.accuracy_label = QLabel('Accuracy:')
        self.accuracy_output = QLineEdit()
        self.groupBox2Layout.addWidget(self.results_label)
        self.groupBox2Layout.addWidget(self.results_output)
        self.groupBox2Layout.addWidget(self.accuracy_label)
        self.groupBox2Layout.addWidget(self.accuracy_output)
        #box3
        self.groupBox3 = QGroupBox('Hyperparameters')
        self.groupBox3Layout = QGridLayout()  # Grid
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.ests_lab = QLabel('Number of estimators')
        self.ests_lab.adjustSize()
        self.ests = QLineEdit(self)
        self.ests.setText("100")
        self.lrate_lab = QLabel('Learning Rate')
        self.lrate_lab.adjustSize()
        self.lrate = QLineEdit(self)
        self.lrate.setText("0.3")
        self.groupBox3Layout.addWidget(self.ests_lab, 0, 0)
        self.groupBox3Layout.addWidget(self.ests, 0, 1)
        self.groupBox3Layout.addWidget(self.ests_lab, 1, 0)
        self.groupBox3Layout.addWidget(self.ests, 1, 1)
        self.groupBox3Layout.addWidget(self.lrate_lab, 2, 0)
        self.groupBox3Layout.addWidget(self.lrate, 2, 1)

        # Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        # roc
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        # importance
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()
        self.groupBoxG3 = QGroupBox('Features Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 2)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 1)
        self.layout.addWidget(self.groupBoxG3, 1, 2)
        self.layout.addWidget(self.groupBox3, 1, 0)
        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.new_occ_data = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[0]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[1]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[2]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[3]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            if len(self.new_occ_data) == 0:
                self.new_occ_data = occ_data[feats[4]]
            else:
                self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)
        split = float(self.split_size_edit.text())
        estimators = int(self.ests.text())
        learn_r = float(self.lrate.text())
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.results_output.clear()
        self.results_output.setUndoRedoEnabled(False)

        # modeling
        split = split / 100
        X1 = self.new_occ_data
        y1 = occ_data[target]
        class_le = LabelEncoder()
        y1 = class_le.fit_transform(y1)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=split, random_state=100)
        self.adb_clf = AdaBoostClassifier(DecisionTreeClassifier())
        self.adb_clf.fit(X_train, y_train)
        adb_pred = self.adb_clf.predict(X_test)
        #Diognostics
        conf_matrix = confusion_matrix(y_test, adb_pred)
        self.report = classification_report(y_test, adb_pred)
        self.results_output.appendPlainText(self.report)
        self.adb_acc = accuracy_score(y_test, adb_pred)
        self.accuracy_output.setText(str(self.adb_acc))

        # confusion matrix
        target_classes = ['Vacant', 'Occupied']
        self.ax1.set_xlabel('Predicted')
        self.ax1.set_ylabel('True')
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('rainbow', 14))
        self.ax1.set_yticklabels(target_classes)
        self.ax1.set_xticklabels(target_classes, rotation=45)
        for i in range(len(target_classes)):
            for j in range(len(target_classes)):
                ab_score = self.adb_clf.predict(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, adb_pred)
        auc = roc_auc_score(y_test, adb_pred)
        lw = 2
        self.ax2.plot(fpr, tpr, color='blue',
                      lw=lw, label='ROC curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='magenta', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive')
        self.ax2.set_ylabel('True Positive')
        self.ax2.set_title('ROC Curve XGBoost')
        self.ax2.legend(loc="lower right")
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #Importances
        importances = self.adb_clf.feature_importances_
        f_importances = pd.Series(importances, self.new_occ_data.columns)
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
        X_feats = f_importances.index
        contrib = list(f_importances)
        self.ax3.barh(X_feats, contrib)
        self.ax3.set_aspect('auto')
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()
        plt.tight_layout()
        plt.show()

class CorrelationPlot(QMainWindow):
    '''class creates correlation plot'''
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(CorrelationPlot, self).__init__()
        self.Title = 'Correlation Plot'
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)
        # box1
        self.groupBox1 = QGroupBox('Correlation Plot Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)  # check boxes
        self.feature0 = QCheckBox(feats[0], self)
        self.feature1 = QCheckBox(feats[1], self)
        self.feature2 = QCheckBox(feats[2], self)
        self.feature3 = QCheckBox(feats[3], self)
        self.feature4 = QCheckBox(feats[4], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.btnExecute, 3, 0)
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        # box 2
        self.groupBox2 = QGroupBox('Correlation Plot')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2Layout.addWidget(self.canvas)
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()
        self.update()

    def update(self):
        self.ax1.clear()
        self.new_occ_data = pd.DataFrame(occ_data[target])
        if self.feature0.isChecked():
            self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[0]]], axis=1)
        if self.feature1.isChecked():
            self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[1]]], axis=1)
        if self.feature2.isChecked():
            self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[2]]], axis=1)
        if self.feature3.isChecked():
            self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[3]]], axis=1)
        if self.feature4.isChecked():
            self.new_occ_data = pd.concat([self.new_occ_data, occ_data[feats[4]]], axis=1)
        vsticks = ["dummy"]
        vsticks1 = list(self.new_occ_data.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = self.new_occ_data.corr()
        self.ax1.matshow(res_corr, cmap=plt.cm.get_cmap('RdBu_r', 14))
        self.ax1.set_yticklabels(vsticks1)
        self.ax1.set_xticklabels(vsticks1, rotation=90)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class occPredictor(QMainWindow):
    '''class to predict occupancy'''
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(occPredictor, self).__init__()
        self.Title = "Occupancy Prediction"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)
        # box1
        self.groupBox1 = QGroupBox('Predict Occupancy')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.pred_temp_label = QLabel('Temperature')
        self.pred_temp = QLineEdit(self)
        self.pred_humid_lable = QLabel('Humidity Measure')
        self.pred_humid = QLineEdit(self)
        self.pred_light_label = QLabel('Light Measure:')
        self.pred_light = QLineEdit(self)
        self.pred_co2_lable = QLabel('CO2 Measure:')
        self.pred_co2 = QLineEdit(self)
        self.pred_humid_rat_label = QLabel('Humidity Ratio Measure:')
        self.pred_humid_rat = QLineEdit(self)
        self.btnPredict = QPushButton("Determine Occupancy")
        self.pred_output_label = QLabel('Prediction:')
        self.btnPredict.clicked.connect(self.update)
        self.pred_output = QLineEdit(self)
        self.groupBox1Layout.addWidget(self.pred_temp_label, 0, 0)
        self.groupBox1Layout.addWidget(self.pred_temp, 0, 1)
        self.groupBox1Layout.addWidget(self.pred_humid_lable, 1, 0)
        self.groupBox1Layout.addWidget(self.pred_humid, 1, 1)
        self.groupBox1Layout.addWidget(self.pred_light_label, 2, 0)
        self.groupBox1Layout.addWidget(self.pred_light, 2, 1)
        self.groupBox1Layout.addWidget(self.pred_co2_lable, 3, 0)
        self.groupBox1Layout.addWidget(self.pred_co2, 3, 1)
        self.groupBox1Layout.addWidget(self.pred_humid_rat_label, 4, 0)
        self.groupBox1Layout.addWidget(self.pred_humid_rat, 4, 1)
        self.groupBox1Layout.addWidget(self.btnPredict, 5, 0)
        self.groupBox1Layout.addWidget(self.pred_output_label, 6, 0)
        self.groupBox1Layout.addWidget(self.pred_output, 6, 1)
        # Box 2
        self.groupBox2 = QGroupBox('Predictive Models')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.dt_opt = QRadioButton(mods[0], self)
        self.rf_opt = QRadioButton(mods[1], self)
        self.lr_opt = QRadioButton(mods[2], self)
        self.knn_opt = QRadioButton(mods[3], self)
        self.adb_opt = QRadioButton(mods[5], self)
        self.dt_opt.setChecked(False)
        self.rf_opt.setChecked(True)
        self.lr_opt.setChecked(False)
        self.knn_opt.setChecked(False)
        self.adb_opt.setChecked(False)
        self.groupBox2Layout.addWidget(self.dt_opt, 0, 0)
        self.groupBox2Layout.addWidget(self.rf_opt, 0, 1)
        self.groupBox2Layout.addWidget(self.lr_opt, 1, 0)
        self.groupBox2Layout.addWidget(self.knn_opt, 1, 1)
        self.groupBox2Layout.addWidget(self.adb_opt, 2, 1)
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()

    def update(self):

        self.classifier = []
        if self.dt_opt.isChecked():
            self.classifier = DecisionTreeClassifier(max_depth=10)
        elif self.rf_opt.isChecked():
            self.classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
        elif self.lr_opt.isChecked():
            self.classifier = LogisticRegression()
        elif self.knn_opt.isChecked():
            self.classifier = KNeighborsClassifier(n_neighbors=3)
        elif self.adb_opt.isChecked():
            self.classifier = AdaBoostClassifier(n_estimators=100)
        # construct classifier
        X_ = occ_data[feats]
        y_ = occ_data[target]
        class_le = LabelEncoder()
        # encode y
        y_ = class_le.fit_transform(y_)
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=30, random_state=100)
        self.classifier.fit(X_train, y_train)
        temp_input = float(self.pred_temp.text())
        humid_input = float(self.pred_humid.text())
        light_input = float(self.pred_light.text())
        co2_input = float(self.pred_co2.text())
        humid_rat_input = float(self.pred_humid_rat.text())
        self.occupancy_pred = self.classifier.predict(
            [[temp_input, humid_input, light_input, co2_input, humid_rat_input]])

        if self.occupancy_pred == 0:
            self.pred_output.setText(str('Vacant'))
        elif self.occupancy_pred == 1:
            self.pred_output.setText(str('Occupied'))

class App(QMainWindow):
    '''creates application'''
    def __init__(self):
        super(App, self).__init__()
        self.Title = 'Occupancy Detection Software'
        self.initUI()

    def initUI(self):
        self.main_widget = QWidget(self)
        self.setWindowTitle(self.Title)
        self.setStyleSheet('QGroupBox {font: 21pt Helvetica Bold; color: blue}, QLabel {font: 16pt Helvetica;}')
        self.layout = QGridLayout(self.main_widget)
        #box2
        self.groupBox2 = QGroupBox('Occupancy Detection Application')
        self.groupBox2Layout = QGridLayout()
        self.occ_info = QLabel('This software utilizes multiple machine learning algorithms to make predictions'
                               ' concerning the occupancy of a space. The software is built-in python and uses the'
                               ' scikit learn library for all machine learning operations.')
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2Layout.addWidget(self.occ_info, 0, 0)
        #Box3
        self.groupBox3 = QGroupBox('Analyze Distribution')
        self.groupBox3Layout = QGridLayout()
        self.analysis_info = QLabel(
            'The "Analyze Distribution" menu option contains a correlation matrix tool.'
            ' Features can be selected using the checkbox to explore different relationships')
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3Layout.addWidget(self.analysis_info, 0, 0)
        #box4
        self.groupBox4 = QGroupBox('Model Selection')
        self.groupBox4Layout = QGridLayout()
        self.model_info = QLabel('All of the machine learning models offered through this software are located in'
                                 ' the "Model Selection" menu option. Each model contains the relevant hyperparameter'
                                 ' options and diagnostics.\n\nAvailable Models:\n\n-Decision Tree Classifier'
                                 '\n\n-Random Forest Classifier\n\n-Logistic Regression \n\n-K-nearest Neighbors'
                                 ' \n\n-XGBoost\n\n-AdaBoost')
        self.models = QLabel()
        self.groupBox4.setLayout(self.groupBox4Layout)
        self.groupBox4Layout.addWidget(self.model_info, 0, 0)
        self.groupBox4Layout.addWidget(self.models, 0, 0)
        #box5
        self.groupBox5 = QGroupBox('Predict Occupancy Status')
        self.groupBox5Layout = QGridLayout()
        self.predict = QLabel(
            'To predict the occupancy status of a space, input relevant readings from an occupancy detector.'
            ' \n\nPredictions are based on the following fields:\n\n-Temperature \n\n-Humidity'
            ' \n\n-Light \n\n-CO2 \n\n-Humidity ratio\n\nTo use the prediction system, click the'
            ' "Predict Occupancy Status" menu option, select the desired model and click the'
            ' "Predict" button. The result will be show below')
        self.groupBox5.setLayout(self.groupBox5Layout)
        self.groupBox5Layout.addWidget(self.predict)
        #arrangement
        self.layout.addWidget(self.groupBox2, 0, 0)
        self.layout.addWidget(self.groupBox3, 1, 0)
        self.layout.addWidget(self.groupBox4, 2, 0)
        self.layout.addWidget(self.groupBox5, 3, 0)
        self.setCentralWidget(self.main_widget)
        self.resize(1475, 800)
        self.show()
        # Menu bar
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightgreen')
        mainMenu.setNativeMenuBar(False)# set native menu bar false
        fileMenu = mainMenu.addMenu('File')
        Analysis_menu = mainMenu.addMenu('Analyze Distribution')
        MLModelMenu = mainMenu.addMenu('Model Selection')
        predictMenu = mainMenu.addMenu('Predict Occupancy Status')
        # exit sequence
        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)
        # Analysis
        corrButton = QAction(QIcon('analysis.png'), 'Correlation Plot', self)
        corrButton.setStatusTip('Correlation plot of occupancy metrics')
        corrButton.triggered.connect(self.correlation_plot)
        Analysis_menu.addAction(corrButton)
        # Predict
        occ_prediction_button = QAction(QIcon(), 'Predict Occupancy', self)
        occ_prediction_button.setStatusTip('Predict occupancy status by user input')
        occ_prediction_button.triggered.connect(self.occ_pred)
        predictMenu.addAction(occ_prediction_button)
        # Decision Tree Model
        MLModel1Button = QAction(QIcon(), 'Decision Tree', self)
        MLModel1Button.setStatusTip('Decision Tree Classifier')
        MLModel1Button.triggered.connect(self.d_tree)
        # Random Forest Classifier
        MLModel2Button = QAction(QIcon(), 'Random Forest', self)
        MLModel2Button.setStatusTip('Random Forest Classifier')
        MLModel2Button.triggered.connect(self.r_forest)
        # Logistic Regression
        MLModel3Button = QAction(QIcon(), 'Logistic Regression', self)
        MLModel3Button.setStatusTip('Logistic Regression Classifier')
        MLModel3Button.triggered.connect(self.log_reg)
        # K-Nearest Neighbors
        MLModel4Button = QAction(QIcon(), 'K-Nearest Neighbors', self)
        MLModel4Button.setStatusTip('K-Nearest Neighbors Classifier')
        MLModel4Button.triggered.connect(self.knearest)
        # XGBoost
        MLModel5Button = QAction(QIcon(), 'XGBoost', self)
        MLModel5Button.setStatusTip('Extreme Gradient Boosting Classifier')
        MLModel5Button.triggered.connect(self.xgBoost)
        # AdaBoost
        MLModel6Button = QAction(QIcon(), 'AdaBoost', self)
        MLModel6Button.setStatusTip('AdaBoost Classifier')
        MLModel6Button.triggered.connect(self.a_boost)
        # apply menu selections
        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)
        MLModelMenu.addAction(MLModel3Button)
        MLModelMenu.addAction(MLModel4Button)
        MLModelMenu.addAction(MLModel5Button)
        MLModelMenu.addAction(MLModel6Button)

        self.dialogs = list()

    def correlation_plot(self):
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def d_tree(self):
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def r_forest(self):
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def log_reg(self):
        dialog = LogReg()
        self.dialogs.append(dialog)
        dialog.show()

    def knearest(self):
        dialog = kNearestNeighbors()
        self.dialogs.append(dialog)
        dialog.show()

    def xgBoost(self):
        dialog = XGboost()
        self.dialogs.append(dialog)
        dialog.show()

    def a_boost(self):
        dialog = adaBoost()
        self.dialogs.append(dialog)
        dialog.show()

    def occ_pred(self):
        dialog = occPredictor()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    '''initiates app'''
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.showMaximized()
    sys.exit(app.exec_())

def occupancy():
    global occ_data
    global feats
    global target
    global mods
    global reg_methods
    global states
    global X
    global y

    occ_data = pd.read_csv('occupancy-data.csv')
    feats = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    target = ['Occupancy']
    states = ['Vacant', 'Occupied']
    mods = ['DecisionTreeClassifier', 'RandomForestClassifier', 'LogisticRegression',
            'KNeighborsClassifier', 'XGBClassifier', 'AdaBoostClassifier']
    reg_methods = ['l1', 'l2']
    X = occ_data[feats]
    y = occ_data[target]

# driver
if __name__ == '__main__':
    occupancy()
    main()

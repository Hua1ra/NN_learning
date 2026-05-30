from PyQt6.QtWidgets import QFormLayout
from PyQt6.QtWidgets import QDateEdit, QDialog, QLineEdit, QPushButton
from PyQt6.QtCore import QDate

class AddFaceDialog(QDialog):
    def __init__(self,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add New Face')
        layout = QFormLayout(self)
        # Fields
        self.first_name_input = QLineEdit()
        self.last_name_input = QLineEdit()
        self.surname_input = QLineEdit()
        self.date_input = QDateEdit(QDate.currentDate())
        self.date_input.setCalendarPopup(True)
        layout.addRow('First name:',
                      self.first_name_input)
        layout.addRow('Last name:',
                      self.last_name_input)
        layout.addRow('Surname:',
                      self.surname_input)
        layout.addRow('Birthdate:',
                      self.date_input)
        # Button
        self.save_btn = QPushButton('Save')
        layout.addRow(self.save_btn)
        # Connect
        # noinspection PyUnresolvedReferences
        self.save_btn.clicked.connect(self.accept)

    def get_data(self):
        return {
            'first_name': self.first_name_input.text(),
            'last_name': self.last_name_input.text(),
            'surname': self.surname_input.text(),
            'birthdate': self.date_input.date().toString('yyyy-MM-dd')
        }
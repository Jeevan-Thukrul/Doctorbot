from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys
from PyQt5.QtWidgets import QWidget,QApplication

#setting a custom style

setStyleQte = """QTextEdit {
    font-family: "Courier"; 
    font-size: 12pt; 
    font-weight: 600; 
    text-align: right;
    background-color: Gainsboro;
}"""

setStyleUI = """QLineEdit {
    font-family: "Courier";
    font-weight: 600; 
    text-align: left;
    background-color: Gainsboro;
}"""

class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.v = None
        self.layout = QtWidgets.QVBoxLayout(self)
        self.font = QFont()
        self.font.setPointSize(12)
        self.labelUser = QtWidgets.QLabel("User Chats Here:")
        self.labelUser.setAlignment(Qt.AlignCenter)
        self.labelBot = QtWidgets.QLabel("Chat Logs are Here:")
        self.labelBot.setAlignment(Qt.AlignCenter)
        self.chatlog = QtWidgets.QTextEdit()
        self.userinput = QtWidgets.QLineEdit()
        self.userinput.returnPressed.connect(self.processUserMsg)

        self.GuiSetup()

    def GuiSetup(self):
        '''
        Styling and Layout.
        '''
        self.chatlog.setStyleSheet(setStyleQte)
        self.userinput.setStyleSheet(setStyleUI)
        self.userinput.setFont(self.font)
        self.layout.addWidget(self.labelBot)
        self.layout.addWidget(self.chatlog)
        self.layout.addWidget(self.labelUser)
        self.layout.addWidget(self.userinput)

    def processBot(self):
        self.chatlog.setAlignment(Qt.AlignRight)
        self.chatlog.append("Hello There User !")
        self.userinput.setFocus()
        self.chatlog.append("\n")
    def processUserMsg(self):
        umsg = self.userinput.text()
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.append(umsg)
        self.chatlog.moveCursor()
        # self.chatlog.setAlignment(Qt.AlignRight)
        self.userinput.setText("")
        self.processBot()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.setGeometry(10, 10, 540, 540)
    win.show()
    sys.exit(app.exec_())
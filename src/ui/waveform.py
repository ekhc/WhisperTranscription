from PySide6 import QtCore, QtGui, QtWidgets

class WaveformWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(WaveformWidget, self).__init__(parent)
        self.data = []  # list of audio amplitude values

        self._pen = QtGui.QPen(QtGui.QColor("#0000FF"))
        self._pen.setWidth(2)
        self._background_color = QtGui.QColor("#FFFFFF")
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, self._background_color)
        self.setPalette(palette)

    def update_data(self, data):
        """
        Update the waveform data and trigger a repaint.

        :param data: list of float values (typically between -1 and 1) representing audio amplitude.
        """
        self.data = data
        self.update()  # Schedule a repaint

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Fill the background.
        painter.fillRect(self.rect(), self._background_color)

        if not self.data:
            return

        rect = self.rect()
        width = rect.width()
        height = rect.height()
        mid_y = height / 2

        # Downsample the audio data to roughly one sample per pixel.
        num_points = len(self.data)
        # Ensure we do not divide by zero.
        step = max(1, num_points // width)

        # Build a list of points to draw.
        points = []
        for x in range(width):
            data_index = x * step
            if data_index < num_points:
                amplitude = self.data[data_index]
            else:
                amplitude = 0
            # Map amplitude (-1 to 1) to y-coordinate (0 to height)
            y = mid_y - (amplitude * mid_y)
            points.append(QtCore.QPointF(x, y))

        # Draw the waveform using a QPainterPath for smoothness.
        path = QtGui.QPainterPath()
        if points:
            path.moveTo(points[0])
            for pt in points[1:]:
                path.lineTo(pt)
            painter.setPen(self._pen)
            painter.drawPath(path)


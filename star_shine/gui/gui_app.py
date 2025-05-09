"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the graphical user interface.

Code written by: Luc IJspeert
"""
import os
import sys

from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QSplitter, QMenuBar
from PySide6.QtWidgets import QLabel, QTextEdit, QLineEdit, QFileDialog, QMessageBox, QPushButton
from PySide6.QtWidgets import QTableView, QHeaderView
from PySide6.QtGui import QAction, QFont, QScreen, QStandardItemModel, QStandardItem, QTextCursor, QIcon

from star_shine.core import utility as ut
from star_shine.api import Data, Result, Pipeline
from star_shine.gui import gui_log, gui_plot, gui_analysis, gui_config
from star_shine.config import helpers as hlp


# load configuration
config = hlp.get_config()


class MainWindow(QMainWindow):
    """The main window of the Star Shine application.

    Contains a graphical user interface for loading data, performing analysis,
    displaying results, and visualizing plots.
    """

    def __init__(self):
        super().__init__()

        # Get screen dimensions
        screen = QApplication.primaryScreen()
        screen_size = screen.availableSize()
        h_size = int(screen_size.width() * config.h_size_frac)  # some fraction of the screen width
        v_size = int(screen_size.height() * config.v_size_frac)  # some fraction of the screen height

        # Set some window things
        self.setWindowTitle("Star Shine")
        self.setGeometry(100, 50, h_size, v_size)  # x, y, width, height

        # App icon
        icon_path = os.path.join(hlp.get_images_path(), 'Star_Shine_dark_simple_small_transparent.png')
        self.setWindowIcon(QIcon(icon_path))

        # Set font size for the entire application
        font = QFont()
        font.setPointSize(11)
        QApplication.setFont(font)

        # Create the central widget and set the layout
        self._setup_central_widget()

        # Create the menu bar
        self._setup_menu_bar()

        # Add widgets to the layout
        self._add_widgets_to_layout()

        # custom gui-specific logger (will be reloaded and connected when data is loaded)
        self.logger = gui_log.get_custom_gui_logger('gui_logger', '')
        self.logger.log_signal.connect(self.append_text)

        # add the api classes for functionality
        self.pipeline = None
        self.pipeline_thread = None

        # some things that are needed
        self.data_dir = config.data_dir
        self.save_dir = config.save_dir
        self.save_subdir = ''

    def _setup_central_widget(self):
        """Set up the central widget and its layout."""
        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # create a horizontal layout
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # create a splitter
        self.splitter = QSplitter()
        self.central_widget.layout().addWidget(self.splitter)

        return None

    def _add_widgets_to_layout(self):
        """Add widgets to the main window layout."""
        # Left column: input and text output field
        left_column = self._create_left_column()
        self.splitter.addWidget(left_column)

        # Middle column: frequencies
        middle_column = self._create_middle_column()
        self.splitter.addWidget(middle_column)

        # right column: plot area
        right_column = self._create_right_column()
        self.splitter.addWidget(right_column)

        # Set initial sizes for each column
        h_size = self.width()
        self.splitter.setSizes([h_size*3//9, h_size*2//9, h_size*4//9])

        return None

    def _setup_menu_bar(self):
        """Set up the menu bar with file and info menus."""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Add "File" menu and set it up
        file_menu = menu_bar.addMenu("File")
        self._setup_file_menu(file_menu)

        # Add "View" menu and set it up
        view_menu = menu_bar.addMenu("View")
        self._setup_view_menu(view_menu)

        # Add "Info" menu
        info_menu = menu_bar.addMenu("Info")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        info_menu.addAction(about_action)

        return None

    def _setup_file_menu(self, file_menu):
        """Set up the file menu."""
        # Add "Load Data" button to "File" menu
        load_data_action = QAction("Load Data", self)
        load_data_action.triggered.connect(self.load_data_external)
        file_menu.addAction(load_data_action)

        # Add "Set Save Location" button to "File" menu
        set_save_location_action = QAction("Set Save Location", self)
        set_save_location_action.triggered.connect(self.set_save_location)
        file_menu.addAction(set_save_location_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Load Data Object" button to "File" menu
        load_data_object_action = QAction("Load Data Object", self)
        load_data_object_action.triggered.connect(self.load_data)
        file_menu.addAction(load_data_object_action)

        # Add "Save Data Object" button to "File" menu
        save_data_object_action = QAction("Save Data Object", self)
        save_data_object_action.triggered.connect(self.save_data)
        file_menu.addAction(save_data_object_action)

        # Add "Load Result Object" button to "File" menu
        load_result_object_action = QAction("Load Result Object", self)
        load_result_object_action.triggered.connect(self.load_result)
        file_menu.addAction(load_result_object_action)

        # Add "Save Result Object" button to "File" menu
        save_result_object_action = QAction("Save Result Object", self)
        save_result_object_action.triggered.connect(self.save_result)
        file_menu.addAction(save_result_object_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Settings" action to open the settings dialog
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)

        # Add a horizontal separator
        file_menu.addSeparator()

        # Add "Exit" button to "File" menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        return None

    def _setup_view_menu(self, view_menu):
        """Set up the view menu."""
        # Add a horizontal separator
        view_menu.addSeparator()

        # Add "Load Data" button to "File" menu
        self.show_residual_action = QAction("Show residual", self)
        self.show_residual_action.setCheckable(True)
        self.show_residual_action.setChecked(False)
        self.show_residual_action.triggered.connect(self.update_plots)
        view_menu.addAction(self.show_residual_action)

        return None

    def _create_left_column(self):
        """Create and return the left column widget.

        Returns
        -------
        QWidget
            The left column widget containing input fields, buttons, and text output.
        """
        # create a vertical layout for in the left column of the main layout
        l_col_widget = QWidget()
        l_col_layout = QVBoxLayout(l_col_widget)

        # Input field for file path
        input_sub_layout = QHBoxLayout()
        self.file_path_label = QLabel("File Path:")
        self.file_path_edit = QLineEdit()
        input_sub_layout.addWidget(self.file_path_label)
        input_sub_layout.addWidget(self.file_path_edit)
        l_col_layout.addLayout(input_sub_layout)

        # Button for starting analysis
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_analysis)  # Connect the action to a custom method
        l_col_layout.addWidget(self.analyze_button)

        # Log area
        self.text_field = QTextEdit()
        self.text_field.setReadOnly(True)  # Make the text edit read-only
        self.text_field.setPlainText("Log\n")
        l_col_layout.addWidget(self.text_field)

        return l_col_widget

    def _create_middle_column(self):
        """Create and return the middle column widget.

        Returns
        -------
        QWidget
            The middle column widget containing a table view.
        """
        # create a vertical layout for in the middle column of the main layout
        m_col_widget = QWidget()
        m_col_layout = QVBoxLayout(m_col_widget)

        # add the data model formula above the table
        equation_str = "Model: flux = \u2211\u1D62 (a\u1D62 sin(2\u03C0f\u1D62t + \u03C6\u1D62)) + bt + c"
        formula_label = QLabel(equation_str)
        m_col_layout.addWidget(formula_label)

        # Create the table view and model
        self.table_view = QTableView()
        self.table_model = QStandardItemModel(0, 3)  # Start with 0 rows and 3 columns
        self.table_model.setHorizontalHeaderLabels(["Frequency", "Amplitude", "Phase"])
        self.table_view.setModel(self.table_model)

        # Set the horizontal header's stretch mode for each column
        h_header = self.table_view.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # Stretch all columns proportionally

        # Set the vertical header's stretch mode for each row
        v_header = self.table_view.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Add the button and table view to the middle column layout
        m_col_layout.addWidget(self.table_view)

        return m_col_widget

    def _create_right_column(self):
        """Create and return the right column widget.

        Returns
        -------
        QWidget
            The right column widget containing plot areas.
        """
        # create a vertical layout for in the right column of the main layout
        r_col_widget = QWidget()
        r_col_layout = QVBoxLayout(r_col_widget)

        # upper plot area for the data
        self.upper_plot_area = gui_plot.PlotWidget(title='Data', xlabel='time', ylabel='flux')
        r_col_layout.addWidget(self.upper_plot_area)

        # lower plot area for the periodogram
        self.lower_plot_area = gui_plot.PlotWidget(title='Periodogram', xlabel='frequency', ylabel='amplitude')
        r_col_layout.addWidget(self.lower_plot_area)

        # connect the click event
        self.lower_plot_area.click_signal.connect(self.click_periodogram)

        return r_col_widget

    def append_text(self, text):
        """Append a line of text at the end of the plain text output box.

        Parameters
        ----------
        text: str
            The text to append.
        """
        cursor = self.text_field.textCursor()
        cursor.movePosition(QTextCursor.End)  # Move cursor to the end of the text
        cursor.insertText(text + '\n')  # insert the text
        self.text_field.setTextCursor(cursor)
        self.text_field.ensureCursorVisible()

        return None

    def update_table(self, display_err=True):
        """Fill the table with the given data."""
        # get the result parameters
        col1 = self.pipeline.result.f_n
        col2 = self.pipeline.result.a_n
        col3 = self.pipeline.result.ph_n
        col1_err = self.pipeline.result.f_n_err
        col2_err = self.pipeline.result.a_n_err
        col3_err = self.pipeline.result.ph_n_err

        # display sinusoid parameters in the table
        self.table_model.setRowCount(len(col1))
        for row, row_items in enumerate(zip(col1, col2, col3, col1_err, col2_err, col3_err)):
            # convert to strings
            c1 = ut.float_to_str_scientific(row_items[0], row_items[3], error=display_err, brackets=False)
            c2 = ut.float_to_str_scientific(row_items[1], row_items[4], error=display_err, brackets=False)
            c3 = ut.float_to_str_scientific(row_items[2], row_items[5], error=display_err, brackets=False)

            # convert to table items
            c1_item = QStandardItem(c1)
            c2_item = QStandardItem(c2)
            c3_item = QStandardItem(c3)

            # insert into table
            self.table_model.setItem(row, 0, c1_item)
            self.table_model.setItem(row, 1, c2_item)
            self.table_model.setItem(row, 2, c3_item)

        return None

    def update_plots(self, new_plot=False):
        """Update the plotting area with the current data."""
        # collect plot data in a dict
        upper_plot_data = {}
        lower_plot_data = {}

        # upper plot area - time series
        upper_plot_data['scatter_xs'] = [self.pipeline.data.time]
        upper_plot_data['scatter_ys'] = [self.pipeline.data.flux]
        # lower plot area - periodogram
        freqs, ampls = self.pipeline.data.periodogram()
        lower_plot_data['plot_xs'] = [freqs]
        lower_plot_data['plot_ys'] = [ampls]

        # include result attributes if present
        if self.pipeline.result.target_id != '' and not self.show_residual_action.isChecked():
            # upper plot area - time series
            upper_plot_data['plot_xs'] = [self.pipeline.data.time]
            upper_plot_data['plot_ys'] = [self.pipeline.model()]
            upper_plot_data['plot_colors'] = ['grey']
            # lower plot area - periodogram
            freqs, ampls = self.pipeline.periodogram(subtract_model=True)
            lower_plot_data['plot_xs'].append(freqs)
            lower_plot_data['plot_ys'].append(ampls)
            lower_plot_data['vlines_xs'] = [self.pipeline.result.f_n]
            lower_plot_data['vlines_ys'] = [self.pipeline.result.a_n]
            lower_plot_data['vlines_colors'] = ['grey']

        # only show residual if toggle checked
        if self.pipeline.result.target_id != '' and self.show_residual_action.isChecked():
            # upper plot area - time series
            residual = self.pipeline.data.flux - self.pipeline.model()
            upper_plot_data['scatter_xs'] = [self.pipeline.data.time]
            upper_plot_data['scatter_ys'] = [residual]
            # lower plot area - periodogram
            freqs, ampls = self.pipeline.periodogram(subtract_model=True)
            lower_plot_data['plot_xs'] = [freqs]
            lower_plot_data['plot_ys'] = [ampls]

        # set the plot data
        self.upper_plot_area.set_plot_data(**upper_plot_data)
        self.lower_plot_area.set_plot_data(**lower_plot_data)

        # start with a fresh plot
        if new_plot:
            self.upper_plot_area.new_plot()
            self.lower_plot_area.new_plot()
        else:
            self.upper_plot_area.update_plot()
            self.lower_plot_area.update_plot()

        return None

    def new_dataset(self, data):
        """Set up pipeline and logger for the new data that was loaded"""
        # for saving, make a folder if not there yet
        self.save_subdir = f"{data.target_id}_analysis"

        full_dir = os.path.join(self.save_dir, self.save_subdir)
        if not os.path.isdir(full_dir):
            os.mkdir(full_dir)  # create the subdir

        # custom gui-specific logger
        self.logger = gui_log.get_custom_gui_logger(data.target_id, full_dir)
        self.logger.log_signal.connect(self.append_text)

        # Make ready the pipeline class
        self.pipeline = Pipeline(data=data, save_dir=self.save_dir, logger=self.logger)

        # set up a pipeline thread
        self.pipeline_thread = gui_analysis.PipelineThread(self.pipeline)
        self.pipeline_thread.result_signal.connect(self.handle_result_signal)

        # clear and update the plots
        self.update_plots(new_plot=True)

        return None

    def set_save_location(self):
        """Open a dialog to select the save location."""
        # Open a directory selection dialog
        new_dir = QFileDialog.getExistingDirectory(self, caption="Select Save Location", dir=self.save_dir)

        if new_dir:
            self.save_dir = new_dir
            self.append_text(f"Save location set to: {self.save_dir}")

        return None

    def load_data_external(self):
        """Read data from a file or multiple files using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_paths, _ = QFileDialog.getOpenFileNames(self, caption="Read Data", dir=self.save_dir,
                                                     filter="All Files (*)")

        # do nothing in case no file(s) selected
        if not file_paths:
            return None

        # load data into instance
        data = Data.load_data(file_list=file_paths, data_dir='', target_id='', data_id='', logger=self.logger)

        # set up some things
        self.new_dataset(data)

        return None

    def load_data(self):
        """Load data from a file using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_path, _ = QFileDialog.getOpenFileName(self, caption="Load Data", dir=self.save_dir,
                                                    filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        # load data into instance
        data = Data.load(file_name=file_path, data_dir='', logger=self.logger)

        # set up some things
        self.new_dataset(data)

        return None

    def save_data(self):
        """Save data to a file using a dialog window."""
        suggested_path = os.path.join(self.save_dir, self.pipeline.data.target_id + '_data.hdf5')
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save Data", dir=suggested_path,
                                                   filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        self.pipeline.data.save(file_path)

        return None

    def load_result(self):
        """Load result from a file using a dialog window."""
        # get the path(s) from a standard file selection screen
        file_path, _ = QFileDialog.getOpenFileName(self, caption="Load Result", dir=self.save_dir,
                                                    filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        # load result into instance
        self.pipeline.result = Result.load(file_name=file_path, logger=self.logger)

        # clear and update the plots
        self.update_plots(new_plot=False)

        return None

    def save_result(self):
        """Save result to a file using a dialog window."""
        suggested_path = os.path.join(self.save_dir, self.pipeline.data.target_id + '_result.hdf5')
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save Data", dir=suggested_path,
                                                   filter="HDF5 Files (*.hdf5);;All Files (*)")

        # do nothing in case no file selected
        if not file_path:
            return None

        self.pipeline.result.save(file_path)

        return None

    def handle_result_signal(self):
        """Handle the emitted result signal: update the GUI with the results."""
        # display sinusoid parameters in the table
        self.update_table(display_err=True)

        # Update the plot area with the results
        self.update_plots(new_plot=False)

        return None

    def perform_analysis(self):
        """Perform analysis on the loaded data and display results."""
        # check whether data is loaded
        if len(self.pipeline.data.file_list) == 0:
            self.logger.error("Input Error: please provide data files.")
            return None

        # set up and start a new thread for the analysis
        self.pipeline_thread.start()

        return None

    def click_periodogram(self, x, y, button):
        """Handle click events on the periodogram plot."""
        # Guard against empty data
        if self.pipeline is None:
            self.append_text(f"Plot clicked at coordinates: ({x}, {y})")
            return None

        # Left click
        if button == 1:
            self.pipeline_thread.extract_approx(x)

        # Right click
        if button == 3:
            self.pipeline_thread.remove_approx(x)

        return None

    def show_settings_dialog(self):
        """Show a 'settings' dialog with configuration for the application."""
        dialog = gui_config.SettingsDialog(config=config, parent=self)

        if dialog.exec():
            # Update any dependent components with new configuration values
            screen = QApplication.primaryScreen()
            screen_size = screen.availableSize()
            h_size = int(screen_size.width() * config.h_size_frac)  # some fraction of the screen width
            v_size = int(screen_size.height() * config.v_size_frac)  # some fraction of the screen height
            self.setGeometry(100, 50, h_size, v_size)

        return None

    def show_about_dialog(self):
        """Show an 'about' dialog with information about the application."""
        version = hlp.get_version()
        message = (f"STAR SHINE version {version}\n"
                   "Satellite Time-series Analysis Routine "
                   "using Sinusoids and Harmonics through Iterative Non-linear Extraction\n"
                   "Repository: https://github.com/LucIJspeert/star_shine\n"
                   "Code written by: Luc IJspeert")
        QMessageBox.about(self, "About", message)

        return None


def launch_gui():
    """Launch the Star Shine GUI."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()

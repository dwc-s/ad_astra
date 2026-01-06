"""
Ad Astra - Astronomical Observation Planner

This application allows users to define an observer location and time window,
then queries the SIMBAD astronomical database to find celestial objects
(stars, galaxies, nebulae) that are visible above a specified horizon
and brighter than a specified magnitude limit.
"""
import sys
import csv
import json
import os
import time
import urllib.parse
from datetime import datetime
import pytz
import numpy as np
import astropy.units as u
import astroplan
from astroplan import FixedTarget, AltitudeConstraint, AtNightConstraint
from astropy.time import Time
from timezonefinder import TimezoneFinder
from astroquery.simbad import Simbad
from astroquery.hips2fits import hips2fits
from astropy.visualization import simple_norm, AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from astropy.coordinates import SkyCoord
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QTextEdit, QFormLayout, QMessageBox, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QComboBox, QLabel, QDialog, 
                             QListWidget, QAbstractItemView, QHeaderView, QCheckBox, QTabWidget,
                             QTextBrowser, QFileDialog, QScrollArea, QSlider, QFrame, QGridLayout)
import warnings
from astropy.utils.exceptions import AstropyWarning

# --- Style Configuration ---
# This color is used for text within Matplotlib plots, which is not controlled by the CSS file.
PLOT_TEXT_COLOR = 'white'
# This color is for the Matplotlib figure background. It should match the dialog background in the CSS.
PLOT_BG_COLOR = '#2b2b35'


# Suppress specific Astropy warnings
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
warnings.filterwarnings('ignore', message='.*ERFA function "dtf2d" yielded.*')

# Mapping of SIMBAD O-types to human-readable descriptions
# Source: https://simbad.cds.unistra.fr/guide/otypes.htx
OTYPE_MAP = {
     '?': 'Object of Unknown Nature',
     '..1': '{pr*} Pre-Main Sequence Star ',
     '..10': 'Barium Star',
     '..11': 'Dwarf Carbon Star',
     '..12': 'Carbon-Enhanced Metal Poor Star',
     '..13': '{Al*} Eclipsing Binary of Algol type',
     '..14': '{bL*}Eclipsing Binary of beta Lyr type',
     '..15': '{WU*} Eclipsing Binary of W UMa type',
     '..16': '{NL*} Nova-like Binary',
     '..17': '{DN*} Dwarf Nova',
     '..18': '{DQ*} CV of DQ Her type  Intermediate polar.',
     '..19': '{AM*} CV of AM CVn type',
     '..2': 'LBV=Luminous Blue Variable',
     '..20': 'Irregular Variable with rapid variations',
     '..21': '{Fl*} Flare Star',
     '..22': 'Star showing Eclipses by its Planet',
     '..23': '{*iC} Star towards a Cluster',
     '..24': '{*iA} Star towards an Association',
     '..25': '{*iN} Star towards a Nebula',
     '..26': '{*i*} Star in double system',
     '..27': '{BNe} Bright Nebula',
     '..28': '{HzG} Galaxy with high redshift',
     '..29': '{ERO} ERO/VRO, Extremely/Very Red Object',
     '..3': '{FU*} FU Ori Variable',
     '..30': 'ULIRG, Ultra Luminous Infrared Galaxy',
     '..31': '{LyA, DLA, mAL, LLS, BAL} Absorption Line System',
     '..32': '{red} Very Red Source',
     '..4': 'Red Clump Star',
     '..5': '{sr*} Semi-Regular Variable',
     '..6': 'O-rich AGB Star',
     '..7': '{ZZ*} Pulsating White Dwarf',
     '..8': 'ELMWD=Extremely Low Mass White Dwarf',
     '..9': 'CH Star',
     '*': 'Star',
     '**': 'Double or Multiple Star',
     'a2*': 'alpha2 CVn Variable',
     'AB*': 'Asymptotic Giant Branch Star',
     'Ae*': 'Herbig Ae/Be Star',
     'AGN': 'Active Galaxy Nucleus',
     'As*': 'Association of Stars',
     'bC*': 'beta Cep Variable',
     'bCG': 'Blue Compact Galaxy',
     'BD*': 'Brown Dwarf',
     'Be*': 'Be Star',
     'BH': 'Black Hole',
     'BiC': 'Brightest Galaxy in a Cluster (BCG)',
     'Bla': 'Blazar',
     'BLL': 'BL Lac',
     'blu': 'Blue Object',
     'BS*': 'Blue Straggler',
     'bub': 'Bubble',
     'BY*': 'BY Dra Variable',
     'C*': 'Carbon Star',
     'cC*': 'Classical Cepheid Variable',
     'Ce*': 'Cepheid Variable',
     'CGb': 'Cometary Globule / Pillar',
     'CGG': 'Compact Group of Galaxies',
     'Cl*': 'Cluster of Stars',
     'Cld': 'Cloud',
     'ClG': 'Cluster of Galaxies',
     'cm': 'Centimetric Radio Source',
     'cor': 'Dense Core',
     'CV*': 'Cataclysmic Binary',
     'DNe': 'Dark Cloud (nebula)',
     'dS*': 'delta Sct Variable',
     'EB*': 'Eclipsing Binary',
     'El*': 'Ellipsoidal Variable',
     'Em*': 'Emission-line Star',
     'EmG': 'Emission-line galaxy',
     'EmO': 'Emission Object',
     'Er*': 'Eruptive Variable',
     'err': 'Not an Object (Error, Artefact, ...)',
     'ev': 'Transient Event',
     'Ev*': 'Evolved Star',
     'FIR': 'Far-IR source (λ >= 30 µm)',
     'flt': 'Interstellar Filament',
     'G': 'Galaxy',
     'gam': 'Gamma-ray Source',
     'gB': 'Gamma-ray Burst',
     'gD*': 'gamma Dor Variable',
     'GiC': 'Galaxy towards a Cluster of Galaxies',
     'GiG': 'Galaxy towards a Group of Galaxies',
     'GiP': 'Galaxy in Pair of Galaxies',
     'glb': 'Globule (low-mass dark cloud)',
     'GlC': 'Globular Cluster',
     'gLe': 'Gravitational Lens',
     'gLS': 'Gravitational Lens System (lens+images)',
     'GNe': 'Nebula',
     'GrG': 'Group of Galaxies',
     'grv': 'Gravitational Source',
     'GWE': 'Gravitational Wave Event',
     'H2G': 'HII Galaxy',
     'HB*': 'Horizontal Branch Star',
     'HH': 'Herbig-Haro Object',
     'HI': 'HI (21cm) Source',
     'HII': 'HII Region',
     'HS*': 'Hot Subdwarf',
     'HV*': 'High Velocity Star',
     'HVC': 'High-velocity Cloud',
     'HXB': 'High Mass X-ray Binary',
     'IG': 'Interacting Galaxies',
     'IR': 'Infra-Red Source',
     'Ir*': 'Irregular Variable',
     'ISM': 'Interstellar Medium Object',
     'LeG': 'Gravitationally Lensed Image of a Galaxy',
     'LeI': 'Gravitationally Lensed Image',
     'LeQ': 'Gravitationally Lensed Image of a Quasar',
     'Lev': '(Micro)Lensing Event',
     'LIN': 'LINER-type Active Galaxy Nucleus',
     'LM*': 'Low-mass Star',
     'LP*': 'Long-Period Variable',
     'LSB': 'Low Surface Brightness Galaxy',
     'LXB': 'Low Mass X-ray Binary',
     'Ma*': 'Massive Star',
     'Mas': 'Maser',
     'MGr': 'Moving Group',
     'Mi*': 'Mira Variable',
     'MIR': 'Mid-IR Source (3 to 30 µm)',
     'mm': 'Millimetric Radio Source',
     'MoC': 'Molecular Cloud',
     'mR': 'Metric Radio Source',
     'MS*': 'Main Sequence Star',
     'mul': 'Composite Object, Blend',
     'N*': 'Neutron Star',
     'NIR': 'Near-IR Source (λ < 3 µm)',
     'No*': 'Classical Nova',
     'OH*': 'OH/IR Star',
     'OpC': 'Open Cluster',
     'Opt': 'Optical Source',
     'Or*': 'Orion Variable',
     'out': 'Outflow',
     'pA*': 'Post-AGB Star',
     'PaG': 'Pair of Galaxies',
     'PCG': 'Proto Cluster of Galaxies',
     'Pe*': 'Chemically Peculiar Star',
     'Pl': 'Extra-solar Planet',
     'PM*': 'High Proper Motion Star',
     'PN': 'Planetary Nebula',
     'PoC': 'Part of Cloud',
     'PoG': 'Part of a Galaxy',
     'Psr': 'Pulsar',
     'Pu*': 'Pulsating Variable',
     'QSO': 'Quasar',
     'Rad': 'Radio Source',
     'rB': 'Radio Burst',
     'RC*': 'R CrB Variable',
     'reg': 'Region defined in the Sky',
     'rG': 'Radio Galaxy',
     'RG*': 'Red Giant Branch star',
     'RNe': 'Reflection Nebula',
     'Ro*': 'Rotating Variable',
     'RR*': 'RR Lyrae Variable',
     'RS*': 'RS CVn Variable',
     'RV*': 'RV Tauri Variable',
     'S*': 'S Star',
     's*b': 'Blue Supergiant',
     's*r': 'Red Supergiant',
     's*y': 'Yellow Supergiant',
     'SB*': 'Spectroscopic Binary',
     'SBG': 'Starburst Galaxy',
     'SCG': 'Supercluster of Galaxies',
     'SFR': 'Star Forming Region',
     'sg*': 'Evolved Supergiant',
     'sh': 'Interstellar Shell',
     'smm': 'Sub-Millimetric Source',
     'SN*': 'SuperNova',
     'SNR': 'SuperNova Remnant',
     'St*': 'Stellar Stream',
     'SX*': 'SX Phe Variable',
     'Sy*': 'Symbiotic Star',
     'Sy1': 'Seyfert 1 Galaxy',
     'Sy2': 'Seyfert 2 Galaxy',
     'SyG': 'Seyfert Galaxy',
     'TT*': 'T Tauri Star',
     'ULX': 'Ultra-luminous X-ray Source',
     'UV': 'UV-emission Source',
     'V*': 'Variable Star',
     'var': 'Variable source',
     'vid': 'Underdense Region of the Universe',
     'WD*': 'White Dwarf',
     'WR*': 'Wolf-Rayet',
     'WV*': 'Type II Cepheid Variable',
     'X': 'X-ray Source',
     'XB*': 'X-ray Binary',
     'Y*O': 'Young Stellar Object',
 }

class ClickableLabel(QLabel):
    """A QLabel that emits a clicked signal with its stored data."""
    clicked = pyqtSignal(object, str)

    def __init__(self, data, title, parent=None):
        super().__init__(parent)
        self.data = data
        self.title = title
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self.data, self.title)

class ImageDialog(QDialog):
    """A resizable dialog to show a single large image."""
    def __init__(self, image_data, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 800)
        
        layout = QVBoxLayout(self)
        self.figure = Figure(facecolor=PLOT_BG_COLOR)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        self.plot_image(image_data, title)

    def plot_image(self, data, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if data.ndim == 3:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            
            # Apply Asinh stretch to each channel to improve contrast
            stretched_data = np.zeros_like(data, dtype=float)
            for i in range(3):
                channel = data[:, :, i]
                channel = np.nan_to_num(channel)
                stretch = AsinhStretch(a=0.1)
                norm = ImageNormalize(stretch=stretch, vmin=np.min(channel), vmax=np.max(channel))
                stretched_data[:, :, i] = norm(channel)

            ax.imshow(stretched_data, origin='lower')
        else:
            data = np.nan_to_num(data)
            norm = ImageNormalize(stretch=AsinhStretch(a=0.1), vmin=np.min(data), vmax=np.max(data))
            ax.imshow(data, origin='lower', cmap='gray', norm=norm)
            
        ax.set_title(title, color=PLOT_TEXT_COLOR)
        ax.axis('off')
        self.figure.tight_layout()
        self.canvas.draw()

class MultiImageDialog(QDialog):
    """A dialog to display multiple images in a grid."""
    def __init__(self, title, images, layout_shape=(1, 1), parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 900)
        self.setStyleSheet(f"background-color: {PLOT_BG_COLOR}; color: {PLOT_TEXT_COLOR};")
        
        layout = QVBoxLayout(self)
        self.figure = Figure(facecolor=PLOT_BG_COLOR)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        self.plot_images(images, layout_shape)
        
    def plot_images(self, images, layout_shape):
        # images: list of tuples (data, title, cmap)
        rows, cols = layout_shape
        axes = self.figure.subplots(rows, cols)
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i, ax in enumerate(axes):
            if i < len(images):
                data, title, cmap = images[i]
                data = np.nan_to_num(data)
                
                if cmap == 'gray':
                     norm = ImageNormalize(stretch=AsinhStretch(a=0.1), vmin=np.nanmin(data), vmax=np.nanmax(data))
                else:
                     norm = simple_norm(data, 'sqrt', percent=99)
                
                im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
                ax.set_title(title, color=PLOT_TEXT_COLOR)
                ax.axis('off')
                
                # Add center marker
                h, w = data.shape
                ax.plot(w / 2 - 0.5, h / 2 - 0.5, 'ko', markersize=4, markerfacecolor='none', markeredgecolor='cyan')
            else:
                ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()

class NumericTableWidgetItem(QTableWidgetItem):
    """
    Custom TableItem to ensure numerical sorting for columns like Magnitude and Coordinates.
    Standard QTableWidgetItem sorts alphabetically (e.g., "10" comes before "2").
    """
    def __init__(self, text, sort_value=None):
        super().__init__(text)
        self.sort_value = sort_value
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def __lt__(self, other):
        if (getattr(self, 'sort_value', None) is not None and 
            getattr(other, 'sort_value', None) is not None):
            return bool(self.sort_value < other.sort_value)

        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)

class LocationManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Locations")
        self.resize(500, 400)
        self.parent_window = parent
        
        layout = QVBoxLayout(self)
        
        # List of locations
        layout.addWidget(QLabel("Saved Locations:"))
        self.loc_list = QListWidget()
        self.loc_list.addItems(sorted(self.parent_window.locations.keys()))
        self.loc_list.itemClicked.connect(self.load_location)
        layout.addWidget(self.loc_list)
        
        # Form
        form_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.lat_edit = QLineEdit()
        self.lon_edit = QLineEdit()
        self.elev_edit = QLineEdit()
        
        form_layout.addRow("Name:", self.name_edit)
        form_layout.addRow("Latitude (deg):", self.lat_edit)
        form_layout.addRow("Longitude (deg):", self.lon_edit)
        form_layout.addRow("Elevation (m):", self.elev_edit)
        layout.addLayout(form_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.new_btn = QPushButton("New / Clear")
        self.new_btn.clicked.connect(self.clear_form)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_location)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self.delete_location)
        
        btn_layout.addWidget(self.new_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.del_btn)
        layout.addLayout(btn_layout)

    def load_location(self, item):
        name = item.text()
        if name in self.parent_window.locations:
            data = self.parent_window.locations[name]
            self.name_edit.setText(name)
            self.lat_edit.setText(str(data['latitude']))
            self.lon_edit.setText(str(data['longitude']))
            self.elev_edit.setText(str(data['elevation']))

    def clear_form(self):
        self.loc_list.clearSelection()
        self.name_edit.clear()
        self.lat_edit.clear()
        self.lon_edit.clear()
        self.elev_edit.clear()

    def save_location(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Name cannot be empty.")
            return
        try:
            lat = float(self.lat_edit.text())
            lon = float(self.lon_edit.text())
            elev = float(self.elev_edit.text())
            
            self.parent_window.locations[name] = {
                'latitude': lat, 'longitude': lon, 'elevation': elev
            }
            self.parent_window.save_data()
            self.refresh_list()
            self.parent_window.refresh_locations()
            QMessageBox.information(self, "Success", f"Location '{name}' saved.")
        except ValueError:
            QMessageBox.warning(self, "Error", "Latitude, Longitude, and Elevation must be valid numbers.")

    def delete_location(self):
        name = self.name_edit.text().strip()
        if not name: return
        
        if name in self.parent_window.locations:
            reply = QMessageBox.question(self, 'Confirm Delete', f"Delete '{name}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                del self.parent_window.locations[name]
                self.parent_window.save_data()
                self.clear_form()
                self.refresh_list()
                self.parent_window.refresh_locations()
        else:
            QMessageBox.warning(self, "Error", "Location not found.")

    def refresh_list(self):
        self.loc_list.clear()
        self.loc_list.addItems(sorted(self.parent_window.locations.keys()))

class AliasesDialog(QDialog):
    def __init__(self, object_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Aliases: {object_name}")
        self.resize(400, 500)
        
        layout = QVBoxLayout(self)
        
        self.info_label = QLabel(f"Fetching aliases for {object_name}...")
        layout.addWidget(self.info_label)
        
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        layout.addWidget(self.text_browser)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)
        
        self.fetch_aliases(object_name)

    def fetch_aliases(self, name):
        QApplication.processEvents()
        try:
            s = Simbad()
            s.ROW_LIMIT = 0
            table = s.query_objectids(name)
            
            if table:
                html_content = ""
                for row in table:
                    val = row[0]
                    if isinstance(val, bytes): val = val.decode('utf-8')
                    alias = ' '.join(str(val).split())
                    encoded_alias = urllib.parse.quote_plus(alias)
                    url = f"https://en.wikipedia.org/w/index.php?search={encoded_alias}"
                    html_content += f'<a href="{url}" style="color: #add8e6; font-family: Courier;">{alias}</a><br>'
                self.text_browser.setHtml(html_content)
                self.info_label.setText(f"Found {len(table)} aliases for {name}:")
            else:
                self.info_label.setText(f"No aliases found for {name}.")
        except Exception as e:
            self.info_label.setText(f"Error: {str(e)}")

class SaveListDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save to List")
        self.resize(300, 100)
        self.list_name = None
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter name for new list or select existing:"))
        
        self.combo = QComboBox()
        self.combo.setEditable(True)
        
        # Load existing names
        if os.path.exists('adastra_lists.json'):
            try:
                with open('adastra_lists.json', 'r') as f:
                    data = json.load(f)
                    self.combo.addItems(sorted(data.keys()))
            except:
                pass
        
        layout.addWidget(self.combo)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def save(self):
        text = self.combo.currentText().strip()
        if text:
            self.list_name = text
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "List name cannot be empty")

class ListManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Lists")
        self.resize(900, 500)
        self.lists_data = {}
        self.current_list_name = None
        
        self.load_data()
        
        layout = QHBoxLayout(self)
        
        # Left side: Lists
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Saved Lists:"))
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_list_selected)
        left_layout.addWidget(self.list_widget)
        
        self.del_list_btn = QPushButton("Delete List")
        self.del_list_btn.clicked.connect(self.delete_list)
        left_layout.addWidget(self.del_list_btn)
        
        layout.addWidget(left_widget, 1)
        
        # Right side: Contents
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("List Contents:"))
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "RA", "Dec", "Observable", "Mag"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        right_layout.addWidget(self.table)
        
        self.del_item_btn = QPushButton("Remove Item from List")
        self.del_item_btn.clicked.connect(self.delete_item)
        right_layout.addWidget(self.del_item_btn)
        
        layout.addWidget(right_widget, 2)
        
        self.refresh_lists()

    def load_data(self):
        if os.path.exists('adastra_lists.json'):
            try:
                with open('adastra_lists.json', 'r') as f:
                    self.lists_data = json.load(f)
            except:
                self.lists_data = {}

    def save_data(self):
        with open('adastra_lists.json', 'w') as f:
            json.dump(self.lists_data, f, indent=4)

    def refresh_lists(self):
        self.list_widget.clear()
        self.list_widget.addItems(sorted(self.lists_data.keys()))
        self.table.setRowCount(0)
        self.current_list_name = None

    def on_list_selected(self, item):
        self.current_list_name = item.text()
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(0)
        if self.current_list_name and self.current_list_name in self.lists_data:
            items = self.lists_data[self.current_list_name]
            self.table.setRowCount(len(items))
            for i, item_data in enumerate(items):
                self.table.setItem(i, 0, QTableWidgetItem(item_data.get("Name", "")))
                self.table.setItem(i, 1, QTableWidgetItem(item_data.get("Type", "")))
                self.table.setItem(i, 2, QTableWidgetItem(item_data.get("RA", "")))
                self.table.setItem(i, 3, QTableWidgetItem(item_data.get("Dec", "")))
                self.table.setItem(i, 4, QTableWidgetItem(item_data.get("Observable", "")))
                self.table.setItem(i, 5, QTableWidgetItem(item_data.get("Mag", "")))

    def delete_list(self):
        if not self.current_list_name: return
        reply = QMessageBox.question(self, "Confirm", f"Delete list '{self.current_list_name}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            del self.lists_data[self.current_list_name]
            self.save_data()
            self.refresh_lists()

    def delete_item(self):
        if not self.current_list_name: return
        row = self.table.currentRow()
        if row < 0: return
        
        items = self.lists_data[self.current_list_name]
        del items[row]
        self.lists_data[self.current_list_name] = items
        self.save_data()
        self.refresh_table()

class AdAstraWindow(QMainWindow):
    def __init__(self):
        """
        Initialize the Main Window, Layouts, and Widgets.
        """
        super().__init__()
        
        # In-memory store for location data
        self.locations = {}
        
        self.current_dd_coord = None
        self.current_dd_name = None
        self.overlay_maps = {} # Store fetched gas/dust maps for overlay
        self.dynamic_photo_widgets = [] # Keep track of photo widgets

        self.setWindowTitle("Ad Astra")
        self.resize(1200, 800) # Increased default size
        
        # Set window icon (requires 'icon.png' in the same directory)
        self.setWindowIcon(QIcon('icon.png'))

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Observable Objects ---
        self.observable_tab = QWidget()
        main_layout = QHBoxLayout(self.observable_tab)

        # Left Container
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Form Layout for inputs
        form_layout = QFormLayout()

        # Location selection
        self.location_combo = QComboBox()
        self.location_combo.currentIndexChanged.connect(self.on_location_change)
        
        self.new_loc_btn = QPushButton("New location")
        self.new_loc_btn.clicked.connect(self.open_location_manager)

        loc_layout = QHBoxLayout()
        loc_layout.addWidget(self.location_combo)
        loc_layout.addWidget(self.new_loc_btn)
        
        form_layout.addRow("Location:", loc_layout)
        
        # Plain text details
        self.lat_label = QLabel("-")
        self.lon_label = QLabel("-")
        self.elev_label = QLabel("-")
        
        form_layout.addRow("Latitude:", self.lat_label)
        form_layout.addRow("Longitude:", self.lon_label)
        form_layout.addRow("Elevation:", self.elev_label)

        # User inputs
        self.time_begin_edit = QLineEdit()
        self.time_begin_edit.setPlaceholderText("YYYY-MM-DD HH:MM")
        form_layout.addRow("Start Time:", self.time_begin_edit)

        self.horizon_edit = QLineEdit()
        self.horizon_edit.setText("0")
        form_layout.addRow("Horizon Limit (deg):", self.horizon_edit)

        self.mag_edit = QLineEdit()
        self.mag_edit.setPlaceholderText("e.g. 6.0 (default)")
        form_layout.addRow("Limiting Magnitude:", self.mag_edit)

        left_layout.addLayout(form_layout)
        
        # Object Type Filters
        self.star_check = QCheckBox("Stars")
        self.galaxy_check = QCheckBox("Galaxies")
        self.nebula_check = QCheckBox("Nebulae")
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.star_check)
        filter_layout.addWidget(self.galaxy_check)
        filter_layout.addWidget(self.nebula_check)
        left_layout.addLayout(filter_layout)

        # Action Button
        self.check_btn = QPushButton("Check Observable Objects")
        self.check_btn.clicked.connect(self.check_observability)
        left_layout.addWidget(self.check_btn)

        left_layout.addStretch()
        main_layout.addWidget(left_widget, 1)

        # Right Container (Table + Button)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(["", "Name", "Type", "RA", "Dec", "Observable", "Mag", "Deep Dive"])
        
        for i in range(self.results_table.columnCount()):
            self.results_table.horizontalHeaderItem(i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Web-style Table Formatting
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setShowGrid(False)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
        # Add tooltips to headers to explain columns
        self.results_table.horizontalHeaderItem(0).setToolTip("Select to save")
        self.results_table.horizontalHeaderItem(1).setToolTip("Primary Identifier")
        self.results_table.horizontalHeaderItem(2).setToolTip("Object Classification")
        self.results_table.horizontalHeaderItem(3).setToolTip("Right Ascension (ICRS)")
        self.results_table.horizontalHeaderItem(4).setToolTip("Declination (ICRS)")
        self.results_table.horizontalHeaderItem(5).setToolTip("Time range when object is visible (Local Time)")
        self.results_table.horizontalHeaderItem(6).setToolTip("Visual Magnitude (Lower is brighter)")
        self.results_table.horizontalHeaderItem(7).setToolTip("View detailed object data")
        
        right_layout.addWidget(self.results_table)
        
        btns_layout = QHBoxLayout()
        self.save_list_btn = QPushButton("Save Checked Items to List")
        self.save_list_btn.clicked.connect(self.save_checked_items)
        btns_layout.addWidget(self.save_list_btn)
        self.manage_lists_btn = QPushButton("Manage Lists")
        self.manage_lists_btn.clicked.connect(self.open_list_manager)
        btns_layout.addWidget(self.manage_lists_btn)
        right_layout.addLayout(btns_layout)
        
        main_layout.addWidget(right_widget, 2)

        self.tabs.addTab(self.observable_tab, "Observable Objects")

        # --- Tab 2: Object Deep Dive ---
        self.deep_dive_tab = QWidget()
        
        tab_layout = QVBoxLayout(self.deep_dive_tab)
        
        # Top Search Bar
        search_layout = QHBoxLayout()
        self.dd_search_edit = QLineEdit()
        self.dd_search_edit.setPlaceholderText("Enter object name (e.g. M31, Sirius, NGC 224)...")
        self.dd_search_edit.returnPressed.connect(self.perform_deep_dive_search)
        search_layout.addWidget(self.dd_search_edit)
        
        self.dd_search_btn = QPushButton("Search")
        self.dd_search_btn.clicked.connect(self.perform_deep_dive_search)
        search_layout.addWidget(self.dd_search_btn)
        tab_layout.addLayout(search_layout)

        # Main content area (Info on left, Maps on right)
        content_layout = QHBoxLayout()
        
        # Left side: Info Panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        
        deep_dive_layout = QVBoxLayout(scroll_content)
        
        self.dd_results_widget = QWidget()
        self.dd_results_widget.setVisible(False)
        res_layout = QVBoxLayout(self.dd_results_widget)
        
        info_form = QFormLayout()
        info_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.dd_name_label = QLabel()
        self.dd_aliases_label = QLabel()
        self.dd_aliases_label.setWordWrap(True)
        self.dd_aliases_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self.dd_type_label = QLabel()
        self.dd_coords_label = QLabel()
        self.dd_mag_label = QLabel()
        self.dd_dist_label = QLabel() # New label for distance
        
        info_form.addRow("Main Name:", self.dd_name_label)
        info_form.addRow("Aliases:", self.dd_aliases_label)
        info_form.addRow("Type:", self.dd_type_label)
        info_form.addRow("Coordinates (RA/Dec):", self.dd_coords_label)
        info_form.addRow("Magnitude (V):", self.dd_mag_label)
        info_form.addRow("Distance (ly):", self.dd_dist_label) # Add to form
        res_layout.addLayout(info_form)
        
        # New Buttons
        dd_btns_row1 = QHBoxLayout()
        self.photos_btn = QPushButton("Photos")
        self.photos_btn.clicked.connect(self.open_photos_window)
        self.spectroscopy_btn = QPushButton("Spectroscopy")
        self.spectroscopy_btn.clicked.connect(lambda: self.log("Spectroscopy functionality coming soon"))
        dd_btns_row1.addWidget(self.photos_btn)
        dd_btns_row1.addWidget(self.spectroscopy_btn)
        
        dd_btns_row2 = QHBoxLayout()
        self.spatial_cube_btn = QPushButton("Spatial Cube")
        self.spatial_cube_btn.clicked.connect(lambda: self.log("Spatial Cube functionality coming soon"))
        self.hidden_stars_btn = QPushButton("Hidden Stars")
        self.hidden_stars_btn.clicked.connect(lambda: self.log("Hidden Stars functionality coming soon"))
        dd_btns_row2.addWidget(self.spatial_cube_btn)
        dd_btns_row2.addWidget(self.hidden_stars_btn)
        
        dd_btns_row3 = QHBoxLayout()
        self.dd_gas_btn = QPushButton("Check for Surrounding Gas")
        self.dd_gas_btn.clicked.connect(self.open_surrounding_gas_window)
        self.comet_tracker_btn = QPushButton("Comet Tracker")
        self.comet_tracker_btn.clicked.connect(lambda: self.log("Comet Tracker functionality coming soon"))
        dd_btns_row3.addWidget(self.dd_gas_btn)
        dd_btns_row3.addWidget(self.comet_tracker_btn)
        
        res_layout.addLayout(dd_btns_row1)
        res_layout.addLayout(dd_btns_row2)
        res_layout.addLayout(dd_btns_row3)
        
        deep_dive_layout.addWidget(self.dd_results_widget)
        deep_dive_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        content_layout.addWidget(scroll_area, 1) # Info panel takes 1/3 of space

        # Right side: Container for all maps (Scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        self.right_scroll_content = QWidget()
        self.right_layout = QVBoxLayout(self.right_scroll_content)

        # Gas Map container (Always at top)
        self.gas_map_figure = Figure(facecolor=PLOT_BG_COLOR)
        self.gas_map_canvas = FigureCanvasQTAgg(self.gas_map_figure)
        self.gas_map_canvas.setMinimumHeight(500) # Fixed height to prevent shrinking
        self.gas_map_canvas.setVisible(False) # Hidden by default
        self.right_layout.addWidget(self.gas_map_canvas)
        
        # Container for dynamic photo widgets
        self.photos_container = QWidget()
        self.photos_layout = QVBoxLayout(self.photos_container)
        self.right_layout.addWidget(self.photos_container)
        
        self.right_layout.addStretch()
        right_scroll.setWidget(self.right_scroll_content)
        
        content_layout.addWidget(right_scroll, 2) # Changed from 3 to 2 for 1:2 ratio
        
        tab_layout.addLayout(content_layout)
        self.tabs.addTab(self.deep_dive_tab, "Object Deep Dive")

        # Load initial config
        self.load_data()

    def load_data(self):
        """Loads all locations and session data from adastra_data.json."""
        data_file = 'adastra_data.json'
        if not os.path.exists(data_file):
            self.log("No data file found. Trying to import from old config.csv...")
            self.locations = {}
            try:
                with open('config.csv', newline='') as csvfile:
                    c = csv.reader(csvfile)
                    next(c)  # Skip header
                    row = next(c)
                    if len(row) == 1: row = row[0].split(',')
                    lat, lon, elev, name = row
                    self.locations[name.strip()] = {
                        "latitude": float(lat), "longitude": float(lon), "elevation": float(elev)
                    }
                    self.log("Successfully imported location from config.csv.")
                    self.save_data() # Create the new data file
            except Exception as e:
                self.log(f"Could not import from config.csv: {e}")

        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                self.locations = data.get('locations', {})
                
                # Populate dropdown
                self.location_combo.blockSignals(True)
                self.location_combo.clear()
                self.location_combo.addItems(sorted(self.locations.keys()))
                self.location_combo.blockSignals(False)

                # Set to last used location
                last_loc = data.get('last_used_location')
                if last_loc in self.locations:
                    self.location_combo.setCurrentText(last_loc)
                
                # Trigger update of labels
                self.on_location_change(self.location_combo.currentIndex())
                
                # Restore times
                self.time_begin_edit.setText(data.get('last_start_time', ''))
                
                # Restore window geometry
                if 'window_geometry' in data:
                    self.restoreGeometry(bytes.fromhex(data['window_geometry']))
                
                # Restore column widths
                if 'column_widths' in data:
                    widths = data['column_widths']
                    for i, w in enumerate(widths):
                        if i < self.results_table.columnCount():
                            self.results_table.setColumnWidth(i, w)

                self.log("Loaded saved locations and session data.")

        except Exception as e:
            self.log(f"Could not load data file: {e}")

    def save_data(self):
        """Saves all locations and session data to adastra_data.json."""
        try:
            col_widths = [self.results_table.columnWidth(i) for i in range(self.results_table.columnCount())]
            
            data = {
                'last_used_location': self.location_combo.currentText(),
                'last_start_time': self.time_begin_edit.text(),
                'locations': self.locations,
                'window_geometry': self.saveGeometry().toHex().data().decode(),
                'column_widths': col_widths
            }
            with open('adastra_data.json', 'w') as f:
                json.dump(data, f, indent=4)
            self.log("Saved locations and session data.")
        except Exception as e:
            self.log(f"Could not save data: {e}")

    def closeEvent(self, event):
        self.save_data()
        event.accept()
        
    def save_checked_items(self):
        """Saves checked items from the results table to a JSON list."""
        checked_rows = []
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                # Gather data (Name is in col 1, Type in col 2, etc.)
                name_item = self.results_table.item(row, 1)
                name = name_item.sort_value if hasattr(name_item, 'sort_value') else "Unknown"
                
                type_ = self.results_table.item(row, 2).text()
                ra = self.results_table.item(row, 3).text()
                dec = self.results_table.item(row, 4).text()
                obs = self.results_table.item(row, 5).text()
                mag = self.results_table.item(row, 6).text()
                
                checked_rows.append({
                    "Name": name, "Type": type_, "RA": ra, "Dec": dec,
                    "Observable": obs, "Mag": mag
                })
        
        if not checked_rows:
            QMessageBox.information(self, "Info", "No items checked.")
            return

        dialog = SaveListDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            list_name = dialog.list_name
            lists_data = {}
            if os.path.exists('adastra_lists.json'):
                try:
                    with open('adastra_lists.json', 'r') as f:
                        lists_data = json.load(f)
                except:
                    pass
            
            if list_name in lists_data:
                lists_data[list_name].extend(checked_rows)
            else:
                lists_data[list_name] = checked_rows
            
            try:
                with open('adastra_lists.json', 'w') as f:
                    json.dump(lists_data, f, indent=4)
                self.log(f"Saved {len(checked_rows)} items to list '{list_name}'")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save list: {e}")

    def on_location_change(self, index):
        """Updates the Lat/Lon/Elev fields when a new location is selected."""
        loc_name = self.location_combo.currentText()
        if loc_name in self.locations:
            loc_data = self.locations[loc_name]
            self.lat_label.setText(str(loc_data.get('latitude', '-')))
            self.lon_label.setText(str(loc_data.get('longitude', '-')))
            self.elev_label.setText(str(loc_data.get('elevation', '-')))
        else:
            self.lat_label.setText("-")
            self.lon_label.setText("-")
            self.elev_label.setText("-")

    def open_location_manager(self):
        dialog = LocationManagerDialog(self)
        dialog.exec()

    def show_aliases(self, object_name):
        dialog = AliasesDialog(object_name, self)
        dialog.exec()

    def open_list_manager(self):
        dialog = ListManagerDialog(self)
        dialog.exec()

    def refresh_locations(self):
        current = self.location_combo.currentText()
        self.location_combo.blockSignals(True)
        self.location_combo.clear()
        self.location_combo.addItems(sorted(self.locations.keys()))
        if current in self.locations:
            self.location_combo.setCurrentText(current)
        self.location_combo.blockSignals(False)
        self.on_location_change(self.location_combo.currentIndex())

    def log(self, message):
        """Helper to print to console and update the GUI status bar."""
        print(message)
        self.statusBar().showMessage(message)

    def create_observer(self):
        """
        Instantiates the astroplan.Observer object based on user input.
        This object is required for all subsequent astronomical calculations.
        Returns True if successful, False otherwise.
        """
        try:
            loc_name = self.location_combo.currentText()
            if not loc_name or loc_name not in self.locations:
                raise ValueError("Please select a valid location.")
            loc_data = self.locations[loc_name]
            lat = float(loc_data['latitude']) * u.deg
            lon = float(loc_data['longitude']) * u.deg
            elev = float(loc_data['elevation']) * u.m
            name = loc_name

            tf = TimezoneFinder()
            # Automatically determine timezone string (e.g., 'America/New_York') from coordinates
            timezone_str = tf.timezone_at(lng=lon.value, lat=lat.value)
            
            # Create the Observer object
            self.observer = astroplan.Observer(latitude=lat, longitude=lon, elevation=elev, name=name, timezone=timezone_str)

            self.log(f"\nObserver Created: {self.observer.name}\nLocation: {self.observer.location}\nTimezone: {self.observer.timezone}")
            if self.time_begin_edit.text(): self.log(f"Start: {self.time_begin_edit.text()}")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return False

    def check_observability(self):
        """
        Performs a grid search of the visible sky to find observable objects.
        1. Generates a grid of Alt/Az points representing the local sky.
        2. Queries SIMBAD for objects around those points that meet magnitude criteria.
        """
        if not self.create_observer():
            return

        # Get Time
        start_str = self.time_begin_edit.text()

        if not start_str:
            QMessageBox.warning(self, "Warning", "Please enter start time.")
            return

        try:
            # Parse start time as local time using the observer's timezone
            dt_naive = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
            tz = self.observer.timezone
            if isinstance(tz, str):
                tz = pytz.timezone(tz)
            dt_aware = tz.localize(dt_naive)
            t_start = Time(dt_aware)
            
            # Automatically calculate end time as next nautical dawn
            t_end = self.observer.twilight_morning_nautical(t_start, which='next')
            self.log(f"Observation window: {t_start.iso} to {t_end.iso} (Nautical Dawn)")

            # Get Horizon Limit
            horizon = 0.0
            if self.horizon_edit.text():
                horizon = float(self.horizon_edit.text())

            # Get Magnitude Limit
            mag_limit = 6.0
            if self.mag_edit.text():
                mag_limit = float(self.mag_edit.text())
            
            # Get Filter States
            show_stars = self.star_check.isChecked()
            show_galaxies = self.galaxy_check.isChecked()
            show_nebulae = self.nebula_check.isChecked()
            filter_active = show_stars or show_galaxies or show_nebulae

            # Save all current settings for the next session
            self.save_data()

            # Configure SIMBAD
            custom_simbad = Simbad()
            custom_simbad.ROW_LIMIT = 0

            # Define Grid of Cone Queries
            # Strategy: Instead of querying the whole sky, we query specific regions 
            # that are currently visible to the observer.
            # We start 10 degrees above the user-defined horizon limit to ensure good visibility.
            alt_min = horizon + 10.0
            if alt_min >= 90: alt_min = 89.0
            
            # Generate grid points (Alt/Az)
            grid_coords = []
            step = 15 # Step size in degrees for the grid
            
            def get_grid(time_obj):
                coords = []
                for alt in range(int(alt_min), 90, step):
                    for az in range(0, 360, step):
                        c = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame='altaz', 
                                     obstime=time_obj, location=self.observer.location)
                        coords.append(c.transform_to('icrs'))
                return coords

            # 1. Grid at Start Time (Current Sky)
            grid_coords.extend(get_grid(t_start))
            
            # 2. Grid at End Time (Future Sky - captures objects rising during the night)
            if (t_end - t_start).to(u.hour).value > 1.0:
                grid_coords.extend(get_grid(t_end))
            
            # Filter duplicates/overlapping regions to optimize query count
            # If a new point is within 10 degrees (cone radius) of an existing point, skip it.
            final_grid = []
            if grid_coords:
                final_grid.append(grid_coords[0])
                for c in grid_coords[1:]:
                    # Check separation from existing points
                    # Create catalog from RA/Dec to avoid frame mismatch errors (due to different obstimes)
                    catalog = SkyCoord([x.ra for x in final_grid], [x.dec for x in final_grid], frame='icrs')
                    if c.separation(catalog).min() > 10 * u.deg:
                        final_grid.append(c)
            
            grid_coords = final_grid

            self.log(f"Starting SIMBAD grid search ({len(grid_coords)} queries)...")
            self.log("Scanning region >10 deg above horizon. Please wait...")
            
            unique_objects = {} # Dictionary to store results and prevent duplicates (Key: Main ID)
            
            for i, coord in enumerate(grid_coords):
                self.log(f"Querying region {i+1}/{len(grid_coords)}...")
                # Keep GUI responsive
                QApplication.processEvents()
                
                try:
                    # Server-side filtering using TAP (ADQL)
                    # We use ADQL (Astronomical Data Query Language) to filter data on the server.
                    # This is much faster than downloading everything and filtering in Python.
                    ra_deg = coord.ra.deg
                    dec_deg = coord.dec.deg
                    
                    # Query Explanation:
                    # JOIN basic and flux tables.
                    # CONTAINS/POINT/CIRCLE: Select objects within 10 degrees of our grid point.
                    # flux."filter" = 'V': Look for Visual magnitude data.
                    # flux.flux < {mag_limit}: Filter for brightness (lower magnitude = brighter).
                    query = f"""
                        SELECT basic.main_id, basic.ra, basic.dec, basic.otype, flux.flux
                        FROM basic
                        JOIN flux ON basic.oid = flux.oidref
                        WHERE 1=CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra_deg}, {dec_deg}, 10))
                        AND flux."filter" = 'V'
                        AND flux.flux < {mag_limit}
                    """
                    
                    result_table = custom_simbad.query_tap(query)
                    
                    if result_table:
                        for row in result_table:
                            try:
                                # Decode bytes to string if necessary (common issue with VOTable data)
                                name = row['main_id']
                                if isinstance(name, bytes): name = name.decode('utf-8')
                                name = str(name)
                                
                                # Skip if we already found this object in a previous grid overlap
                                if name in unique_objects:
                                    continue
                                
                                ra = float(row['ra'])
                                dec = float(row['dec'])
                                otype = row['otype']
                                if isinstance(otype, bytes): otype = otype.decode('utf-8')
                                otype = str(otype).strip()
                                otype = OTYPE_MAP.get(otype, otype) # Convert to human readable
                                mag = float(row['flux'])
                                
                                # Apply Type Filters
                                if filter_active:
                                    is_star = any(k in otype for k in ["Star", "Binary", "Dwarf", "Giant", "Supergiant", "Pulsar", "Nova", "Variable", "Stellar"])
                                    is_galaxy = any(k in otype for k in ["Galaxy", "Galaxies", "Quasar", "Blazar", "BL Lac", "AGN"])
                                    is_nebula = any(k in otype for k in ["Nebula", "Cloud", "Remnant", "Region", "Bubble", "Globule", "Filament", "Shell"])
                                    
                                    keep = False
                                    if show_stars and is_star: keep = True
                                    if show_galaxies and is_galaxy: keep = True
                                    if show_nebulae and is_nebula: keep = True
                                    
                                    if not keep:
                                        continue
                                
                                unique_objects[name] = {
                                    'name': name,
                                    'type': otype,
                                    'ra': ra,
                                    'dec': dec,
                                    'mag': mag,
                                    'target': FixedTarget(coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), name=name)
                                }
                            except Exception:
                                continue
                except Exception as e:
                    print(f"Query failed: {e}", file=sys.stderr)
                    self.log(f"Query failed: {e}")
                
                # 1 second delay to respect SIMBAD server usage policies
                time.sleep(1)

            # Display Results
            self.results_table.setSortingEnabled(False)
            self.results_table.setRowCount(0)

            # Pre-calculate window start for time range logic
            # If start time is already night, use it. Otherwise wait for nautical dusk.
            is_night_start = self.observer.is_night(t_start, horizon=-12*u.deg)
            if is_night_start:
                window_start = t_start
            else:
                window_start = self.observer.twilight_evening_nautical(t_start, which='next')
                if window_start > t_end:
                    window_start = t_end
            
            tz_obj = tz

            objects_list = list(unique_objects.values())
            for i, obj in enumerate(objects_list):
                
                # Calculate Time Range
                range_str = "-"
                range_sort = 9999999.0
                
                if window_start < t_end:
                    try:
                        target = obj['target']
                        h_quant = horizon * u.deg
                        
                        # Check if object is up at the start of the effective window
                        is_up = self.observer.target_is_up(window_start, target, horizon=h_quant)
                        
                        s_time = None
                        e_time = None
                        
                        if is_up:
                            s_time = window_start
                            try:
                                # Find next set time
                                next_set = self.observer.target_set_time(window_start, target, which='next', horizon=h_quant)
                                e_time = min(next_set, t_end)
                            except (astroplan.TargetAlwaysUp, astroplan.TargetNeverUp):
                                e_time = t_end
                        else:
                            try:
                                # Find next rise time
                                next_rise = self.observer.target_rise_time(window_start, target, which='next', horizon=h_quant)
                                if next_rise < t_end:
                                    s_time = next_rise
                                    try:
                                        # Find set time after rise
                                        next_set = self.observer.target_set_time(next_rise, target, which='next', horizon=h_quant)
                                        e_time = min(next_set, t_end)
                                    except (astroplan.TargetAlwaysUp, astroplan.TargetNeverUp):
                                        e_time = t_end
                            except (astroplan.TargetAlwaysUp, astroplan.TargetNeverUp):
                                pass
                        
                        if s_time and e_time:
                            s_dt = s_time.to_datetime(timezone=tz_obj)
                            e_dt = e_time.to_datetime(timezone=tz_obj)
                            range_str = f"{s_dt.strftime('%H:%M')} - {e_dt.strftime('%H:%M')}"
                            range_sort = s_time.jd
                    except Exception:
                        pass

                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                
                # Checkbox
                check_item = QTableWidgetItem()
                check_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                check_item.setCheckState(Qt.CheckState.Unchecked)
                self.results_table.setItem(row, 0, check_item)
                
                self.results_table.setItem(row, 1, NumericTableWidgetItem("", sort_value=obj['name']))
                link_label = QLabel(f'{obj["name"]}<br><a href="{obj["name"]}">Show Aliases</a>')
                link_label.setObjectName("ResultsLink")
                link_label.setOpenExternalLinks(False)
                link_label.linkActivated.connect(self.show_aliases)
                link_label.setMinimumHeight(50)
                link_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setCellWidget(row, 1, link_label)
                
                type_item = QTableWidgetItem(obj['type'])
                type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, 2, type_item)
                
                ra_str = obj['target'].coord.ra.to_string(unit=u.hour, sep=('h', 'm', 's'), precision=1, pad=True)
                dec_str = obj['target'].coord.dec.to_string(unit=u.deg, sep=('d', 'm', 's'), precision=1, alwayssign=True, pad=True)
                
                self.results_table.setItem(row, 3, NumericTableWidgetItem(ra_str, sort_value=obj['ra']))
                self.results_table.setItem(row, 4, NumericTableWidgetItem(dec_str, sort_value=obj['dec']))
                self.results_table.setItem(row, 5, NumericTableWidgetItem(range_str, sort_value=range_sort))
                self.results_table.setItem(row, 6, NumericTableWidgetItem(f"{obj['mag']:.2f}"))
                
                dd_btn = QPushButton("Deep Dive")
                dd_btn.clicked.connect(lambda checked, n=obj['name']: self.open_deep_dive(n))
                self.results_table.setCellWidget(row, 7, dd_btn)
            
            self.results_table.setSortingEnabled(True)
            self.results_table.resizeRowsToContents()
            self.log(f"Found {len(unique_objects)} visible objects.")

        except Exception as e:
            self.log(f"Error checking observability: {e}")

    def open_deep_dive(self, name):
        """Switches to Deep Dive tab and searches for the object."""
        self.tabs.setCurrentWidget(self.deep_dive_tab)
        self.dd_search_edit.setText(name)
        self.perform_deep_dive_search()

    def perform_deep_dive_search(self):
        """Searches for a specific object by name/alias and displays details."""
        name = self.dd_search_edit.text().strip()
        if not name: return
        
        # Clear maps from previous searches
        self.gas_map_figure.clear()
        self.gas_map_canvas.draw()
        
        # Clear dynamic photo widgets
        for i in reversed(range(self.photos_layout.count())): 
            item = self.photos_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.dynamic_photo_widgets = []
        
        self.log(f"Searching SIMBAD for '{name}'...")
        QApplication.processEvents()
        
        try:
            # 1. Query Object Properties
            table = None
            try:
                custom_simbad = Simbad()
                custom_simbad.add_votable_fields('V', 'otype', 'ra', 'dec', 'plx') # Added plx
                table = custom_simbad.query_object(name)
            except Exception as e:
                self.log(f"Extended query failed ({e}), retrying with defaults...")
                custom_simbad = Simbad() # Reset to defaults
                table = custom_simbad.query_object(name)
            
            if table:
                row = table[0]
                
                # Extract Data
                if 'MAIN_ID' in row.colnames: main_id = row['MAIN_ID']
                elif 'main_id' in row.colnames: main_id = row['main_id']
                else: main_id = name
                if isinstance(main_id, bytes): main_id = main_id.decode('utf-8')
                main_id = str(main_id)
                
                if 'ra' in row.colnames: ra = row['ra']
                elif 'RA_d' in row.colnames: ra = row['RA_d']
                else: ra = 0.0
                
                if 'dec' in row.colnames: dec = row['dec']
                elif 'DEC_d' in row.colnames: dec = row['DEC_d']
                else: dec = 0.0
                
                if 'OTYPE' in row.colnames: otype = row['OTYPE']
                elif 'otype' in row.colnames: otype = row['otype']
                else: otype = '?'
                if isinstance(otype, bytes): otype = otype.decode('utf-8')
                human_type = OTYPE_MAP.get(str(otype).strip(), str(otype))
                
                mag = "-"
                val = None
                if 'V' in row.colnames: val = row['V']
                elif 'FLUX_V' in row.colnames: val = row['FLUX_V']
                if val is not None and not np.ma.is_masked(val): mag = f"{float(val):.2f}"
                
                # Distance Calculation
                dist_ly = "-"
                if 'PLX_VALUE' in row.colnames: plx = row['PLX_VALUE']
                elif 'plx_value' in row.colnames: plx = row['plx_value']
                else: plx = None
                
                if plx is not None and not np.ma.is_masked(plx) and float(plx) > 0:
                    # d (pc) = 1000 / plx (mas)
                    # d (ly) = d (pc) * 3.262
                    d_pc = 1000.0 / float(plx)
                    d_ly = d_pc * 3.26156
                    dist_ly = f"{d_ly:,.1f}"
                
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                self.current_dd_coord = coord
                self.current_dd_name = main_id
                
                ra_str = coord.ra.to_string(unit=u.hour, sep=('h', 'm', 's'), precision=1, pad=True)
                dec_str = coord.dec.to_string(unit=u.deg, sep=('d', 'm', 's'), precision=1, alwayssign=True, pad=True)
                
                self.dd_name_label.setText(main_id)
                self.dd_type_label.setText(human_type)
                self.dd_coords_label.setText(f"{ra_str}, {dec_str}")
                self.dd_mag_label.setText(mag)
                self.dd_dist_label.setText(dist_ly)
                
                # 2. Query Aliases
                ids_table = None
                try:
                    ids_table = custom_simbad.query_objectids(main_id)
                except Exception as e:
                    self.log(f"Could not fetch aliases: {e}")

                aliases_list = []
                if ids_table:
                    for id_row in ids_table:
                        val = id_row[0]
                        if isinstance(val, bytes): val = val.decode('utf-8')
                        aliases_list.append(' '.join(str(val).split()))
                
                self.dd_aliases_label.setText(f"<div style='padding: 10px;'>{', '.join(aliases_list)}</div>")
                self.dd_results_widget.setVisible(True)
                self.log(f"Deep dive data loaded for {main_id}")
                
                # 3. Load Content (Gas/Dust Maps AND Photos)
                self.load_deep_dive_content()
                
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find object '{name}' in SIMBAD.")
                self.log(f"Object '{name}' not found.")
                
        except Exception as e:
            self.log(f"Deep dive search error: {e}")
            QMessageBox.critical(self, "Error", f"Search failed: {e}")

    def load_deep_dive_content(self):
        """Fetches and displays only the 2MASS image in the main window."""
        if not self.current_dd_coord: return
        fov_deg = 2.0

        # Clear previous content
        self.gas_map_canvas.setVisible(False)
        for i in reversed(range(self.photos_layout.count())): 
            item = self.photos_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.dynamic_photo_widgets = []

        # Fetch 2MASS Color Only
        self.log("Fetching 2MASS Color image...")
        QApplication.processEvents()
        
        try:
            result = hips2fits.query(
                hips='CDS/P/2MASS/color', width=500, height=500,
                ra=self.current_dd_coord.ra, dec=self.current_dd_coord.dec,
                fov=fov_deg*u.deg, projection='TAN', format='fits'
            )
            
            if result:
                self.add_photo_widget(result[0].data, '2MASS Color (IR)')
                self.log("2MASS Color loaded.")
            else:
                self.log("Failed to load 2MASS Color.")
                
        except Exception as e:
            self.log(f"Error fetching 2MASS Color: {e}")

    def open_surrounding_gas_window(self):
        """Opens a new window with Gas, Dust, and Optical images."""
        if not self.current_dd_coord:
            QMessageBox.warning(self, "Warning", "Please search for an object first.")
            return

        fov_deg = 2.0

        self.log("Fetching maps for Surrounding Gas window...")
        QApplication.processEvents()

        surveys = [
            {'id': 'CDS/P/HI4PI/NH', 'name': 'HI4PI (Hydrogen)', 'cmap': 'inferno'},
            {'id': 'CDS/P/AKARI/FIS/N160', 'name': 'AKARI (Dust)', 'cmap': 'inferno'},
            {'id': 'CDS/P/DSS2/red', 'name': 'DSS2 Red', 'cmap': 'gray'},
            {'id': 'CDS/P/DSS2/blue', 'name': 'DSS2 Blue', 'cmap': 'gray'}
        ]
        
        images_to_plot = []
        
        for survey in surveys:
            try:
                result = hips2fits.query(
                    hips=survey['id'], width=500, height=500,
                    ra=self.current_dd_coord.ra, dec=self.current_dd_coord.dec,
                    fov=fov_deg*u.deg, projection='TAN', format='fits'
                )
                if result:
                    images_to_plot.append((result[0].data, survey['name'], survey['cmap']))
                else:
                    # Fallback for Dust
                    if 'AKARI' in survey['name']:
                        try:
                            result = hips2fits.query(
                                hips='CDS/P/Planck/R2/HFI/857', width=500, height=500,
                                ra=self.current_dd_coord.ra, dec=self.current_dd_coord.dec,
                                fov=fov_deg*u.deg, projection='TAN', format='fits'
                            )
                            if result:
                                images_to_plot.append((result[0].data, 'Planck (Dust)', 'inferno'))
                        except:
                            pass
            except Exception as e:
                self.log(f"Failed to fetch {survey['name']}: {e}")

        if images_to_plot:
            dialog = MultiImageDialog(f"Surrounding Gas & Optical: {self.current_dd_name}", images_to_plot, layout_shape=(2, 2), parent=self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Error", "No images could be retrieved.")

    def open_photos_window(self):
        """Opens a new window with survey photos."""
        if not self.current_dd_coord:
            QMessageBox.warning(self, "Warning", "Please search for an object first.")
            return

        fov_deg = 2.0

        self.log("Fetching survey photos...")
        QApplication.processEvents()

        surveys = [
            {'id': 'CDS/P/DSS2/red', 'name': 'DSS2 Red'},
            {'id': 'CDS/P/DSS2/blue', 'name': 'DSS2 Blue'},
            {'id': 'CDS/P/2MASS/color', 'name': '2MASS Color (IR)'}
        ]
        
        images_to_plot = []
        for survey in surveys:
            try:
                result = hips2fits.query(
                    hips=survey['id'], width=500, height=500,
                    ra=self.current_dd_coord.ra, dec=self.current_dd_coord.dec,
                    fov=fov_deg*u.deg, projection='TAN', format='fits'
                )
                if result:
                    # 2MASS Color is RGB, others are grayscale
                    cmap = 'gray'
                    data = result[0].data
                    if data.ndim == 3 and data.shape[0] == 3:
                        data = np.transpose(data, (1, 2, 0))
                        cmap = None # RGB doesn't use cmap
                    
                    images_to_plot.append((data, survey['name'], cmap))
            except Exception as e:
                self.log(f"Failed to fetch {survey['name']}: {e}")
            
        if images_to_plot:
            dialog = MultiImageDialog(f"Survey Photos: {self.current_dd_name}", images_to_plot, layout_shape=(1, 3), parent=self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Error", "No photos could be retrieved.")

    def find_surrounding_gas(self):
        # Kept for button compatibility, but logic moved to load_deep_dive_content
        self.open_surrounding_gas_window()

    def fetch_photo(self):
        # Kept for button compatibility
        self.load_deep_dive_content()

    def add_photo_widget(self, image_data, name):
        """Creates a clickable thumbnail for a photo."""
        # Create a container for the thumbnail
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a ClickableLabel
        # We need to convert the numpy array to a QPixmap for display
        # This is a simplified conversion for thumbnail purposes
        
        # Normalize data for display
        if image_data.ndim == 3:
             if image_data.shape[0] == 3:
                 image_data = np.transpose(image_data, (1, 2, 0))
             
             # Apply Asinh stretch to each channel to improve contrast
             h, w, ch = image_data.shape
             stretched_data = np.zeros((h, w, ch), dtype=float)
             
             for i in range(3):
                 channel = image_data[:, :, i]
                 channel = np.nan_to_num(channel)
                 stretch = AsinhStretch(a=0.1)
                 vmin, vmax = np.min(channel), np.max(channel)
                 if vmax > vmin:
                     norm = ImageNormalize(stretch=stretch, vmin=vmin, vmax=vmax)
                     res = norm(channel)
                     if np.ma.is_masked(res):
                         res = res.filled(0)
                     stretched_data[:, :, i] = res

             # Convert to 8-bit for QImage
             display_data = (stretched_data * 255).astype(np.uint8)
             display_data = np.ascontiguousarray(display_data)
             h, w, ch = display_data.shape
             qimg = QImage(display_data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            norm_data = image_data.astype(float)
            norm_data = np.nan_to_num(norm_data)
            dmin, dmax = np.min(norm_data), np.max(norm_data)
            if dmax > dmin:
                norm_data = (norm_data - dmin) / (dmax - dmin)
            else:
                norm_data[:] = 0
            display_data = (norm_data * 255).astype(np.uint8)
            display_data = np.ascontiguousarray(display_data)
            h, w = display_data.shape
            qimg = QImage(display_data.data, w, h, w, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg)
        # Scale down for thumbnail
        pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        label = ClickableLabel(image_data, name)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.clicked.connect(self.open_image_dialog)
        
        title_lbl = QLabel(name)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(label)
        layout.addWidget(title_lbl)
        
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        
        self.photos_layout.addWidget(container)
        self.photos_layout.addWidget(line)
        self.dynamic_photo_widgets.append(container)
        self.dynamic_photo_widgets.append(line)

    def open_image_dialog(self, image_data, title):
        dialog = ImageDialog(image_data, title, self)
        dialog.exec()

    def open_overlay(self, base_data, overlay_type, base_name):
        if overlay_type in self.overlay_maps:
            overlay_data = self.overlay_maps[overlay_type]
            dialog = OverlayDialog(base_data, overlay_data, f"{overlay_type} on {base_name}", self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Error", f"{overlay_type} map data not available.")


if __name__ == "__main__":
    # Download IERS data (Earth rotation data) required for precise time/coordinate conversions
    print("Checking IERS data...")
    try:
        astroplan.download_IERS_A(show_progress=False)
    except:
        pass
    
    app = QApplication(sys.argv)
    
    # Load external stylesheet
    try:
        with open('style.css', 'r') as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("Warning: style.css not found. Using default styles.")

    app.setWindowIcon(QIcon('icon.png'))
    window = AdAstraWindow()
    window.show()
    sys.exit(app.exec())
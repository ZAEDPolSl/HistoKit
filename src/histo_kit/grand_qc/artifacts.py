from enum import Enum

class Artifact(Enum):
    """
    Enumeration of artifacts detected in Whole Slide Images (WSI).

    Each member corresponds to a specific type of artifact, background or tissue area
    defined during artifact detection with the GrandQC model.

    Attributes
    ----------
    BG_THR : int
        Background after mask detection (black).
    NORM : int
        Normal artifact (gray).
    ART_FOLD : int
        Tissue fold artifact (orange).
    ART_DARKSPOT : int
        Dark spot artifact (green).
    ART_PEN : int
        Pen marking artifact (red).
    ART_EDGE : int
        Edge artifact (pink).
    ART_FOCUS : int
        Out-of-focus region (violet).
    BG_MODEL : int
        Background predicted by the artifact detection model (blue).

    """

    BG_THR = 0       # BACKGROUND (after mask detection): black
    NORM = 1         # ART_NORM: gray
    ART_FOLD = 2     # ART_FOLD: orange
    ART_DARKSPOT = 3 # ART_DARKSPOT: green
    ART_PEN = 4      # ART_PEN: red
    ART_EDGE = 5     # ART_EDGE: pink
    ART_FOCUS = 6    # ART_FOCUS: violet
    BG_MODEL = 7     # BACKGROUND (predicted by artifact detection model): blue



    def __getitem__(self, index):
        _colors = [[0, 0, 0],
                   [128, 128, 128],
                   [255, 99, 71],
                   [0, 255, 0],
                   [255, 0, 0],
                   [255, 0, 255],
                   [75, 0, 130],
                   [50, 120, 230]]
        return _colors[index]

    def color(self):
        """
        Return the RGB color associated with this artifact.

        Returns
        -------
        list of int
            RGB triplet corresponding to the artifact type.
        """
        return self.__getitem__(self.value)

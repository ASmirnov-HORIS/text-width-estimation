from collections import namedtuple

# Classes

class FontFace(namedtuple("FontFace", ["bold", "italic"])):
    __slots__ = ()
    def __str__(self):
        if self.bold and self.italic:
            return "bold+italic"
        elif self.bold and not self.italic:
            return "bold"
        elif not self.bold and self.italic:
            return "italic"
        else:
            return "normal"

Font = namedtuple("Font", ["family", "size", "face"])

# Constants

MONOSPACED_TRAIN_FONT_FAMILIES = [
    "Courier",
]

PROPORTIONAL_TRAIN_FONT_FAMILIES = [
    "Geneva",
    "Georgia",
    "Helvetica",
    "Lucida Grande",
    "Times New Roman",
    "Verdana",
]

MONOSPACED_TEST_FONT_FAMILIES = [
    "Lucida Console",
]

PROPORTIONAL_TEST_FONT_FAMILIES = [
    "Arial",
    "Calibri",
    "Garamond",
    "Rockwell",
]

MONOSPACED_FONT_FAMILIES = MONOSPACED_TRAIN_FONT_FAMILIES + MONOSPACED_TEST_FONT_FAMILIES

PROPORTIONAL_FONT_FAMILIES = PROPORTIONAL_TRAIN_FONT_FAMILIES + PROPORTIONAL_TEST_FONT_FAMILIES

TRAIN_FONT_FAMILIES = MONOSPACED_TRAIN_FONT_FAMILIES + PROPORTIONAL_TRAIN_FONT_FAMILIES

TEST_FONT_FAMILIES = MONOSPACED_TEST_FONT_FAMILIES + PROPORTIONAL_TEST_FONT_FAMILIES

FONT_FAMILIES = TRAIN_FONT_FAMILIES + TEST_FONT_FAMILIES

FONT_SIZES = [9, 11, 12, 14, 16, 20]

FONT_FACES = [
    FontFace(False, False),
    FontFace(True, False),
    FontFace(False, True),
    FontFace(True, True),
]

BASIC_FONT_FAMILY = "Lucida Grande"
BASIC_FONT_SIZE = 14
BASIC_FONT_FACE = FontFace(False, False)
BASIC_FONT = Font(BASIC_FONT_FAMILY, BASIC_FONT_SIZE, BASIC_FONT_FACE)

assert BASIC_FONT_FAMILY in TRAIN_FONT_FAMILIES, "'{0}' should be in {1}".format(BASIC_FONT_FAMILY, TRAIN_FONT_FAMILIES)
assert BASIC_FONT_SIZE in FONT_SIZES, "'{0}' should be in {1}".format(BASIC_FONT_SIZE, FONT_SIZES)
assert BASIC_FONT_FACE in FONT_FACES, "'{0}' should be in {1}".format(BASIC_FONT_FACE, FONT_FACES)

# Functions

def is_monospaced(font):
    if isinstance(font, str):
        if font in MONOSPACED_FONT_FAMILIES:
            return True
        elif font in PROPORTIONAL_FONT_FAMILIES:
            return False
        else:
            raise Exception("Unknown font face: {0}.".format(font))
    elif isinstance(font, Font):
        return is_monospaced(font.family)
    else:
        raise TypeError("Only Font and str are allowed.")
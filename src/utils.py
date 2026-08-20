import pydicom

def load_dicom(filepath):
    """Load a DICOM file and return its pixel array."""
    ds = pydicom.dcmread(filepath)
    return ds.pixel_array
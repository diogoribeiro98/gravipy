import numpy as np

#Helper function to convert from East-North-Up to Earth Centered Earth Fixed Coordinates
def get_ENU_to_ECEF_transformation(lat,lon):
    """Returns the transformation matrix from East-North-Up (ENU)
        coordinates to Earth Centred Earth Fixed (ECEF) coordinates.
        To apply the transformation to a vector b = [E,N,U] use:

        matrix = get_ENU_to_ECEF_transformation(lat,lon)
        btransform = np.dot(mat,b),


    Args:
        lat (float): Latitude in degrees  
        lon (float): Longitude in degrees

    Returns:
        matrix : 3x3 transformation matrix
    """
    #Transform angles do radians
    phi = np.radians(lat)
    lam = np.radians(lon)

    neu_to_ecef = np.array([
    [-np.sin(lam), - np.cos(lam) * np.sin(phi),   np.cos(lam) * np.cos(phi)],
    [ np.cos(lam), - np.sin(lam) * np.sin(phi),   np.sin(lam) * np.cos(phi)],
    [         0.0,                 np.cos(phi),                 np.sin(phi)]
    ])

    return neu_to_ecef


def get_uvw_matrix(HA, DEC):
    """
    Returns transformation matrix from Local XYZ Earth coordinates to the UVW baseline space.

    Args:
        HA  (float): Hour angle of poiting direction (in degrees)
        DEC (float): Declination of poiting direction (in degrees)

    Returns:
        matrix : 3x3 transformation matrix 
    """

    HA_rad  = np.radians(HA)
    dec_rad = np.radians(DEC)

    transformation_matrix = np.array([
    [                 np.sin(HA_rad),                  np.cos(HA_rad),             0.0],
    [-np.sin(dec_rad)*np.cos(HA_rad),  np.sin(dec_rad)*np.sin(HA_rad), np.cos(dec_rad)],
    [ np.cos(dec_rad)*np.cos(HA_rad), -np.cos(dec_rad)*np.sin(HA_rad), np.sin(dec_rad)]
    ])

    return transformation_matrix

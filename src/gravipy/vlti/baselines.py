import numpy as np
import astropy.coordinates as coord
import astropy.units as u

from astropy.time import Time
from .reference_frames import get_ENU_to_ECEF_transformation, get_uvw_matrix

class vlti():
    def __init__(self, 
                 UT1=[ -9.925,-20.335, 0.0],
                 UT2=[ 14.887, 30.502, 0.0],
                 UT3=[ 44.915, 66.183, 0.0],
                 UT4=[103.306, 43.999, 0.0],
                 ):
        
        #Paranal observatory coordinates and ECEF transformation matrix
        self.longitude = -70.40479659
        self.latitude  = -24.62794830
        self.altitude  =  2635.0

        self.coordinates = coord.EarthLocation.from_geodetic(lat=self.latitude, lon=self.longitude, height=self.altitude)

        self.ECEF_transformation_matrix = get_ENU_to_ECEF_transformation(lat=self.coordinates.lat.deg, lon=0.0)

        self.telescopes = {
            'UT1': np.array(UT1),
            'UT2': np.array(UT2),
            'UT3': np.array(UT3),
            'UT4': np.array(UT4)
        }

        self.baselines = {
            'UT12': self.telescopes['UT1']-self.telescopes['UT2'],
            'UT13': self.telescopes['UT1']-self.telescopes['UT3'],
            'UT14': self.telescopes['UT1']-self.telescopes['UT4'],
            'UT23': self.telescopes['UT2']-self.telescopes['UT3'],
            'UT24': self.telescopes['UT2']-self.telescopes['UT4'],
            'UT34': self.telescopes['UT3']-self.telescopes['UT4'],
        }

        self.local_ecef_baselines = {
            'UT12': np.dot(self.ECEF_transformation_matrix,self.baselines['UT12']),
            'UT13': np.dot(self.ECEF_transformation_matrix,self.baselines['UT13']),
            'UT14': np.dot(self.ECEF_transformation_matrix,self.baselines['UT14']),
            'UT23': np.dot(self.ECEF_transformation_matrix,self.baselines['UT23']),
            'UT24': np.dot(self.ECEF_transformation_matrix,self.baselines['UT24']),
            'UT34': np.dot(self.ECEF_transformation_matrix,self.baselines['UT34']),
        }

    def get_uv_coordinates(self, t, RA = '17h45m40.04s', DEC = '-29d0m26.95s'):

        #Observation time    
        time = Time(t, format='isot', scale='utc')

        #Object being observed
        pointing = coord.SkyCoord(RA, DEC, frame='icrs', unit=(u.hourangle, u.deg))

        #Local coordinate poiting
        altaz = coord.AltAz(location=self.coordinates, obstime=time)
        LST = time.sidereal_time('mean', longitude=self.coordinates.lon.deg)  # Local Sidereal Time

        #Calculate Hour angle (adding the longitude of paranal seems to solve the aspro discrepency but this is very weird)
        HA = LST - pointing.ra 

        m = get_uvw_matrix(HA=-HA, DEC=pointing.dec)

        uv_coordinates = dict(self.local_ecef_baselines)

        for name in self.local_ecef_baselines:
            uv_coordinates[name] = np.dot(m,self.local_ecef_baselines[name])[:2]

        return uv_coordinates

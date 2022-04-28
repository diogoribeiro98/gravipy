import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy import wcs
from scipy.optimize import newton
from .star_orbits import star_pms, star_orbits

class GCorbits():
    def __init__(self, verbose=True):
        self.star_orbits = {}
        self.orbit_stars = []
        for s in star_orbits:
            self.star_orbits[s['name']] = s
            self.orbit_stars.append(s['name'])
        self.star_pms = {}
        self.pm_stars = []
        for s in star_pms:
            self.star_pms[s['name']] = s
            self.pm_stars.append(s['name'])
        if verbose:
            print('Stars with orbits:')
            print(self.orbit_stars)
            print('\nStars with proper motions:')
            print(self.pm_stars)

    def pos_orbit(self, star, t, rall=False):
        M0 = 4.40
        R0 = 8.34
        m_unit = M0*1e6*u.solMass
        a_unit = u.arcsec
        t_unit = u.yr

        l_unit = a_unit.to(u.rad)*(R0*u.kpc)
        v_unit = l_unit/t_unit

        G = float(c.G.cgs * m_unit*t_unit**2/l_unit**3)

        mu = G
        a = star['a']/1000
        n = self.mean_motion(mu, a)
        M = self.mod2pi(n*(t-star['T']))
        f = self.true_anomaly(star['e'], M)

        r = a*(1-star['e']*star['e'])/(1+star['e']*np.cos(f))
        # v = np.sqrt(mu/a/(1-star['e']**2))

        cO = np.cos(star['CapitalOmega'])
        sO = np.sin(star['CapitalOmega'])
        co = np.cos(star['Omega'])
        so = np.sin(star['Omega'])
        cf = np.cos(f)
        sf = np.sin(f)
        ci = np.cos(star['i'])
        si = np.sin(star['i'])

        x = r*(sO*(co*cf-so*sf)+cO*(so*cf+co*sf)*ci)
        y = r*(cO*(co*cf-so*sf)-sO*(so*cf+co*sf)*ci)
        z = r*(so*cf+co*sf)*si

        # vx = v*((star['e']+cf)*(ci*co*cO-sO*so)-sf*(co*sO+ci*so*cO))
        # vy = v*((star['e']+cf)*(-ci*co*sO-cO*so)-sf*(co*cO-ci*so*sO))
        # vz = v*((star['e']+cf)*co*si-sf*si*so)
        if rall:
            return np.array([x, y, z])
        else:
            return np.array([x, y])

    def pos_pm(self, star, t):
        vx = star['vx']/1000
        vy = star['vy']/1000
        x = star['x']/1000
        y = star['y']/1000

        x = x+vx*(t-star['T'])
        y = y+vy*(t-star['T'])
        if star['ax'] != 0:
            ax = star['ax']/1000
            x += ax/2*(t-star['T'])**2
            vx += ax*(t-star['T'])
        if star['ay'] != 0:
            ay = star['ay']/1000
            y += ay/2*(t-star['T'])**2
            vy += ay*(t-star['T'])
        return np.array([-x, y])

    def true_anomaly(self, e, M):
        E = self.eccentric_anomaly(e, M)
        if e > 1:
            return 2*np.arctan(np.sqrt((1+e)/(e-1))*np.tanh(E/2))
        else:
            return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

    def mean_motion(self, mu, a):
        return np.sign(a) * np.sqrt(np.fabs(mu/a**3))

    def mod2pi(self, x):
        return (x+np.pi) % (2*np.pi) - np.pi

    def eccentric_anomaly(self, e, M, *args, **kwargs):
        if e < 1:
            f = lambda E: E - e*np.sin(E) - M
            fp = lambda E: 1 - e*np.cos(E)
            E0 = M if e < 0.8 else np.sign(M)*np.pi
            E = self.mod2pi(newton(f, E0, fp, *args, **kwargs))
        else:
            f = lambda E: E - e*np.sinh(E) - M
            fp = lambda E: 1 - e*np.cosh(E)
            E0 = np.sign(M) * np.log(2*np.fabs(M)/e+1.8)
            E = newton(f, E0, fp, *args, **kwargs)
        return E

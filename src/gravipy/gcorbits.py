import numpy as np
from astropy import units as u
from astropy import constants as c
from scipy.optimize import newton
from .star_orbits import star_pms, star_orbits
from datetime import datetime
import matplotlib.pyplot as plt

from .gravdata import convert_date


class GCorbits():
    def __init__(self, verbose=True):
        """
        Package to get positions of stars at a certain point in time
        Orbits and proper motions from Stefan
        Does not contain GR effects and orbits are not necessarily up to date

        Supress outputs with verbose=False

        Main functions:
        plot_orbits : plot the stars for a given time
        pos_orbit : get positions for stars with orbits
        pos_pm : get positions for stars with proper motions
        """
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
        """
        Calculates the position of a star with known orbits
        star: has to be in the list: orbit_stars
        time: the time of evaluation, in float format 20xx.xx
        rall: if true returns also z-position
        """
        star = self.star_orbits[star]

        M0 = 4.40
        R0 = 8.34
        m_unit = M0*1e6*u.solMass
        a_unit = u.arcsec
        t_unit = u.yr

        l_unit = a_unit.to(u.rad)*(R0*u.kpc)
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
        """
        Calculates the position of a star with proper motion
        star: has to be in the list: pm_stars
        time: the time of evaluation, in float format 20xx.xx
        """
        star = self.star_pms[star]
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

    def plot_orbits(self, off=[0,0], t=None, figsize=8, lim=100, long=False):
        """
        Plot the inner region around SgrA* at a given TIME
        t:    time of plot (if None then now)
        lim:  radius to which stars are plotted
        long: more information if True
        """
        if t is None:
            d = datetime.utcnow()
            t = d.strftime("%Y-%m-%dT%H:%M:%S")
        t = convert_date(t)[0]

        starpos = []
        for star in self.star_orbits:
            _s = self.star_orbits[star]
            x, y = self.pos_orbit(star, t)
            starpos.append([_s['name'], x*1000, y*1000, _s['type'], _s['Kmag']])
        for star in self.star_pms:
            _s = self.star_pms[star]
            x, y = self.pos_pm(star, t)
            starpos.append([_s['name'], x*1000, y*1000, _s['type'], _s['Kmag']])

        fig, ax = plt.subplots()
        fig.set_figheight(figsize)
        fig.set_figwidth(figsize)
        for s in starpos:
            n, x, y, ty, mag = s
            if np.any(np.abs(x-off[0]) > lim) or np.any(np.abs(y-off[1]) > lim):
                continue
            if long:
                if ty == 'e':
                    plt.scatter(x, y, c='C2', s=10)
                    plt.text(x-3, y, '%s m$_K$=%.1f' % (n, mag), fontsize=6,
                             color='C2')
                elif ty == 'l':
                    plt.scatter(x, y, c='C0', s=10)
                    plt.text(x-3, y, '%s m$_K$=%.1f' % (n, mag), fontsize=6,
                             color='C0')
                else:
                    plt.scatter(x, y, c='C1', s=10)
                    plt.text(x-3, y, '%s m$_K$=%.1f' % (n, mag), fontsize=6,
                             color='C1')
            else:
                if ty == 'e':
                    plt.scatter(x, y, c='C2', s=10)
                    plt.text(x-3, y, '%s' % (n), fontsize=6, color='C2')
                elif ty == 'l':
                    plt.scatter(x, y, c='C0', s=10)
                    plt.text(x-3, y, '%s' % (n), fontsize=6, color='C0')
                else:
                    plt.scatter(x, y, c='C1', s=10)
                    plt.text(x-3, y, '%s' % (n), fontsize=6, color='C1')

        plt.axis([lim*1.2+off[0], -lim*1.2+off[0],
                  -lim*1.2+off[1], lim*1.2+off[1]])
        plt.gca().set_aspect('equal', adjustable='box')

        fiberrad = 70
        circ = plt.Circle((off), radius=fiberrad, facecolor="None",
                          edgecolor='darkblue', linewidth=0.2)
        ax.add_artist(circ)
        plt.text(0+off[0], -78+off[1], 'GRAVITY Fiber FoV', fontsize='6', color='darkblue',
                 ha='center')
        if np.any(np.abs(off[0]) > lim) or np.any(np.abs(off[1]) > lim):
            pass
        else:
            plt.scatter(0, 0, color='k', s=20, zorder=100)
            plt.text(-4, -8, 'Sgr A*', fontsize='8')

        if off != [0,0]:
            plt.scatter(*off, color='k', marker='X', s=20, zorder=100)
            plt.text(-4+off[0], -8+off[1], 'Pointing*', fontsize='8')

        plt.text(-80+off[0], 100+off[1], 'late type', fontsize=6, color='C0')
        plt.text(-80+off[0], 92+off[1], 'early type', fontsize=6, color='C2')
        plt.text(-80+off[0], 84+off[1], 'unknown', fontsize=6, color='C1')

        plt.xlabel('dRa [mas]')
        plt.ylabel('dDec [mas]')
        plt.show()

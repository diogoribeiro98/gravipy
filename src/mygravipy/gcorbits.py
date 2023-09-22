import numpy as np
from astropy import units as u
from astropy import constants as c
from scipy.optimize import newton
from .star_orbits import star_pms, star_orbits
from datetime import datetime
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
import logging
import glob
import re
from .gravdata import convert_date, log_level_mapping, fiber_coupling


class GCorbits():
    def __init__(self, t=None, loglevel='INFO'):
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
        log_level = log_level_mapping.get(loglevel, logging.INFO)
        gcorb_logger = logging.getLogger(__name__)
        gcorb_logger.setLevel(log_level)
        if not gcorb_logger.hasHandlers():
            ch = logging.StreamHandler()
            # ch.setLevel(log_level)
            formatter = logging.Formatter('%(levelname)s: %(name)s - %(message)s')
            ch.setFormatter(formatter)
            gcorb_logger.addHandler(ch)
        self.gcorb_logger = gcorb_logger

        if t is None:
            d = datetime.utcnow()
            t = d.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            t = convert_date(t)[0]
        except ValueError:
            raise ValueError('t has to be given as YYYY-MM-DDTHH:MM:SS')
        self.t = t

        self.star_poly = {}
        self.poly_stars = []
        self.star_orbits = {}
        self.orbit_stars = []
        self.star_pms = {}
        self.pm_stars = []

        _s = resource_filename('mygravipy', f'Datafiles/s*.dat')
        dfiles = sorted(glob.glob(_s))

        for d in dfiles:
            _d = d[-8:-4]
            snum = int(_d[_d.find('/s')+2:])
            _data = []
            with open(d, 'r') as file:
                for line in file:
                    # Process each line
                    l = line.strip()
                    l = l.replace('\t', ' ')
                    if l == '; position data date RA delta RA DEC delta DEC':
                        break
                    _data.append(l)

            # Check for polynomial
            ra_s = None
            de_s = None
            for _s in _data:
                if 'polyFitResultRA' in _s:
                    ra_s = _s
                elif 'polyFitResultDec' in _s:
                    de_s = _s
            if ra_s is not None and de_s is not None:
                gcorb_logger.debug(f'Polynomial found for S{snum}')
                
                ra = [float(m) for m in re.findall(r'-?\d+\.\d+', ra_s)]
                de = [float(m) for m in re.findall(r'-?\d+\.\d+', de_s)]
                tref = ra[0]
                ra = ra[1:]
                de = de[1:]
                npol = (len(ra))//2
                
                s = {'name': f'S{snum}',
                     'type': '',
                     'Kmag': 20,
                     'ra': ra,
                     'de': de,
                     'tref': tref,
                     'npol': npol}
                self.star_poly[f'S{snum}'] = s
                self.poly_stars.append(f'S{snum}')
            
            # Check for orbit
            else:
                sdx = -1
                for _sdx, _s in enumerate(_data):
                    if _s == '; best fitting orbit paramters':
                        sdx = _sdx + 1
                        break
                if sdx == -1:
                    gcorb_logger.warning(f'No orbit or polynomial found for S{snum}')
                    continue          
                data_new = []
                for _s in _data[sdx:]:
                    data_new.append(_s[:_s.find(' ; ')])
                data_new = [
                    [float(m) for m in re.findall(r'-?\d+\.\d+', line)][0]
                     for line in data_new]
                data_new = np.array(data_new)
                
                if len(data_new) != 14:
                    gcorb_logger.debug(f'No orbit or polynomial found for S{snum}')
                    continue

                s = {'name': f'S{snum}',
                     'type': '',
                     'a': data_new[0]*1e3,
                     'e': data_new[1],
                     'P': data_new[2],
                     'T': data_new[3],
                     'i': data_new[4]/180*np.pi,
                     'CapitalOmega': data_new[5]/180*np.pi,
                     'Omega': data_new[6]/180*np.pi,
                     'Kmag': 20,
                     'type': ''}
                self.star_orbits[f'S{snum}'] = s
                self.orbit_stars.append(f'S{snum}')
                

        for s in star_orbits:
            if s['name'] in self.star_orbits:
                self.star_orbits[s['name']]['type'] = s['type']
                self.star_orbits[s['name']]['Kmag'] = s['Kmag']
            else:
                self.star_orbits[s['name']] = s
                self.orbit_stars.append(s['name'])
                gcorb_logger.debug(f'Added {s["name"]} from old orbits')

        for s in star_pms:
            if s['name'] in self.star_orbits:
                self.star_orbits[s['name']]['type'] = s['type']
                self.star_orbits[s['name']]['Kmag'] = s['Kmag']
            elif s['name'] in self.star_poly:
                self.star_poly[s['name']]['type'] = s['type']
                self.star_poly[s['name']]['Kmag'] = s['Kmag']
            else:
                self.star_pms[s['name']] = s
                self.pm_stars.append(s['name'])
                gcorb_logger.debug(f'Added {s["name"]} from old pm stars')

        gcorb_logger.info(f'Evaluating for {t:.4f}')
        gcorb_logger.debug('Stars with orbits:')
        gcorb_logger.debug(self.orbit_stars)
        gcorb_logger.debug('')
        gcorb_logger.debug('Stars with polynomias:')
        gcorb_logger.debug(self.poly_stars)
        gcorb_logger.debug('')
        gcorb_logger.debug('Stars with proper motions:')
        gcorb_logger.debug(self.pm_stars)


        # calculate starpos
        starpos = []
        starpos.append(['SGRA', 0, 0, '', 15.7])
        for star in self.star_orbits:
            _s = self.star_orbits[star]
            x, y = self.pos_orbit(star)
            starpos.append([_s['name'], x*1000, y*1000,
                            _s['type'], _s['Kmag']])
        for star in self.star_poly:
            _s = self.star_poly[star]
            x, y = self.pos_poly(star)
            starpos.append([_s['name'], x*1000, y*1000, _s['type'], _s['Kmag']])
        for star in self.star_pms:
            _s = self.star_pms[star]
            x, y = self.pos_pm(star)
            starpos.append([_s['name'], x*1000, y*1000, _s['type'], _s['Kmag']])
        self.starpos = starpos


    def star_pos(self, star):
        try:
            return self.pos_orbit(star)
        except KeyError:
            try:
                return self.pos_poly(star)
            except KeyError:
                return self.pos_pm(star)

    def star_kmag(self, star):
        try:
            return self.star_orbits[star]['Kmag']
        except KeyError:
            return self.star_pms[star]['Kmag']

    def pos_poly(self, star):
        t = self.t
        star = self.star_poly[star]

        res = [0, 0]
        for p in range(star['npol']):
            res[0] += star['ra'][p*2]*(t - star['tref'])**p
            res[1] += star['de'][p*2]*(t - star['tref'])**p
        
        return np.array(res)


    def pos_orbit(self, star, rall=False):
        """
        Calculates the position of a star with known orbits
        star: has to be in the list: orbit_stars
        time: the time of evaluation, in float format 20xx.xx
        rall: if true returns also z-position
        """
        t = self.t
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

        if rall:
            return np.array([x, y, z])
        else:
            return np.array([x, y])

    def pos_pm(self, star):
        """
        Calculates the position of a star with proper motion
        star: has to be in the list: pm_stars
        """
        t = self.t
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

    def find_stars(self, x, y, fiberrad=70, plot=False, plotlim=400):
        """
        find all stars whcih are within one fiberad from x, y
        returns a list of stars
        if plot is True, plots the stars in the inner region
        plotlim: radius of the plot
        """
        self.gcorb_logger.info(f'Finding stars within {fiberrad} mas from {x}, {y}')
        starpos = self.starpos
        stars = []
        for s in starpos:
            n, sx, sy, _, mag = s
            dist = np.sqrt((sx-x)**2+(sy-y)**2)
            if dist < fiberrad:
                dmag = -2.5*np.log10(fiber_coupling(dist))
                stars.append([n, sx-x, sy-y, dist, mag, mag + dmag])
                self.gcorb_logger.info(f'{n} at a distance of [{sx-x:.2f} {sy-y:.2f}]')

        if plot:
            fig, ax = plt.subplots()
            for s in starpos:
                n, sx, sy, _, _ = s
                if np.any(np.abs(sx) > plotlim) or np.any(np.abs(sy) > plotlim):
                    continue
                color = 'grey'
                # check if n in stars[:,0]
                for s in stars:
                    if n == s[0]:
                        color = 'C0'
                plt.scatter(sx, sy, c=color, s=7)
                plt.text(sx-3, sy, '%s' % (n), fontsize=5, color=color)
            plt.axis([plotlim*1.2, -plotlim*1.2,
                      -plotlim*1.2, plotlim*1.2])
            circ = plt.Circle([x, y], radius=fiberrad, facecolor="None",
                            edgecolor='C0', linewidth=0.2)
            ax.add_artist(circ)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('dRa [mas]')
            plt.ylabel('dDec [mas]')
            plt.show()
        return stars, starpos



    def plot_orbits(self, off=[0, 0], t=None, figsize=8, lim=100, long=False):
        """
        Plot the inner region around SgrA* at a given TIME
        lim:  radius to which stars are plotted
        long: more information if True
        """
        starpos = self.starpos
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

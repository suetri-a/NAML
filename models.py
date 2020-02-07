import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.integrate import solve_ivp, cumtrapz
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import linregress
from skmisc.loess import loess
from abc import ABC, abstractmethod


class NonArrheniusBase(ABC):
    
    def __init__(self, oil_type, experiment):
        # Read in oil 
        self.oil_type, self.experiment = oil_type, experiment
        
        expdirname = 'datasets/'+ oil_type + '/' + experiment
        hr_names = [name for name in os.listdir(expdirname) if os.path.isdir(os.path.join(expdirname, name)) 
                    and name[-5:].lower()=='c_min']
        self.heating_rates = [hr[:-5] for hr in hr_names]
        
        data_frames = [pd.read_excel(os.path.join(expdirname, hr+'.xls')) for hr in hr_names]

        O2convs, Temps, dXdt = [], [], []
        INTERPNUM = 100
        
        for df in data_frames:
            # Read in data
            Time = df.Time.values
            O2 = df.O2.values
            Temp = df.Temperature.values

            if oil_type == 'synthetic':
                O2_conversion = O2
                O2_conversion /= O2_conversion[-1]
                dO2_conversion = np.gradient(O2_conversion)

            else:
                ind100C = np.amin(np.asarray(Temp > 100).nonzero())
                ind200C = np.amin(np.asarray(Temp > 180).nonzero())
                inds750 = np.asarray(Temp > 745).nonzero()[0]
                ind750C1 = inds750[np.round(0.75*inds750.shape[0]).astype(int)]
                ind750C2 = inds750[np.round(0.9*inds750.shape[0]).astype(int)]

                # Gather datapoints and perform linear regression correction
                correction_times = np.concatenate([Time[ind100C:ind200C+1], Time[ind750C1:ind750C2+1]])
                correction_O2s = np.concatenate([O2[ind100C:ind200C+1], O2[ind750C1:ind750C2+1]])
                slope, intercept, _, _, _ = linregress(correction_times, correction_O2s)
                O2_baseline = slope*Time + intercept
                
                # Calculate %O2 consumption and conversion
                O2_consumption = np.maximum((O2_baseline - O2) / O2_baseline[0], 0)
                O2_consumption[:ind200C] = 0 # zero out non-reactive regions
                O2_consumption[ind750C1:] = 0
                O2_conversion = cumtrapz(O2_consumption, x=Time, initial=0)
                dO2_conversion = 60*O2_consumption / O2_conversion[-1]
                O2_conversion /= O2_conversion[-1]

            plt.plot(Temp, np.maximum(O2_conversion,0))
            
            # Downsample and append
            time_downsampled = np.linspace(Time.min(), Time.max(), num=INTERPNUM)
            Temps.append(np.interp(time_downsampled, Time, Temp))
            O2convs.append(np.interp(time_downsampled, Time, O2_conversion))
            dXdt.append(np.interp(time_downsampled, Time, dO2_conversion))

        
        # plt.legend([str(h)+' C/min' for h in self.heating_rates])
        plt.show()

        
        # # Append boundary conditions
        # # Left BC - set rate at 20C to be 0 for all conversions
        BC_TEMP = 20.0
        O2convs.append(np.linspace(0,1,num=INTERPNUM))
        Temps.append(BC_TEMP*np.ones((INTERPNUM)))
        dXdt.append(np.zeros((INTERPNUM)))
        
        # Top BC - set rate at conversion=1 to be 0
        O2convs.append(np.ones((INTERPNUM)))
        Temps.append(np.linspace(BC_TEMP, 750, num=INTERPNUM))
        dXdt.append(np.zeros((INTERPNUM)))

        # Bottom BC - set rate at conversion=0 to be 0
        O2convs.append(np.zeros((INTERPNUM)))
        Temps.append(np.linspace(BC_TEMP, 750, num=INTERPNUM))
        dXdt.append(np.zeros((INTERPNUM)))
                
        # Create numpy arrays and store to object
        self.O2convs = np.vstack(O2convs)
        self.Temps = np.vstack(Temps)
        self.dXdt = np.vstack(dXdt)

    
    def get_sim_func(self, hr, hr_times = None, max_temp = 750):
        '''
        Input:
            hr - heating rate scalar or array of heating rates and corresponding time points
                where hr[1,:] is the time values and hr[2,:] is the heating rate values. Units
                in C/min.
            hr_times - time points corresponding to heating schedule defined in hr. Units in min.
            max_temp - maximum temperature. Units in C.
        '''
        
        def f(t, y):
            
            if y[0] > max_temp:
                dT = 0.0
            else:
                dT = hr if hr_times is None else np.interp(t, hr_times, hr)
                
            if y[1]>=1:
                dX = 0
            else:
                dX = self.conversion_lookup(y[0], y[1])
            
            return np.array([dT, dX])
        
        return f
    
    
    @abstractmethod
    def conversion_lookup(self, T, O2conv):
        '''
        Lookup conversion rate based on temperature and conversion.
        
        Inputs:
            T - temperature or array-like of temperature values
            O2conv - O2 conversion or array-like of O2 conversion values
        
        Returns:
            dO2conv - O2 conversion rate values 
        
        '''
        pass
    
    
    def print_consumption_curves(self, overlay_pred = False):
        gradplt = plt.figure()
        convplt = plt.figure()
        gradax = gradplt.add_subplot(111)
        convax = convplt.add_subplot(111)
        
        for i in range(len(self.heating_rates)):
            gradax.plot(self.Temps[i,:], self.dXdt[i,:])
            convax.plot(self.Temps[i,:], self.O2convs[i,:])
        
        convax.set_title(self.oil_type + ' - ' + self.experiment)
        convax.set_xlabel('Temperature')
        convax.set_ylabel(r'$O_2$ conversion')
        convax.legend([str(h)+' C/min' for h in self.heating_rates])
        convplt.show()
        
        gradax.set_title(self.oil_type + ' - ' + self.experiment)
        gradax.set_xlabel('Temperature')
        gradax.set_ylabel(r'$\frac{d O_2}{dt}$ conversion')
        gradax.legend([str(h)+' C/min' for h in self.heating_rates])
        gradplt.show()
        
    
    def print_surf_plot(self):
        
        Tgrid, O2grid = np.mgrid[20:750:200j, 0:1:200j]
        dXdt = self.conversion_lookup(Tgrid, O2grid)
        plt.figure()
        plt.imshow(dXdt.T, extent=(20,750,0,1), aspect="auto", origin='lower', 
                    cmap=plt.get_cmap('hot')) #, vmin=0.0, vmax=0.25
        plt.plot(self.Temps.flatten(), self.O2convs.flatten(), 'w.', ms=1)
        plt.xlabel('Temperature (C)')
        plt.ylabel('Conversion (% consumption)')
        plt.title('Conversion rate surface')
        plt.colorbar()
        plt.show()


class NonArrheniusInterp(NonArrheniusBase):
    
    def __init__(self, oil_type, experiment):
        super().__init__(oil_type, experiment)

        # Build interpolation surface
        T_MAX = np.amax(self.Temps)
        def lookup_surface1(Tq, O2q):
            data = griddata((self.Temps.flatten()/T_MAX, self.O2convs.flatten()), 
                            self.dXdt.flatten(), (Tq/T_MAX, O2q), method='linear')
            return data

        self.interp_surface = lookup_surface1

        # Build extrapolation surface
        fine_T, fine_O2conv = np.mgrid[20:750:200j, 0:1:200j]
        fine_dO2 = self.interp_surface(fine_T, fine_O2conv)

        def lookup_surface2(Tq, O2q):
            data = griddata((fine_T[np.isfinite(fine_dO2)]/T_MAX, fine_O2conv[np.isfinite(fine_dO2)]), 
                            fine_dO2[np.isfinite(fine_dO2)], (Tq/T_MAX, O2q), method='nearest')
            return data
    
        self.extrap_surface = lookup_surface2

    
    def conversion_lookup(self, T, O2conv):

        dO2conv = self.interp_surface(T, O2conv)

        if np.isscalar(T):
            if np.isnan(dO2conv):
                dO2conv = self.extrap_surface(T, O2conv)
        else:
            dO2conv[np.isnan(dO2conv)] = self.extrap_surface(T[np.isnan(dO2conv)], O2conv[np.isnan(dO2conv)])

        return dO2conv


class NonArrheniusML(NonArrheniusBase):

    def __init__(self, oil_type, experiment):
        super().__init__(oil_type, experiment)

        self.tau = 0.12
        
        ##### Compute estimate of variance
        Temps_norm = self.Temps / np.amax(self.Temps)
        X = np.expand_dims(np.column_stack([Temps_norm.flatten(), self.O2convs.flatten(), np.ones_like(self.O2convs.flatten())]), 0) #1 x M x 3
        xq = np.squeeze(X)
        W = np.exp(-np.sum((np.transpose(X,axes=(0,2,1)) - np.expand_dims(xq,2))**2, 
                            axis=1, keepdims=True) / self.tau**2 / 2) # N x 1 x M
        XTW = np.transpose(X, axes=(0,2,1))*W # N x 3 x M

        L = np.squeeze(np.matmul(np.expand_dims(xq,1), np.linalg.solve(np.matmul(XTW, X), XTW)))
        Lbar = np.eye(L.shape[0]) - L
        
        self.delta1 = np.trace(np.matmul(Lbar.T, Lbar))
        yhat =  self.conversion_lookup(self.Temps, self.O2convs).flatten()
        epshat = self.dXdt.flatten() - yhat
        self.sigma = np.sqrt(np.sum(epshat**2) / self.delta1)

        
    def conversion_lookup(self, T, O2conv, tau = None):
        '''
        Short wrapper to retrieve reaction rate
        '''
        dX, _ = self.get_rate_and_var(T, O2conv, compute_var=False, tau = tau)
        return dX


    def get_rate_and_var(self, T, O2conv, compute_var = True, tau = None):
        '''
        Calculate convsersion rate and variance at each point in input arrays T and O2conv
        
        '''

        if tau is None:
            tau = self.tau

        T_norm = T / np.amax(self.Temps)
        Temps_norm = self.Temps / np.amax(self.Temps)

        sigmas = None

        # Non-vectorized operation for scalar input
        if np.isscalar(T):
            xq = np.expand_dims(np.array([T_norm, O2conv, 1.0]),0)
            X = np.column_stack([Temps_norm.flatten(), self.O2convs.flatten(), np.ones_like(self.O2convs.flatten())])
            y = self.dXdt.flatten()
            W = np.exp(-np.sum((X - xq)**2, axis=1, keepdims=True) / tau**2 / 2)
            XTW = np.transpose(X*W)

            L = np.dot(np.squeeze(xq), np.linalg.solve(np.matmul(XTW, X), XTW))
            dX = np.dot(L, y)

            if compute_var:
                sigmas = self.sigma*np.sqrt(np.sum(L**2))

        
        # Vectorized version for array inputs
        else:
            xq = np.column_stack([T_norm.flatten(), O2conv.flatten(), np.ones_like(O2conv.flatten())]) # N x 3
            X = np.expand_dims(np.column_stack([Temps_norm.flatten(), self.O2convs.flatten(), np.ones_like(self.O2convs.flatten())]), 0) #1 x M x 3
            y = self.dXdt.flatten() # M 
            W = np.exp(-np.sum((np.transpose(X,axes=(0,2,1)) - np.expand_dims(xq,2))**2, 
                                axis=1, keepdims=True) / tau**2 / 2) # N x 1 x M
            XTW = np.transpose(X, axes=(0,2,1))*W # N x 3 x M

            L = np.squeeze(np.matmul(np.expand_dims(xq,1), np.linalg.solve(np.matmul(XTW, X), XTW)))
            dX = np.matmul(L, y)

            dX = np.reshape(dX, T.shape)

            if compute_var:
                sigmas = self.sigma*np.sqrt(np.sum(L**2, axis=1))
                sigmas = np.reshape(sigmas, T.shape)
        
        return dX, sigmas


    def print_uncertainty_surf(self):
        '''
        Plot surface of variances across the estimation

        '''

        Tgrid, O2grid = np.mgrid[20:750:200j, 0:1:200j]
        _, sigmas = self.get_rate_and_var(Tgrid, O2grid)
        plt.figure()
        plt.imshow(np.log(sigmas.T), extent=(20,750,0,1), aspect="auto", origin='lower', 
                    cmap=plt.get_cmap('plasma'))
        plt.plot(self.Temps.flatten(), self.O2convs.flatten(), 'w.', ms=1)
        plt.xlabel('Temperature (C)')
        plt.ylabel('Conversion (% mol)')
        plt.title('Log-variance of rate estimate')
        plt.colorbar()
        plt.show()
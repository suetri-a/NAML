import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.integrate import solve_ivp, cumtrapz
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import linregress, norm, truncnorm
from abc import ABC, abstractmethod


def load_rto_data(data_path, clean_data = True):

    df = pd.read_excel(data_path + '.xls')
    # Read in data
    Time = df.Time.values
    O2 = df.O2.values
    if hasattr(df, 'Temperature'):
        Temp = df.Temperature.values
    else:
        Temp = df.Temp.values

    # Perform data cleaning if necessary
    if clean_data:
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
        O2_consumption = np.maximum(O2_baseline - O2, 0)
        O2_consumption[:ind200C] = 0 # zero out non-reactive regions
        O2_consumption[ind750C1:] = 0
        O2_conversion = cumtrapz(O2_consumption, x=Time, initial=0)
        O2_conversion /= O2_conversion[-1]
        dO2_conversion = np.gradient(O2_conversion, Time/60)

    else:
        ind80C = np.amin(np.asarray(Temp > 80).nonzero())
        O2_baseline = O2[-1]
        O2_consumption = np.maximum(O2_baseline - O2, 0)
        O2_consumption[:ind80C] = 0
        O2_conversion = cumtrapz(O2_consumption, x=Time, initial=0)
        O2_conversion /= O2_conversion[-1]
        dO2_conversion = np.gradient(O2_conversion, Time/60)

    return Time, dO2_conversion, O2_conversion, Temp


class NonArrheniusBase(ABC):
    
    def __init__(self, *args, **kwargs):
        
        # Set up data container 
        self.oil_type, self.experiment = kwargs['oil_type'], kwargs['experiment'] 

        kwargs.setdefault('heating_rates', None)
        kwargs.setdefault('clean_data', True)
        kwargs.setdefault('interpnum',200)

        expdirname = os.path.join('datasets', self.oil_type, self.experiment)
        hr_names = [name[:-4] for name in os.listdir(expdirname) if name[-4:].lower()=='.xls']
                
        if kwargs['heating_rates'] is None:
            self.heating_rates = hr_names
        else:
            self.heating_rates = [hr for hr in hr_names if hr in kwargs['heating_rates']]
        
        # # Remove heating rates to be ignored
        # if kwargs['ignore_heating_rates'] is not None:
        #     for hr in kwargs['ignore_heating_rates']:
        #         if hr in self.heating_rates:
        #             i = self.heating_rates.index(hr)
        #             del self.heating_rates[i]
        #             del hr_names[i]

        # Begin loading data
        Times, O2convs, Temps, dXdt = [], [], [], []
        INTERPNUM = kwargs['interpnum']
        
        for hr in self.heating_rates:
            # Read RTO data
            Time, dO2_conversion, O2_conversion, Temp = load_rto_data(os.path.join(expdirname, hr), 
                                                                        clean_data=kwargs['clean_data'])
            
            # Downsample and append
            time_downsampled = np.linspace(Time.min(), Time.max(), num=INTERPNUM)
            Times.append(time_downsampled)
            Temps.append(np.interp(time_downsampled, Time, Temp))
            O2convs.append(np.interp(time_downsampled, Time, O2_conversion))
            dXdt.append(np.interp(time_downsampled, Time, dO2_conversion))

        
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
                
        # Create numpy arrays and store to object
        self.Times = np.vstack(Times)
        self.O2convs = np.vstack(O2convs)
        self.Temps = np.vstack(Temps)
        self.dXdt = np.vstack(dXdt)
        

    def get_sim_func(self, heating, max_temp = 750):
        '''
        Input:
            hr - heating rate scalar or array of heating rates and corresponding time points
                where heating[0] is the time values and heating[1] is the temperature values. Units
                in C/min for linear heating ramp.
            max_temp - maximum temperature. Units in C.
        '''

        if np.isscalar(heating):
            def f(t, y):
                dT = heating if y[0] < max_temp else 0.0
                if y[1] < 1.0:
                    dX = self.conversion_lookup(y[0], y[1])
                else:
                    dX = 0.0
                return np.array([dT, dX])
        else:
            def f(t,y):
                T = np.interp(t, heating[0], heating[1])
                if y < 1.0:
                    dX = self.conversion_lookup(T, y)
                else:
                    dX = 0
                # print('At time {} conversion={} and temperature={} and dX={}'.format(str(t), str(y), str(T),str(dX)))
                return dX
        
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
    
    
    def print_consumption_curves(self, save_path = None):
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

        if isinstance(save_path, str):
            gradplt.savefig(save_path[:-4] + '_conversion' + save_path[-4:])
            convplt.savefig(save_path[:-4] + '_consumption' + save_path[-4:])

        gradplt.show()
        
    
    def print_surf_plot(self, save_path = None, vmin=None, vmax=None):
        
        Tgrid, O2grid = np.mgrid[20:750:200j, 0:1:200j]
        dXdt = self.conversion_lookup(Tgrid, O2grid)
        plt.figure()
        plt.imshow(dXdt.T, extent=(20,750,0,1), aspect="auto", origin='lower', 
                    cmap=plt.get_cmap('hot'), vmin=vmin, vmax=vmax)
        plt.plot(self.Temps.flatten(), self.O2convs.flatten(), 'w.', ms=1)
        plt.xlabel('Temperature (C)')
        plt.ylabel('Conversion (% consumption)')
        # plt.title('Conversion rate surface')
        plt.colorbar()

        if isinstance(save_path, str):
            plt.savefig(save_path)

        print('Minimum conversion rate: {}, maximum conversion rate: {}'.format(dXdt.min(), dXdt.max()))
        plt.show()


    def simulate_rto(self, y0, tspan, heating, max_temp=750):
        
        f = self.get_sim_func(heating, max_temp = max_temp)
        if not np.isscalar(heating):
            times = heating[0]
        else:
            times = None
        sol = solve_ivp(f, tspan, y0, t_eval=times)

        return sol.t, sol.y


    def print_rto_experiment(self, y0, tspan, heating, title=None, max_temp=750, save_path = None):
        '''
        Inputs:
            y0 - initial condition for simulation
            tspan - time span for simulation
            heating - either linear heating rate (scalar value) or list of form [Time, Temps]
            title - title for the experiment
            max_temp - maximum temperature
            save_path - path to save files excluding '.png' extension

        '''

        t, y = self.simulate_rto(y0, tspan, heating, max_temp = max_temp)

        if np.isscalar(heating):
            consumption = y[0,:]
            temperature = y[1,:]
        else:
            consumption = y
            temperature = np.interp(t, heating[0], heating[1])

        plt.figure()
        plt.plot(t, consumption)
        plt.xlabel('Time [min]')
        plt.ylabel('Consumption [% mol]')
        if title is not None:
            plt.title(title + ' - Consumption')
        
        if save_path is not None:
            plt.savefig(save_path+'_conversion.png')

        plt.figure()
        plt.plot(t, temperature)
        plt.xlabel('Time [min]')
        plt.ylabel('Temperature [C]')

        if title is not None:
            plt.title(title + ' - Temperature')
        if save_path is not None:
            plt.savefig(save_path+'_temperature.png')


    def print_overlay(self, gt_data, save_path=None, name = None):
        
        t, y = self.simulate_rto([0.0], [0,gt_data['Time']], [gt_data['Time'], gt_data['Temp']])

        plt.figure()
        plt.plot(t, np.interp(t, gt_data['Time'], gt_data['Temp']))
        plt.plot(t, y)
        plt.xlabel('Time [min]')
        plt.ylabel('Consumption [% mol]')
        if name is not None:
            plt.title('Overlay of Simulated and Experimental Data - '+name)
        plt.legend(['Experimental Data', 'Simulated'])
        if isinstance(save_path,str):
            plt.savefig(save_path + '_consumption.png')
        plt.show()

        plt.figure()
        plt.plot(t, np.interp(t, gt_data['Time'], gt_data['Temp']))
        plt.plot(t, y)
        plt.xlabel('Time [min]')
        plt.ylabel('Temperature [C]')
        if name is not None:
            plt.title('Temperature Profile for Overlay - '+name)
        if isinstance(save_path,str):
            plt.savefig(save_path + '_temperature.png')
        plt.show()

        
    def compute_sim_mse(self, gt_data):

        _, y = self.simulate_rto([0.0], [0,gt_data['Time']], [gt_data['Time'], gt_data['Temp']])
        y_gt = gt_data['O2conv']
        MSE = np.sum((y - y_gt)**2)
        return MSE


class NonArrheniusInterp(NonArrheniusBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kwargs.setdefault('constrained', False)

        self.tau = 0.12

        self.constrained = kwargs['constrained']
        
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
            if not np.isscalar(O2conv):
                O2conv = np.asscalar(O2conv)

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


        # Implement constrained model
        if self.constrained:
            
            if compute_var:
                alphas = - dX / sigmas
                phi_alpha = norm.pdf(alphas)
                Z = 1 - norm.cdf(alphas)
                sigmas = sigmas*np.sqrt(1 + alphas*phi_alpha/Z - (phi_alpha / Z)**2)

            dX = np.maximum(dX, 0) # clamp negative rates
        
        return dX, sigmas


    def print_uncertainty_surf(self, save_path=None,vmin=None,vmax=None):
        '''
        Plot surface of variances across the estimation

        '''

        Tgrid, O2grid = np.mgrid[20:750:200j, 0:1:200j]
        _, sigmas = self.get_rate_and_var(Tgrid, O2grid)
        plt.figure()
        plt.imshow(np.log(sigmas.T), extent=(20,750,0,1), aspect="auto", origin='lower', 
                    cmap=plt.get_cmap('plasma'),vmin=vmin,vmax=vmax)
        plt.plot(self.Temps.flatten(), self.O2convs.flatten(), 'w.', ms=1)
        plt.xlabel('Temperature (C)')
        plt.ylabel('Conversion (% mol)')
        # plt.title('Log-std. deviation of rate estimate')
        plt.colorbar()

        if isinstance(save_path, str):
            plt.savefig(save_path)
        
        print('Minimum log-std. dev.: {}, maximum log-std. dev.: {}'.format(np.log(sigmas.min()), np.log(sigmas.max())))
            
        plt.show()


    def print_O2_confint(self, t, T, O2conv, save_path = None):

        dX, sigmas = self.get_rate_and_var(T, O2conv)

        if self.constrained:
            lb = truncnorm.ppf(0.025, 0, np.inf, dX, sigmas)
            ub = truncnorm.ppf(0.975, 0, np.inf, dX, sigmas)
        else:
            lb = dX - 2*sigmas
            ub = dX + 2*sigmas
        
        plt.figure()
        plt.plot(t, dX)
        plt.fill_between(t, lb, ub)
        plt.xlabel('Time (min)')
        plt.ylabel('Consumption (% mol)')
        plt.title('Conversion Rate with Uncertainty Bounds')

        if isinstance(save_path, str):
            plt.savefig(save_path)
            
        plt.show()
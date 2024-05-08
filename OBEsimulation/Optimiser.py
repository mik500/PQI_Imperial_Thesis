import numpy as np
from scipy.interpolate import interp1d
#from scipy.interpolate import LinearNDInterpolator
#from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import interpn
#from scipy.interpolate import griddata
import timeit
import nlopt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
import os


class optimiser:
    """
    Class to optimise solver
    """
    def __init__(self, sol, method='RK4', _complex=False, verbose=False, live_plot=False, save_to_file=False):
        self.sol = sol
        self.method = method
        self._complex = _complex
        self.verbose = verbose
        self.live_plot = live_plot
        self.save_to_file = save_to_file
        self.adapt_method = 'Wolfe'
        # Default params for adams adaptive step size
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8
        self.down_factor = 0.1 # also used by Wolfe and Backtrack
        self.step_tol = 1e-3
        self.max_points = None
        # Default params for Wolfe adaptive step size
        self.c1 = 1e-9
        self.c2 = 0.9
        self.up_factor = 2
        # Params for live plotting
        self.max_plot_points = 250
        # Params for save to file
        self.directory = None
        self.ammendments = None

    @classmethod
    def plot_from_file(cls, filename, max_plot_points=250):
        ### Read file
        with np.load(filename) as data:
            Control_number = data['Control_number']
            storage_eff_hist = data['storage_eff_hist']
            total_eff_hist = data['total_eff_hist']
            grad_norms_hist = data['grad_norms_hist']
            step_sizes_hist = data['step_sizes_hist']
            count_hist = data['count_hist']
            Control_best = data['Control_best']

            ### From file name make tpoints
            start = '_m_'
            end = '_tbounds'
            m = int((filename[filename.find(start)+len(start):filename.rfind(end)]).replace('d', '.'))
            start = '_tbounds_'
            end = '_T_'
            tbounds = [float(tb.replace('d', '.')) for tb in filename[filename.find(start)+len(start):filename.rfind(end)][1:-1].split(' ')]
            t = np.linspace(tbounds[0], tbounds[1], m)
            step = m//max_plot_points

            ### Check if control complex
            if Control_best.dtype == np.complex:
                _complex = True
            else:
                _complex = False

            # create efficiency and counts plot:
            subplots = make_subplots(rows=1, cols=2, subplot_titles=("Efficiency", "Completed steps"))
            # create efficiency subplot
            fig_eff_count = go.FigureWidget(subplots)
            fig_eff_count.update_layout(height=600, width=900)
            storage_eff_trace = go.Scatter(x=list(range(1, len(storage_eff_hist)+1)), y=storage_eff_hist, mode='lines+markers', name='Storage Efficiency')
            total_eff_effs_trace = go.Scatter(x=list(range(1, len(storage_eff_hist)+1)), y=total_eff_hist, mode='lines+markers', name='Total Efficiency')
            fig_eff_count.add_traces([storage_eff_trace, total_eff_effs_trace], rows=1, cols=1)
            fig_eff_count.update_shapes(yref = 'y', rows=1, cols=1)
            # set yaxis min
            fig_eff_count.update_shapes(y0 = 0, rows=1, cols=1)

            # create counts subplot
            #self.fig_counts = go.FigureWidget()
            count_trace = go.Scatter(x=list(range(1, len(count_hist)+1)), y=count_hist, mode='lines+markers', name='Count')
            fig_eff_count.add_traces([count_trace], rows=1, cols=2)
            # set yaxis min
            fig_eff_count.update_shapes(y0 = 0, rows=1, cols=2)
            fig_eff_count.update_shapes(yref = 'y2', rows=1, cols=2)

            # create gradient norms, step_sizes np.abs(Control) and np.angle(Control) if complex fields plot
            # each Control will have its own subplot
            subplots = make_subplots(rows=int(np.ceil(Control_number/2)), cols=2)
            fig_grads = go.FigureWidget(subplots)
            fig_grads.update_layout(height=600, width=900, title_text="Gradient norms")
            fig_steps = go.FigureWidget(subplots)
            fig_steps.update_layout(height=600, width=900, title_text="Step sizes")
            fig_Controls = go.FigureWidget(subplots)
            fig_Controls.update_layout(height=600, width=900, title_text="Controls amplitude")
            if _complex:
                fig_Controls_phase = go.FigureWidget(subplots)
                fig_Controls_phase.update_layout(height=600, width=900, title_text="Controls phase")
            
            for Cno in range(0, Control_number):
                grad_traces = []
                step_traces = []
                control_traces = []
                control_phase_traces = []
                grad_traces.append(go.Scatter(x=list(range(1, len(count_hist)+1)), y=grad_norms_hist[:, Cno, 0], mode='lines+markers', name='Norm Grad Control: ' + str(Cno) + ', LCP'))
                grad_traces.append(go.Scatter(x=list(range(1, len(count_hist)+1)), y=grad_norms_hist[:, Cno, 1], mode='lines+markers', name='Norm Grad Control: ' + str(Cno) + ', RCP'))
                fig_grads.add_traces(grad_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
                # set yaxis min
                fig_grads.update_shapes(y0 = 0, rows=(Cno//2)+1, cols=(Cno%2)+1)

                step_traces.append(go.Scatter(x=list(range(1, len(count_hist)+1)), y=step_sizes_hist[:, Cno, 0], mode='lines+markers', name='Step Size Control: ' + str(Cno) + ', LCP'))
                step_traces.append(go.Scatter(x=list(range(1, len(count_hist)+1)), y=step_sizes_hist[:, Cno, 1], mode='lines+markers', name='Step Size Control: ' + str(Cno) + ', RCP'))
                fig_steps.add_traces(step_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
                
                control_traces.append(go.Scatter(x=t[::step], y=np.abs(Control_best[Cno, ::step, 0]), mode='lines+markers', name='Control: ' + str(Cno) + ', LCP'))
                control_traces.append(go.Scatter(x=t[::step], y=np.abs(Control_best[Cno, ::step, 1]), mode='lines+markers', name='Control: ' + str(Cno) + ', RCP'))
                fig_Controls.add_traces(control_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
                fig_Controls.update_shapes(y0 = 0, rows=(Cno//2)+1, cols=(Cno%2)+1)

                if _complex:
                    control_phase_traces.append(go.Scatter(x=t[::step], y=np.angle(Control_best[Cno, ::step, 0]), mode='lines+markers', name='Control: ' + str(Cno) + ', LCP'))
                    control_phase_traces.append(go.Scatter(x=t[::step], y=np.angle(Control_best[Cno, ::step, 1]), mode='lines+markers', name='Control: ' + str(Cno) + ', RCP'))
                    fig_Controls_phase.add_traces(control_phase_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
                    fig_Controls_phase.update_shapes(y0 = -1.05*np.pi, rows=(Cno//2)+1, cols=(Cno%2)+1)
                    fig_Controls_phase.update_shapes(y1 = 1.05*np.pi, rows=(Cno//2)+1, cols=(Cno%2)+1)
            
            if _complex:
                wid = widgets.VBox([fig_eff_count, fig_grads, fig_steps, fig_Controls, fig_Controls_phase])
            else:
                wid = widgets.VBox([fig_eff_count, fig_grads, fig_steps, fig_Controls])
                
            display(wid)


    def create_plots(self, Control_number):
        # create efficiency and counts plot:
        subplots = make_subplots(rows=1, cols=2, subplot_titles=("Efficiency", "Completed steps"))
        # create efficiency subplot
        self.fig_eff_count = go.FigureWidget(subplots)
        self.fig_eff_count.update_layout(height=600, width=900)
        storage_eff_trace = go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Storage Efficiency')
        total_eff_effs_trace = go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Total Efficiency')
        self.fig_eff_count.add_traces([storage_eff_trace, total_eff_effs_trace], rows=1, cols=1)
        self.fig_eff_count.update_shapes(yref = 'y', rows=1, cols=1)
        # set yaxis min
        self.fig_eff_count.update_shapes(y0 = 0, rows=1, cols=1)

        # create counts subplot
        #self.fig_counts = go.FigureWidget()
        count_trace = go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Count')
        self.fig_eff_count.add_traces([count_trace], rows=1, cols=2)
        # set yaxis min
        self.fig_eff_count.update_shapes(y0 = 0, rows=1, cols=2)
        self.fig_eff_count.update_shapes(yref = 'y2', rows=1, cols=2)

        # create gradient norms, step_sizes np.abs(Control) and np.angle(Control) if complex fields plot
        # each Control will have its own subplot
        subplots = make_subplots(rows=int(np.ceil(Control_number/2)), cols=2)
        self.fig_grads = go.FigureWidget(subplots)
        self.fig_grads.update_layout(height=600, width=900, title_text="Gradient norms")
        self.fig_steps = go.FigureWidget(subplots)
        self.fig_steps.update_layout(height=600, width=900, title_text="Step sizes")
        self.fig_Controls = go.FigureWidget(subplots)
        self.fig_Controls.update_layout(height=600, width=900, title_text="Controls amplitude")
        if self._complex:
            self.fig_Controls_phase = go.FigureWidget(subplots)
            self.fig_Controls_phase.update_layout(height=600, width=900, title_text="Controls phase")
        
        for Cno in range(0, Control_number):
            grad_traces = []
            step_traces = []
            control_traces = []
            control_phase_traces = []
            grad_traces.append(go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Norm Grad Control: ' + str(Cno) + ', LCP'))
            grad_traces.append(go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Norm Grad Control: ' + str(Cno) + ', RCP'))
            self.fig_grads.add_traces(grad_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
            # set yaxis min
            self.fig_grads.update_shapes(y0 = 0, rows=(Cno//2)+1, cols=(Cno%2)+1)

            step_traces.append(go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Step Size Control: ' + str(Cno) + ', LCP'))
            step_traces.append(go.Scatter(x=[0], y=[[]], mode='lines+markers', name='Step Size Control: ' + str(Cno) + ', RCP'))
            self.fig_steps.add_traces(step_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
            
            control_traces.append(go.Scatter(x=self.sol.tpoints, y=[np.zeros(self.max_plot_points)], mode='lines+markers', name='Control: ' + str(Cno) + ', LCP'))
            control_traces.append(go.Scatter(x=self.sol.tpoints, y=[np.zeros(self.max_plot_points)], mode='lines+markers', name='Control: ' + str(Cno) + ', RCP'))
            self.fig_Controls.add_traces(control_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
            self.fig_Controls.update_shapes(y0 = 0, rows=(Cno//2)+1, cols=(Cno%2)+1)

            if self._complex:
                control_phase_traces.append(go.Scatter(x=self.sol.tpoints, y=[np.zeros(self.max_plot_points)], mode='lines+markers', name='Control: ' + str(Cno) + ', LCP'))
                control_phase_traces.append(go.Scatter(x=self.sol.tpoints, y=[np.zeros(self.max_plot_points)], mode='lines+markers', name='Control: ' + str(Cno) + ', RCP'))
                self.fig_Controls_phase.add_traces(control_phase_traces, rows=(Cno//2)+1, cols=(Cno%2)+1)
                self.fig_Controls_phase.update_shapes(y0 = -1.05*np.pi, rows=(Cno//2)+1, cols=(Cno%2)+1)
                self.fig_Controls_phase.update_shapes(y1 = 1.05*np.pi, rows=(Cno//2)+1, cols=(Cno%2)+1)
        
        if self._complex:
            self.wid = widgets.VBox([self.fig_eff_count, self.fig_grads, self.fig_steps, self.fig_Controls, self.fig_Controls_phase])
        else:
            self.wid = widgets.VBox([self.fig_eff_count, self.fig_grads, self.fig_steps, self.fig_Controls])
            
        display(self.wid)
    
    def generate_filename(self, metadata):
        protocol = metadata["protocol"]
        config = metadata["config"]
        deltas = metadata["deltas"]
        OD = metadata["OD"]
        L = metadata["L"]
        n = metadata["n"]
        m = metadata["m"]
        tbounds = metadata["tbounds"]
        T = metadata["T"]
        vno = metadata["vno"]

        if self.ammendments:
            filename = f"\\\\?\\{self.directory}\\protocol={protocol}_config={config}_deltas={deltas}_OD={OD}_L={L}_n={n}_m={m}_tbounds={tbounds}_T={T}_vno={vno}_{self.ammendments}.npz"
        else:
            filename = f"\\\\?\\{self.directory}\\protocol={protocol}_config={config}_deltas={deltas}_OD={OD}_L={L}_n={n}_m={m}_tbounds={tbounds}_T={T}_vno={vno}.npz"
        
        # Create the directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        # Create an empty file if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, "w+"):
                pass
    
        return filename

    def control_to_plot(self, control):
        """
        Find important points of controls to plot
        """
        mag0 = np.abs(control[:, 0]/self.sol.gamma)
        mag1 = np.abs(control[:, 1]/self.sol.gamma)
        # diff0 = np.diff(mag0)
        # diff1 = np.diff(mag1)
        # # find max points number of largest changes
        # indices0 = np.argpartition(np.abs(diff0), -self.max_points)[-self.max_points:] + 1
        # indices1 = np.argpartition(np.abs(diff1), -self.max_points)[-self.max_points:] + 1
        # indices0 = np.concatenate((np.array([0]),np.sort(indices0)))
        # indices1 = np.concatenate((np.array([0]),np.sort(indices1)))
        # mag0 = mag0[indices0]
        # mag1 = mag1[indices1]
        # magt0 = self.sol.tpoints[indices0]
        # magt1 = self.sol.tpoints[indices1]
        step = self.sol.m//self.max_plot_points
        mag0 = mag0[::step]
        mag1 = mag1[::step]
        magt0 = self.sol.tpoints[::step]
        magt1 = self.sol.tpoints[::step]
        if self._complex:
            # repeat for phase
            phase0 = np.angle(control[:, 0])
            phase1 = np.angle(control[:, 1])
            # diff0 = np.diff(np.unwrap(phase0))
            # diff1 = np.diff(np.unwrap(phase1))
            # # find max points number of largest changes
            # indices0 = np.argpartition(np.abs(diff0), -self.max_points)[-self.max_points:] + 1
            # indices1 = np.argpartition(np.abs(diff1), -self.max_points)[-self.max_points:] + 1
            # indices0 = np.concatenate((np.array([0]),np.sort(indices0)))
            # indices1 = np.concatenate((np.array([0]),np.sort(indices1)))
            # phase0 = phase0[indices0]
            # phase1 = phase1[indices1]
            # phaset0 = self.sol.tpoints[indices0]
            # phaset1 = self.sol.tpoints[indices1]
            phase0 = phase0[::step]
            phase1 = phase1[::step]
            phaset0 = self.sol.tpoints[::step]
            phaset1 = self.sol.tpoints[::step]
        else:
            phase0 = []
            phase1 = []
            phaset0 = []
            phaset1 = []
        return np.array([mag0, mag1]), [magt0, magt1], [phase0, phase1], [phaset0, phaset1]

    def update_plots(self, Control_number, lists):
        [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_curr] = lists

        # Can we update by calling name of trace?!

        # efficiency plots
        self.fig_eff_count.data[0].x = list(range(1, len(storage_eff_hist)+1))
        self.fig_eff_count.data[0].y = storage_eff_hist
        self.fig_eff_count.data[1].x = list(range(1, len(storage_eff_hist)+1))
        self.fig_eff_count.data[1].y = total_eff_hist
        # update y axis max
        ymax = max([max(storage_eff_hist), max(total_eff_hist)]) # storage should always be larger for normal memory
        self.fig_eff_count.update_shapes(y1 = 1.05*ymax, rows=1, cols=1)

        # count plot
        self.fig_eff_count.data[2].x = list(range(1, len(storage_eff_hist)+1))
        self.fig_eff_count.data[2].y = count_hist
        self.fig_eff_count.update_shapes(y1 = 1.05*max(count_hist), rows=1, cols=2)

        # update gradient norms, step_sizes and Control fields plot
        # update ylimits
        ys_grad = np.array(grad_norms_hist)
        ys_steps = np.array(step_sizes_hist)

        for Cno in range(0, Control_number):
            self.fig_grads.data[2*Cno].x = list(range(1, len(storage_eff_hist)+1))
            self.fig_grads.data[2*Cno].y = ys_grad[:, Cno, 0]
            self.fig_grads.data[2*Cno + 1].x = list(range(1, len(storage_eff_hist)+1))
            self.fig_grads.data[2*Cno + 1].y = ys_grad[:, Cno, 1]
            self.fig_grads.update_shapes(y1 = 1.05*max(ys_grad[:, Cno, :].flatten()), rows=(Cno//2)+1, cols=(Cno%2)+1)

            self.fig_steps.data[2*Cno].x = list(range(1, len(storage_eff_hist)+1))
            self.fig_steps.data[2*Cno].y = ys_steps[:, Cno, 0]
            self.fig_steps.data[2*Cno + 1].x = list(range(1, len(storage_eff_hist)+1))
            self.fig_steps.data[2*Cno + 1].y = ys_steps[:, Cno, 1]
            self.fig_steps.update_shapes(y0 = 0.95*min(ys_steps[:, Cno, :].flatten()), rows=(Cno//2)+1, cols=(Cno%2)+1)
            self.fig_steps.update_shapes(y1 = 1.05*max(ys_steps[:, Cno, :].flatten()), rows=(Cno//2)+1, cols=(Cno%2)+1)

            mag, magt, phase, phaset = self.control_to_plot(Control_curr[Cno])
            self.fig_Controls.data[2*Cno].x = magt[0]
            self.fig_Controls.data[2*Cno].y = mag[0]
            self.fig_Controls.data[2*Cno + 1].x = magt[1]
            self.fig_Controls.data[2*Cno + 1].y = mag[1]
            self.fig_Controls.update_shapes(y1 = 1.05*max(mag.flatten()), rows=(Cno//2)+1, cols=(Cno%2)+1)

            if self._complex:
                self.fig_Controls_phase.data[2*Cno].x = phaset[0]
                self.fig_Controls_phase.data[2*Cno].y = phase[0]
                self.fig_Controls_phase.data[2*Cno + 1].x = phaset[1]
                self.fig_Controls_phase.data[2*Cno + 1].y = phase[1]
                        
    def grad(self, arrays, coarrays, Control, field = 1):
        if self.sol.protocol == 'EIT':
            P = arrays[0]
            S = arrays[1]
            coP = coarrays[0]
            coS = coarrays[1]
            if self._complex:
                return( -1j*np.trapz( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                       np.conj(coS),
                                       np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, P) )

                            - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                        coP,
                                        np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaQ, np.conj(S) ) )
                            , x=self.sol.zCheby, axis=1)
                    )
            else:
                return( +2*np.imag( np.trapz( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                       np.conj(coS),
                                       np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, P) )

                            - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                        coP,
                                        np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaQ, np.conj(S) ) )
                            , x=self.sol.zCheby, axis=1) )
                    )
        elif self.sol.protocol == 'Raman':
            E = arrays[0] 
            S = arrays[1]
            coE = coarrays[0]
            coS = coarrays[1]
            if self._complex:
                return( np.trapz(
                                    np.einsum('tzp, tzp -> tzp' , coE,
                                    np.einsum('zghjkvp, tzghjkv -> tzp' , np.einsum('zghjkvp, gh -> zghjkvp', -1j*self.sol.dsqrtQ, np.sqrt(self.sol.pop)),
                                            np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                                        np.einsum('jkqwp,tzghqwv -> tzghjkv', self.sol.OmegaQ, np.conj(S)), 
                                                        1/(1 - 1j*self.sol.DELTAS) ) ) )

                                    + np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS),
                                    np.einsum('jkqwp, tzghjkv -> tzghqwvp', -1j*self.sol.OmegaQ, 
                                    np.einsum('tzghjkv, gjv -> tzghjkv', 
                                            np.einsum('zghjkvp, tzp -> tzghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.sol.dsqrtQ, np.sqrt(self.sol.pop)), E), 1/(1 + 1j*self.sol.DELTAS) ) ) )
                                    
                                    - np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                            np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                            np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, Control), S ),
                                            1/(1+1j*self.sol.DELTAS) ) ) )

                                    - np.einsum('tzghqwv, tzghqwvp -> tzp' , coS,
                                    np.einsum('tjkqw, tzghjkvp -> tzghqwvp', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, Control),
                                    np.einsum('tzghjkvp, gjv -> tzghjkvp', 
                                            np.einsum('jkqwp, tzghqwv -> tzghjkvp', self.sol.OmegaQ, np.conj(S)), 1/(1 - 1j*self.sol.DELTAS) )) )
                                    
                                    # - np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                    #         np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                    #         np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, Control), S ),
                                    #         1/(1+1j*self.sol.DELTAS) ) ) )

                                    # - np.einsum('tzghqwv, tzghqwvp -> tzp', coS, 
                                    #         np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                    #         np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, np.conj(Control)), np.conj(S) ),
                                    #         1/(1-1j*self.sol.DELTAS) ) ) )
                                    
                                    # -2*np.real(np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                    #         np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                    #         np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, Control), S ),
                                    #         1/(1+1j*self.sol.DELTAS) ) ) ) )

                            , x=self.sol.zCheby, axis=1)
                    )
            else:
                return( np.trapz( +2*np.real( np.einsum('tzp, tzp -> tzp' , coE,
                                    np.einsum('zghjkvp, tzghjkv -> tzp' , np.einsum('zghjkvp, gh -> zghjkvp', -1j*self.sol.dsqrtQ, np.sqrt(self.sol.pop)),
                                            np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                                        np.einsum('jkqwp,tzghqwv -> tzghjkv', self.sol.OmegaQ, np.conj(S)), 
                                                        1/(1 - 1j*self.sol.DELTAS) ) ) )

                                    + np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS),
                                    np.einsum('jkqwp, tzghjkv -> tzghqwvp', -1j*self.sol.OmegaQ, 
                                    np.einsum('tzghjkv, gjv -> tzghjkv', 
                                            np.einsum('zghjkvp, tzp -> tzghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.sol.dsqrtQ, np.sqrt(self.sol.pop)), E), 1/(1 + 1j*self.sol.DELTAS) ) ) )
                                            )

                                    - 4*np.real(np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                            np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                            np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, np.abs(Control)), S ),
                                            1/(1+1j*self.sol.DELTAS) ) ) ))
                            , x=self.sol.zCheby, axis=1)
                    )
        elif self.sol.protocol == 'FLAME':
            P = arrays[0]
            S = arrays[1]
            coP = coarrays[0]
            coS = coarrays[1]

            if self._complex:
                term = -1j*(
                                    np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                       np.conj(coS),
                                       np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, P) )

                            - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                        coP,
                                        np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaQ, np.conj(S) ) )
                )

                self.term_trans = interpn((self.sol.tpoints, self.sol.zCheby), term, np.array([self.tdash.T, (self.sol.cNU*(self.yline[:, None] - self.tdash)/2).T]).T, method='linear', bounds_error=False, fill_value=0.0)
                grad_y = (1/(self.sol.tbounds[-1]*self.sol.gamma))*np.trapz(self.term_trans, x=self.sol.tpoints, axis=1)
            else:
                term = ( +2*np.imag( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                       np.conj(coS),
                                       np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, P) )

                            - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                        coP,
                                        np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaQ, np.conj(S) ) ))
                            )
                self.term_trans = interpn((self.sol.tpoints, self.sol.zCheby), term, np.array([self.tdash.T, (self.sol.cNU*(self.yline[:, None] - self.tdash)/2).T]).T, method='linear', bounds_error=False, fill_value=0.0)
                grad_y = (1/(self.sol.tbounds[-1]*self.sol.gamma))*np.trapz(self.term_trans, x=self.sol.tpoints, axis=1)

            return np.transpose( np.array([np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 0]) ,
                     np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 1])]), (1, 0) )
        
        elif self.sol.protocol == 'ORCA':

            E = arrays[0]
            S = arrays[1]
            coE = coarrays[0]
            coS = coarrays[1]

            Control_func = self.sol.counter_prop(Control*self.sol.gamma)

            if self._complex:
                term = (
                                    np.einsum('tzp, tzp -> tzp' , coE,
                                    np.einsum('zghjkvp, tzghjkv -> tzp' , np.einsum('zghjkvp, gh -> zghjkvp', -1j*self.sol.dsqrtQ, np.sqrt(self.sol.pop)),
                                            np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                                        np.einsum('jkqwp,tzghqwv -> tzghjkv', self.sol.OmegaQ, np.conj(S)), 
                                                        1/(1 - 1j*self.sol.DELTAS) ) ) )

                                    + np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS),
                                    np.einsum('jkqwp, tzghjkv -> tzghqwvp', -1j*self.sol.OmegaQ, 
                                    np.einsum('tzghjkv, gjv -> tzghjkv', 
                                            np.einsum('zghjkvp, tzp -> tzghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.sol.dsqrtQ, np.sqrt(self.sol.pop)), E), 1/(1 + 1j*self.sol.DELTAS) ) ) )
                                    
                                    
                                    - np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                            np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                            np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, ztp -> tzjkqw', self.sol.OmegaQ, Control_func(self.sol.t_grid, self.sol.z_grid)), S ),
                                            1/(1+1j*self.sol.DELTAS) ) ) )

                                    - np.einsum('tzghqwv, tzghqwvp -> tzp' , coS,
                                    np.einsum('tjkqw, tzghjkvp -> tzghqwvp', np.einsum('jkqwp, ztp -> tzjkqw', self.sol.OmegaQ, Control_func(self.sol.t_grid, self.sol.z_grid)),
                                    np.einsum('tzghjkvp, gjv -> tzghjkvp', 
                                            np.einsum('jkqwp, tzghqwv -> tzghjkvp', self.sol.OmegaQ, np.conj(S)), 1/(1 - 1j*self.sol.DELTAS) )) )
                                    
                                    # - np.real(np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                    #     np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                    #     np.einsum('tzjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, ztp -> tzjkqw', self.sol.OmegaQ, Control_func(self.sol.t_grid, self.sol.z_grid)), S ),
                                    #     1/(1+1j*self.sol.DELTAS) ) ) ))
                )

                self.term_trans = interpn((self.sol.tpoints, self.sol.zCheby), term, np.array([self.tdash.T, (self.sol.cNU*(self.yline[:, None] - self.tdash)/2).T]).T, method='linear', bounds_error=False, fill_value=0.0)
                grad_y = (1/(self.sol.tbounds[-1]*self.sol.gamma))*np.trapz(self.term_trans, x=self.sol.tpoints, axis=1)
            else:
                term = ( +2*np.real( np.einsum('tzp, tzp -> tzp' , coE,
                                    np.einsum('zghjkvp, tzghjkv -> tzp' , np.einsum('zghjkvp, gh -> zghjkvp', -1j*self.sol.dsqrtQ, np.sqrt(self.sol.pop)),
                                            np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                                        np.einsum('jkqwp,tzghqwv -> tzghjkv', self.sol.OmegaQ, np.conj(S)), 
                                                        1/(1 - 1j*self.sol.DELTAS) ) ) )

                                    + np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS),
                                    np.einsum('jkqwp, tzghjkv -> tzghqwvp', -1j*self.sol.OmegaQ, 
                                    np.einsum('tzghjkv, gjv -> tzghjkv', 
                                            np.einsum('zghjkvp, tzp -> tzghjkv', np.einsum('zghjkvp, gh -> zghjkvp', self.sol.dsqrtQ, np.sqrt(self.sol.pop)), E), 1/(1 + 1j*self.sol.DELTAS) ) ) )
                                            )

                                    - 4*np.real(np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                            np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzghjkv, gjv -> tzghjkv' , 
                                            np.einsum('tjkqw, tzghqwv -> tzghjkv', np.einsum('jkqwp, tp -> tjkqw', self.sol.OmegaQ, np.abs(Control)), S ),
                                            1/(1+1j*self.sol.DELTAS) ) ) ))
                            )
                self.term_trans = interpn((self.sol.tpoints, self.sol.zCheby), term, np.array([self.tdash.T, (self.sol.cNU*(self.yline[:, None] - self.tdash)/2).T]).T, method='linear', bounds_error=False, fill_value=0.0)
                grad_y = (1/(self.sol.tbounds[-1]*self.sol.gamma))*np.trapz(self.term_trans, x=self.sol.tpoints, axis=1)

            return np.transpose( np.array([np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 0]) ,
                     np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 1])]), (1, 0) )

        elif self.sol.protocol == 'TORCA':
            E = arrays[0]
            S = arrays[1]
            coE = coarrays[0]
            coS = coarrays[1]
                       
            Control_func = self.sol.counter_prop(Control*self.sol.gamma)
            
            if self._complex:
                term = (

                        + np.einsum('tzp, tzp -> tzp' , coE,
                        np.einsum('jkqwvp, tzjkqwvp -> tzp' , -1j*self.sol.dsqrtQ[1], 
                                np.einsum('tzjkqwvp, jqv -> tzjkqwvp' , 
                                            np.einsum('ghjkp,tzghqwv -> tzjkqwvp', self.sol.OmegaQ, np.conj(S)), 
                                            1/(1 + self.sol.gammaSNU - 1j*self.sol.DELTAS) ) ) )

                        + np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS),
                        np.einsum('ghjkp, tzgjkqwv -> tzghqwvp', np.einsum('ghjkp, gh -> ghjkp', -1j*self.sol.OmegaQ, np.sqrt(self.sol.pop)), 
                        np.einsum('tzjkqwv, gjv -> tzgjkqwv', 
                                np.einsum('jkqwvp, tzp -> tzjkqwv', self.sol.dsqrtQ[1], E), 1/(1 + 1j*self.sol.DELTAC) ) ) )

                        - 2*np.real( np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                np.einsum('ghjkp, tzjkqwv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzjkqwv, jqv -> tzjkqwv' , 
                                np.einsum('tzghjk, tzghqwv -> tzjkqwv', np.einsum('ghjkp, ztp -> tzghjk', self.sol.OmegaQ, Control_func(self.sol.t_grid, self.sol.z_grid)), S ),
                                1/(1 + self.sol.gammaSNU + 1j*self.sol.DELTAS) ) ) )
                        )

                )
                self.term_trans = interpn((self.sol.tpoints, self.sol.zCheby), term, np.array([self.tdash.T, (self.sol.cNU*(self.yline[:, None] - self.tdash)/2).T]).T, method='linear', bounds_error=False, fill_value=0.0)
                grad_y = (1/(self.sol.tbounds[-1]*self.sol.gamma))*np.trapz(self.term_trans, x=self.sol.tpoints, axis=1)

            else:
                term = (

                        2*np.real( np.einsum('tzp, tzp -> tzp' , coE,
                        np.einsum('jkqwvp, tzjkqwvp -> tzp' , -1j*self.sol.dsqrtQ[1], 
                                np.einsum('tzjkqwvp, jqv -> tzjkqwvp' , 
                                            np.einsum('ghjkp,tzghqwv -> tzjkqwvp', self.sol.OmegaQ, np.conj(S)), 
                                            1/(1 + self.sol.gammaSNU - 1j*self.sol.DELTAS) ) ) )

                        + np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS),
                        np.einsum('ghjkp, tzgjkqwv -> tzghqwvp', np.einsum('ghjkp, gh -> ghjkp', -1j*self.sol.OmegaQ, np.sqrt(self.sol.pop)), 
                        np.einsum('tzjkqwv, gjv -> tzgjkqwv', 
                                np.einsum('jkqwvp, tzp -> tzjkqwv', self.sol.dsqrtQ[1], E), 1/(1 + 1j*self.sol.DELTAC) ) ) ) )

                        - 4*np.real( np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coS), 
                                np.einsum('ghjkp, tzjkqwv -> tzghqwvp' , self.sol.OmegaQ, np.einsum('tzjkqwv, jqv -> tzjkqwv' , 
                                np.einsum('tzghjk, tzghqwv -> tzjkqwv', np.einsum('ghjkp, ztp -> tzghjk', self.sol.OmegaQ, Control_func(self.sol.t_grid, self.sol.z_grid)), S ),
                                1/(1 + self.sol.gammaSNU + 1j*self.sol.DELTAS) ) ) )
                        )

                )
                self.term_trans = interpn((self.sol.tpoints, self.sol.zCheby), term, np.array([self.tdash.T, (self.sol.cNU*(self.yline[:, None] - self.tdash)/2).T]).T, method='linear', bounds_error=False, fill_value=0.0)
                grad_y = (1/(self.sol.tbounds[-1]*self.sol.gamma))*np.trapz(self.term_trans, x=self.sol.tpoints, axis=1)
                

            return np.transpose( np.array([np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 0]) ,
                     np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 1])]), (1, 0) )

        elif self.sol.protocol == 'TORCAP':
            #Pge = arrays[0]
            E = arrays[0]
            S = arrays[1]
            Pes = arrays[2]
            coE = coarrays[0]
            coS = coarrays[1]
            coPes = coarrays[2]
            # transform coherences
            # can we vecotrise this?
            # for g in range(0, len(self.Fg)):
            #     for h in range(0, len(self.mg)):
            #         for j in range(0, len(self.Fj)):
            #             for k in range(0, len(self.mj)):
            #                 for v in range(0, self.vno):
            #                     self.fPgetrans[:, :, g, h, j, k, v] = scipy.interpolate.griddata(
            #                                                         points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
            #                                                         values = Pge[:, :, g, h, j, k, v].T.reshape(-1, 1),
            #                                                         xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
            #                                                         method = 'linear',
            #                                                         fill_value = 0
            #                                                     ).reshape((len(self._y), len(self._tau_grid)))
                                
            #                     self.bPgetrans[:, :, g, h, j, k, v] = scipy.interpolate.griddata(
            #                                                         points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
            #                                                         values = np.flip(self.Pge[:, :, g, h, j, k, v]).T.reshape(-1, 1),
            #                                                         xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
            #                                                         method = 'linear',
            #                                                         fill_value = 0
            #                                                     ).reshape((len(self._y), len(self._tau_grid)))

            for p in range(0, 2):                    
                self.fEtrans[:, :, p] = scipy.interpolate.griddata(
                                    points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
                                    values = E[:, :, p].T.reshape(-1, 1),
                                    xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
                                    method = 'linear',
                                    fill_value = 0
                                ).reshape((len(self._y), len(self._tau_grid)))
                
                self.bEtrans[:, :, p] = scipy.interpolate.griddata(
                                    points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
                                    values = coE[:, :, p].T.reshape(-1, 1),
                                    xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
                                    method = 'linear',
                                    fill_value = 0
                                ).reshape((len(self._y), len(self._tau_grid)))
            

            for g in range(0, len(self.Fg)):
                for h in range(0, len(self.mg)):
                    for q in range(0, len(self.Fq)):
                        for w in range(0, len(self.mq)):
                            for v in range(0, self.vno):
                                self.fStrans[:, :, g, h, q, w, v] = scipy.interpolate.griddata(
                                                                    points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
                                                                    values = S[:, :, g, h, q, w, v].T.reshape(-1, 1),
                                                                    xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
                                                                    method = 'linear',
                                                                    fill_value = 0
                                                                ).reshape((len(self._y), len(self._tau_grid)))
                                
                                self.bStrans[:, :, g, h, q, w, v] = scipy.interpolate.griddata(
                                                                    points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
                                                                    values = coS[:, :, g, h, q, w, v].T.reshape(-1, 1),
                                                                    xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
                                                                    method = 'linear',
                                                                    fill_value = 0
                                                                ).reshape((len(self._y), len(self._tau_grid)))

            for j in range(0, len(self.Fj)):
                for k in range(0, len(self.mj)):
                    for q in range(0, len(self.Fq)):
                        for w in range(0, len(self.mq)):
                            for v in range(0, self.vno):
                                self.fPestrans[:, :, j, k, q, w, v] = scipy.interpolate.griddata(
                                                                    points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
                                                                    values = Pes[:, :, j, k, q, w, v].T.reshape(-1, 1),
                                                                    xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
                                                                    method = 'linear',
                                                                    fill_value = 0
                                                                ).reshape((len(self._y), len(self._tau_grid)))
                                
                                self.bPestrans[:, :, j, k, q, w, v] = scipy.interpolate.griddata(
                                                                    points = np.hstack([self.tau_grid.reshape(-1, 1), self.y.reshape(-1, 1)]),
                                                                    values = coPes[:, :, j, k, q, w, v].T.reshape(-1, 1),
                                                                    xi = np.hstack([self._tau_grid.reshape(-1, 1), self._y.reshape(-1, 1)]),
                                                                    method = 'linear',
                                                                    fill_value = 0
                                                                ).reshape((len(self._y), len(self._tau_grid)))


            self.fEtrans = np.fft.fft(self.fEtrans, n=None, axis=0)
            self.fEtrans = np.concatenate((self.fEtrans[self.m//2:], self.fEtrans[:self.m//2]))
            self.bEtrans = np.fft.fft(self.bEtrans, n=None, axis=0)
            self.bEtrans = np.concatenate((self.bEtrans[self.m//2:], self.bEtrans[:self.m//2]))

            # self.fPgetrans = np.fft.fft(self.fPgetrans, n=None, axis=0)
            # self.fPgetrans = np.concatenate((self.fPgetrans[self.m//2:], self.fPgetrans[:self.m//2]))
            self.fStrans = np.fft.fft(self.fStrans, n=None, axis=0)
            self.fStrans = np.concatenate((self.fStrans[self.m//2:], self.fStrans[:self.m//2]))
            self.fPestrans = np.fft.fft(self.fPestrans, n=None, axis=0)
            self.fPestrans = np.concatenate((self.fPestrans[self.m//2:], self.fPestrans[:self.m//2]))

            # self.bPgetrans = np.fft.fft(self.bPgetrans, n=None, axis=0)
            # self.bPgetrans = np.concatenate((self.bPgetrans[self.m//2:], self.bPgetrans[:self.m//2]))
            self.bStrans = np.fft.fft(self.bStrans, n=None, axis=0)
            self.bStrans = np.concatenate((self.bStrans[self.m//2:], self.bStrans[:self.m//2]))
            self.bPestrans = np.fft.fft(self.bPestrans, n=None, axis=0)
            self.bPestrans = np.concatenate((self.bPestrans[self.m//2:], self.bPestrans[:self.m//2]))
            #transform Control
            Controltrans = np.transpose( np.array([np.interp(self.yline, self.tpoints+2*0.5/self.cNU, Control[:, 0]),
                             np.interp(self.yline, self.tpoints+2*0.5/self.cNU, Control[:, 1])]), (1, 0) )

            # if _complex:
            #     grad_y = np.trapz(

            #         1j*np.einsum('tzghjkv, ghjkp -> tzp', self.bPgetrans, np.einsum('ghjkp, gh -> ghjkp', self.OmegaQ, np.sqrt(self.pop)))

            #         -1j*np.einsum('tzghqwv, tzghqwvp -> tzp', self.bStrans, np.einsum('ghjkp, tzjkqwv -> tzghqwvp', self.OmegaQ, np.conj(self.fPestrans)))

            #         +1j*np.einsum('tzjkqwv, tzjkqwvp -> tzp', np.conj(self.bPestrans), np.einsum('ghjkp, tzghqwv -> tzjkqwvp', self.OmegaQ, self.fStrans))

                        
            #     , x=self.fs, axis=0)
            if _complex:
                grad_y = np.trapz(

                    - np.einsum('tzghqwv, tzghqwvp -> tzp', self.bStrans,
                        np.einsum('ghjkp, tzgjkqwv -> tzghqwvp', np.einsum('ghjkp, gh -> ghjkp', -1j*self.OmegaQ, np.sqrt(self.pop)), 
                        np.einsum('tzjkqwv, gjv -> tzgjkqwv', 
                                np.einsum('jkqwvp, tzp -> tzjkqwv', self.dsqrtQ, np.conj(self.fEtrans)), 1/(1 - 1j*self.DELTAC) ) ) )

                    -1j*np.einsum('tzghqwv, tzghqwvp -> tzp', self.bStrans, np.einsum('ghjkp, tzjkqwv -> tzghqwvp', self.OmegaQ, np.conj(self.fPestrans)))

                    +1j*np.einsum('tzjkqwv, tzjkqwvp -> tzp', np.conj(self.bPestrans), np.einsum('ghjkp, tzghqwv -> tzjkqwvp', self.OmegaQ, self.fStrans))

                        
                , x=self.fs, axis=0)
            else:
                grad_y = np.trapz(

                        -2*np.imag( 

                            np.einsum('tzghjkv, ghjkp -> tzp', self.bPgetrans, np.einsum('ghjkp, gh -> ghjkp', self.OmegaQ, np.sqrt(self.pop)))

                    -np.einsum('tzghqwv, tzghqwvp -> tzp', self.bStrans, np.einsum('ghjkp, tzjkqwv -> tzghqwvp', self.OmegaQ, np.conj(self.fPestrans)))

                    +np.einsum('tzjkqwv, tzjkqwvp -> tzp', np.conj(self.bPestrans), np.einsum('ghjkp, tzghqwv -> tzjkqwvp', self.OmegaQ, self.fStrans))

                        )
                , x=self.fs, axis=0)

            return np.transpose( np.array([np.interp(self.tpoints, self.yline - 2*0.5/self.cNU, grad_y[:, 0]) ,
                     np.interp(self.tpoints, self.yline - 2*0.5/self.cNU, grad_y[:, 1])]), (1, 0) )
        
        elif self.sol.protocol == 'ORCA_GSM':
            if field == 0 or field == 1: # usual Control field (field = 0) or first mapping field (field = 1)
                P = arrays[0]
                S = arrays[1]
                coP = coarrays[0]
                coS = coarrays[1]
                # transform coherences
                self.fPtrans = LinearNDInterpolator(np.hstack([self.sol.t_grid.reshape(-1, 1), self.y.reshape(-1, 1)]), P.transpose(1, 0, 2, 3, 4, 5, 6).reshape(-1, *P[0, 0].shape), fill_value=0.0)(self._t_grid, self._y)
                self.bPtrans = LinearNDInterpolator(np.hstack([self.sol.t_grid.reshape(-1, 1), self.y.reshape(-1, 1)]), coP.transpose(1, 0, 2, 3, 4, 5, 6).reshape(-1, *P[0, 0].shape), fill_value=0.0)(self._t_grid, self._y)
                self.fStrans = LinearNDInterpolator(np.hstack([self.sol.t_grid.reshape(-1, 1), self.y.reshape(-1, 1)]), S.transpose(1, 0, 2, 3, 4, 5, 6).reshape(-1, *S[0, 0].shape), fill_value=0.0)(self._t_grid, self._y)
                self.bStrans = LinearNDInterpolator(np.hstack([self.sol.t_grid.reshape(-1, 1), self.y.reshape(-1, 1)]), coS.transpose(1, 0, 2, 3, 4, 5, 6).reshape(-1, *S[0, 0].shape), fill_value=0.0)(self._t_grid, self._y)

                self.fPtrans = np.fft.fft(self.fPtrans, n=None, axis=0)
                self.fStrans = np.fft.fft(self.fStrans, n=None, axis=0)
                self.bPtrans = np.fft.fft(self.bPtrans, n=None, axis=0)
                self.bStrans = np.fft.fft(self.bStrans, n=None, axis=0)
                
                self.fPtrans = (2/self.sol.m)*np.concatenate((self.fPtrans[self.sol.m//2:], self.fPtrans[:self.sol.m//2]))
                self.fStrans = (2/self.sol.m)*np.concatenate((self.fStrans[self.sol.m//2:], self.fStrans[:self.sol.m//2]))
                self.bPtrans = (2/self.sol.m)*np.concatenate((self.bPtrans[self.sol.m//2:], self.bPtrans[:self.sol.m//2]))
                self.bStrans = (2/self.sol.m)*np.concatenate((self.bStrans[self.sol.m//2:], self.bStrans[:self.sol.m//2]))

                # self.fPtrans = np.sqrt(self.sol.tbounds[-1])*np.concatenate((self.fPtrans[self.sol.m//2:], self.fPtrans[:self.sol.m//2]))
                # self.fStrans = np.sqrt(self.sol.tbounds[-1])*np.concatenate((self.fStrans[self.sol.m//2:], self.fStrans[:self.sol.m//2]))
                # self.bPtrans = np.sqrt(self.sol.tbounds[-1])*np.concatenate((self.bPtrans[self.sol.m//2:], self.bPtrans[:self.sol.m//2]))
                # self.bStrans = np.sqrt(self.sol.tbounds[-1])*np.concatenate((self.bStrans[self.sol.m//2:], self.bStrans[:self.sol.m//2]))
                #transform Control
                Controltrans = np.transpose( np.array([np.interp(self.yline, self.sol.tpoints+2*0.5/self.sol.cNU, Control[:, 0]),
                                np.interp(self.yline, self.sol.tpoints+2*0.5/self.sol.cNU, Control[:, 1])]), (1, 0) )

                if self._complex:
                    # self.sol.dsqrtQ[1] use value in cell
                    grad_y = ( -1j*np.trapz( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                        np.conj(self.bStrans),
                                        np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaQ, self.fPtrans) )

                                - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                            self.bPtrans,
                                            np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaQ, np.conj(self.fStrans) ) )
                                , x=self.fs, axis=0)
                        )
                else:
                    grad_y = ( +2*np.real( np.trapz( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                        np.conj(self.bStrans),
                                        np.einsum('jkqwp, tzghjkv -> tzghqwvp' , -1j*self.sol.OmegaQ, self.fPtrans) )

                                - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                            self.bPtrans,
                                            np.einsum('jkqwp, tzghqwv -> tzghjkvp' , -1j*self.sol.OmegaQ, np.conj(self.fStrans) ) )
                                , x=self.fs, axis=0) )
                        )

                return np.transpose( np.array([np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 0]) ,
                        np.interp(self.sol.tpoints, self.yline - 2*0.5/self.sol.cNU, grad_y[:, 1])]), (1, 0) )

            elif field == 2:
                P = arrays[0]
                S = arrays[1]
                coP = coarrays[0]
                coS = coarrays[1]
                if self._complex:
                    return( -1j*np.trapz( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                        np.conj(coS),
                                        np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaM2Q, P) )

                                - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                            coP,
                                            np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaM2Q, np.conj(S) ) )
                                , x=self.sol.zCheby, axis=1)
                        )
                else:
                    return( +2*np.imag( np.trapz( np.einsum('tzghqwv, tzghqwvp -> tzp', 
                                        np.conj(coS),
                                        np.einsum('jkqwp, tzghjkv -> tzghqwvp' , self.sol.OmegaM2Q, P) )

                                - np.einsum('tzghjkv, tzghjkvp -> tzp', 
                                            coP,
                                            np.einsum('jkqwp, tzghqwv -> tzghjkvp' , self.sol.OmegaM2Q, np.conj(S) ) )
                                , x=self.sol.zCheby, axis=1) )
                        )

        elif self.sol.protocol == '4levelTORCAP':
            if field == 1: # dressing field - co propagating
                Sgs = arrays[0] 
                Sgb = arrays[1]
                Pes = arrays[2] 
                Peb = arrays[3]
                coSgs = coarrays[0] 
                coSgb = coarrays[1]
                coPes = coarrays[2] 
                coPeb = coarrays[3]
                if self._complex:
                    return( np.trapz(
                                np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coSgs), 
                                np.einsum('qwbxp,tzghbxv -> tzghqwvp', -1j*self.sol.OmegaQ2, Sgb))

                                + np.einsum('tzghbxv, tzghbxvp -> tzp', coSgb, 
                                np.einsum('qwbxp,tzghqwv -> tzghbxvp', +1j*self.sol.OmegaQ2, np.conj(Sgs)))

                                + np.einsum('tzjkqwv, tzjkqwvp -> tzp', np.conj(coPes), 
                                np.einsum('qwbxp,tzjkbxv -> tzjkqwvp', -1j*self.sol.OmegaQ2, Peb))

                                + np.einsum('tzjkbxv, tzjkbxvp -> tzp', coPeb, 
                                np.einsum('qwbxp,tzjkqwv -> tzjkbxvp', +1j*self.sol.OmegaQ2, np.conj(Pes)))

                                , x=self.sol.zCheby, axis=1)
                        )
                else:
                    return( np.trapz( +2*np.real(
                                np.einsum('tzghqwv, tzghqwvp -> tzp', np.conj(coSgs), 
                                np.einsum('qwbxp,tzghbxv -> tzghqwvp', -1j*self.sol.OmegaQ2, Sgb))

                                + np.einsum('tzghbxv, tzghbxvp -> tzp', coSgb, 
                                np.einsum('qwbxp,tzghqwv -> tzghbxvp', +1j*self.sol.OmegaQ2, np.conj(Sgs)))

                                + np.einsum('tzjkqwv, tzjkqwvp -> tzp', np.conj(coPes), 
                                np.einsum('qwbxp,tzjkbxv -> tzjkqwvp', -1j*self.sol.OmegaQ2, Peb))

                                + np.einsum('tzjkbxv, tzjkbxvp -> tzp', coPeb, 
                                np.einsum('qwbxp,tzjkqwv -> tzjkbxvp', +1j*self.sol.OmegaQ2, np.conj(Pes)))
                                )

                                , x=self.sol.zCheby, axis=1)
                        )

   
    def arrays_for_counter_prop(self):
        if self.sol.protocol == 'TORCAP':
            # make coord grids for transform
            self.y = self.sol.t_grid + 2*self.sol.z_grid/self.sol.cNU # t_grid, y make up irregular grid 
            self.yline = np.linspace(0, self.sol.tbounds[-1]*self.sol.gamma+2/self.sol.cNU, self.sol.m)
            self._t_grid, self._y = np.meshgrid(self.sol.tpoints, self.yline) # _t_grid, _y make up regular grid 
            self.fs = np.fft.fftshift(np.fft.fftfreq(self.sol.m, d=self.sol.tpoints[1] - self.sol.tpoints[0]))
            # make grids for transformed coherences
            self.fEtrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.E.shape[2:])), dtype=complex)
            self.bEtrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.E.shape[2:])), dtype=complex)
            #self.fPgetrans = np.zeros(np.concatenate(((max(self.m, self.n), max(self.m, self.n)), self.Pge.shape[2:])), dtype=complex)
            self.fStrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.S.shape[2:])), dtype=complex)
            self.fPestrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.Pes.shape[2:])), dtype=complex)
            #self.bPgetrans = np.zeros(np.concatenate(((max(self.m, self.n), max(self.m, self.n)), self.Pge.shape[2:])), dtype=complex)
            self.bStrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.S.shape[2:])), dtype=complex)
            self.bPestrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.Pes.shape[2:])), dtype=complex)
        elif self.sol.protocol == 'ORCA' or self.sol.protocol == 'TORCA':
            # make coord grids for transform
            self.y = self.sol.t_grid + 2*self.sol.z_grid/self.sol.cNU # t_grid, y make up irregular grid 
            self.yline = np.linspace(0, self.sol.tbounds[-1]*self.sol.gamma+2/self.sol.cNU, self.sol.m)
            self._t_grid, self._y = np.meshgrid(self.sol.tpoints, self.yline) # _t_grid, _y make up regular grid 
            tstarts = self.yline - 2/self.sol.cNU
            tends = np.array(self.yline)
            self.tdash = np.linspace(tstarts, tends, self.sol.m).T #(y, t)
            # make grids for transformed grad
            self.term_trans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.E.shape[2:])), dtype=complex)
        elif self.sol.protocol == 'FLAME':
            # make coord grids for transform
            self.y = self.sol.t_grid + 2*self.sol.z_grid/self.sol.cNU # t_grid, y make up irregular grid 
            self.yline = np.linspace(0, self.sol.tbounds[-1]*self.sol.gamma+2/self.sol.cNU, self.sol.m)
            self._t_grid, self._y = np.meshgrid(self.sol.tpoints, self.yline) # _t_grid, _y make up regular grid 
            tstarts = self.yline - 2/self.sol.cNU
            tends = np.array(self.yline)
            self.tdash = np.linspace(tstarts, tends, self.sol.m).T #(y, t)
            # make grids for transformed grad
            self.term_trans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.E.shape[2:])), dtype=complex)
        elif self.sol.protocol == 'ORCA_GSM':
            # make coord grids for transform
            self.y = self.sol.t_grid + 2*self.sol.z_grid/self.sol.cNU # t_grid, y make up irregular grid 
            self.yline = np.linspace(0, self.sol.tbounds[-1]*self.sol.gamma+2/self.sol.cNU, self.sol.m)
            self._t_grid, self._y = np.meshgrid(self.sol.tpoints, self.yline) # _t_grid, _y make up regular grid 
            self.fs = np.fft.fftshift(np.fft.fftfreq(self.sol.m, d=self.sol.tpoints[1] - self.sol.tpoints[0]))
            # make grids for transformed coherences
            self.fPtrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.Pge.shape[2:])), dtype=complex)
            self.fStrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.Sgs.shape[2:])), dtype=complex)
            self.bPtrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.Pge.shape[2:])), dtype=complex)
            self.bStrans = np.zeros(np.concatenate(((max(self.sol.m, self.sol.n), max(self.sol.m, self.sol.n)), self.sol.Sgs.shape[2:])), dtype=complex)

    def backward_retrieval_opt_storage(self, Controls):
        if self.sol.protocol == 'Raman':
            [Control] = Controls
            # co-propagating
            Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            # initial conditions for forward retreival
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.S[0] = np.conj(np.flip(self.sol.S[-1], axis=0))/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # # solve co-function storage (time and space reversed), must time reverse control
            Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0))
            self.sol.solve(Control_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
            total_eff = retrieval_eff * storage_eff
            grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            grads = np.array([grad])
        return storage_eff, total_eff, grads

    def backward_retrieval_opt_all(self, Controls):
        if self.sol.protocol == 'Raman':
            [Control, Control_readout] = Controls
            # co-propagating
            Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            # initial conditions for forward retreival
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.S[0] = np.flip(self.sol.S[-1], axis=0)/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve backwards read out
            Control_readout_func = self.sol.co_prop(Control_readout)
            self.sol.solve(Control_readout_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
            total_eff = retrieval_eff * storage_eff
            Ecopy2 = np.array(self.sol.E)
            Scopy2 = np.array(self.sol.S)
            # need to flip Eout and set it as Einit
            # renormalise?
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            Control_readout_func = self.sol.co_prop(np.flip(np.conj(Control_readout), axis=0))
            self.sol.solve(Control_readout_func, method=self.method)
            grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
            # retrieve from co-system
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.S[0] = np.flip(self.sol.S[-1], axis=0)/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0))
            self.sol.solve(Control_func, method=self.method)
            grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            grads = np.array([grad, grad2])
        return storage_eff, total_eff, grads

    def forward_retrieval_opt_storage(self, Controls):
        if self.sol.protocol == 'EIT':
            [Control, Control_readout] = Controls
            # co-propagating
            Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Pcopy = np.array(self.sol.P)
                Scopy = np.array(self.sol.S)
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.P[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval
                # use very strong gaussian control field - read out all spin wave
                # co-propagating
                Control_readout_func = self.sol.co_prop(Control_readout)
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = storage_eff * retrieval_eff
                # Pcopy2 = np.array(self.sol.P)
                # Scopy2 = np.array(self.sol.S)
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.P[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                # co-propagating
                Control_readout_func = self.sol.co_prop(np.flip(np.conj(Control_readout), axis=0))
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                #grad2 = self.grad([Pcopy2, Scopy2], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
                # retrieve from co-system
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.P[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0))
                self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                grad = self.grad([Pcopy, Scopy], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
                # set up boundary conditions for next run
                grads = np.array([grad, np.zeros((self.sol.m, 2))])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'Raman':
            [Control, Control_readout] = Controls
            # co-propagating
            Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            # initial conditions for forward retreival
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval
            # use very strong gaussian control field - read out all spin wave
            Control_readout_func = self.sol.co_prop(Control_readout)
            self.sol.solve(Control_readout_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
            #norm[norm == 0] = 1 
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # # solve co-function storage (time and space reversed), must time reverse control
            Control_readout_func = self.sol.co_prop(np.flip(np.conj(Control_readout), axis=0))
            self.sol.solve(Control_readout_func, method=self.method)
            # retrieve from co-system
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            #self.S[0] = np.conj(np.flip(Scopy[-1], axis=0))
            #self.S[0] = np.conj(Scopy[-1]) #self.S[-1]
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            #Control_func = self.co_prop(np.conj(np.flip(-Control, axis=0)))
            Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0))
            self.sol.solve(Control_func, method=self.method)
            grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            grads = np.array([grad, np.zeros((self.sol.m, 2))]) # set gradient for readout control to be zero
        #elif self.sol.protocol == 'ORCA':
            # [Control, Control_readout] = Controls
            # # counter-propagating
            # # need to reset detunings
            # self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            # Control_func = self.sol.counter_prop(Control) # func(t, z) => (p)
            # self.sol.solve(Control_func, method=self.method)
            # # need to save P and S coherences for gradient calculation
            # Ecopy = np.array(self.sol.E)
            # Scopy = np.array(self.sol.S)
            # # initial conditions for forward retreival
            # storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # # normalise spin wave?
            # self.sol.E[:] = 0
            # self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            # self.sol.S[1:] = 0
            # self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # # solve forward retrieval
            # # counter-propagating
            # # need to reset detunings
            # self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            # Control_readout_func = self.sol.counter_prop(Control_readout)
            # self.sol.solve(Control_readout_func, method=self.method)
            # retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
            # total_eff = storage_eff * retrieval_eff
            # # need to flip Eout and set it as Einit
            # # renormalise?
            # self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # # initial condition for co-function run
            # self.sol.E[:] = 0
            # self.sol.S[:] = 0
            # # # solve co-function storage (time and space reversed), must time reverse control
            # # counter-propagating
            # # need to reset detunings
            # self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            # Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
            # self.sol.solve(Control_readout_func, method=self.method)
            # # retrieve from co-system
            # # normalise spin wave?
            # self.sol.E[:] = 0
            # self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            # self.sol.S[1:] = 0
            # self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # # counter-propagating
            # # need to reset detunings
            # self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            # Control_func = self.sol.counter_prop(np.flip(np.conj(Control), axis=0))
            # self.sol.solve(Control_func, method=self.method)
            # grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
            # # set up boundary conditions for next run
            # self.sol.E[:] = 0
            # self.sol.S[:] = 0
            # self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # grads = np.array([grad, np.zeros((self.sol.m, 2))]) # set gradient for readout control to be zero
        elif self.sol.protocol == 'ORCA':
            [Control, Control_readout] = Controls
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy = np.array(self.sol.E)
                Scopy = np.array(self.sol.S)
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval
                # use very strong gaussian control field - read out all spin wave
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_readout_func = self.sol.counter_prop(Control_readout)
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = storage_eff * retrieval_eff
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                # retrieve from co-system
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_func = self.sol.counter_prop(np.flip(np.conj(Control), axis=0))
                self.sol.solve(Control_func, method=self.method)
                grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
                grads = np.array([grad, np.zeros((self.sol.m, 2))])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'FLAME':
            [Control, Control_readout] = Controls
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            # need to save P and S coherences for gradient calculation
            Pcopy = np.array(self.sol.P)
            Scopy = np.array(self.sol.S)
            # initial conditions for forward retreival
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_readout_func = self.sol.counter_prop(Control_readout)
            self.sol.solve(Control_readout_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[:] = 0
            # # solve co-function storage (time and space reversed), must time reverse control
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
            self.sol.solve(Control_readout_func, method=self.method)
            # retrieve from co-system
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(np.flip(np.conj(Control), axis=0))
            self.sol.solve(Control_func, method=self.method)
            grad = self.grad([Pcopy, Scopy], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            grads = np.array([grad, np.zeros((self.sol.m, 2))]) # set gradient for readout control to be zero
        return storage_eff, total_eff, grads

    def forward_retrieval_opt_all(self, Controls):
        if self.sol.protocol == 'Raman':
            [Control, Control_readout] = Controls
            # co-propagating
            Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy = np.array(self.sol.E)
                Scopy = np.array(self.sol.S)
                
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval
                # use very strong gaussian control field - read out all spin wave
                Control_readout_func = self.sol.co_prop(Control_readout)
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = retrieval_eff * storage_eff
                Ecopy2 = np.array(self.sol.E)
                Scopy2 = np.array(self.sol.S)
                
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) #interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                Control_readout_func = self.sol.co_prop(np.flip(np.conj(Control_readout), axis=0))
                self.sol.solve(Control_readout_func, method=self.method)
                grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
            if self.sol.solved:
                # retrieve from co-system
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                #self.S[0] = np.conj(np.flip(Scopy[-1], axis=0))
                #self.S[0] = np.conj(Scopy[-1]) #self.S[-1]
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                #Control_func = self.co_prop(np.conj(np.flip(-Control, axis=0)))
                Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0))
                self.sol.solve(Control_func, method=self.method)
                grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
                grads = np.array([grad, grad2])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'EIT':
            [Control, Control_readout] = Controls
            # co-propagating
            Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Pcopy = np.array(self.sol.P)
                Scopy = np.array(self.sol.S)
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.P[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval
                # use very strong gaussian control field - read out all spin wave
                # co-propagating
                Control_readout_func = self.sol.co_prop(Control_readout)
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = storage_eff * retrieval_eff
                Pcopy2 = np.array(self.sol.P)
                Scopy2 = np.array(self.sol.S)
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.P[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                # co-propagating
                Control_readout_func = self.sol.co_prop(np.flip(np.conj(Control_readout), axis=0))
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                grad2 = self.grad([Pcopy2, Scopy2], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
                # retrieve from co-system
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.P[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0))
                self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                grad = self.grad([Pcopy, Scopy], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
                # set up boundary conditions for next run
                grads = np.array([grad, grad2])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'ORCA':
            [Control, Control_readout] = Controls
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy = np.array(self.sol.E)
                Scopy = np.array(self.sol.S)
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval
                # use very strong gaussian control field - read out all spin wave
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_readout_func = self.sol.counter_prop(Control_readout)
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = storage_eff * retrieval_eff
                Ecopy2 = np.array(self.sol.E)
                Scopy2 = np.array(self.sol.S)
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
                # retrieve from co-system
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_func = self.sol.counter_prop(np.flip(np.conj(Control), axis=0))
                self.sol.solve(Control_func, method=self.method)
                grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
                grads = np.array([grad, grad2])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'TORCA':
            [Control, Control_readout] = Controls
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy = np.array(self.sol.E)
                Scopy = np.array(self.sol.S)
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval
                # use very strong gaussian control field - read out all spin wave
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_readout_func = self.sol.counter_prop(Control_readout)
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = storage_eff * retrieval_eff
                Ecopy2 = np.array(self.sol.E)
                Scopy2 = np.array(self.sol.S)
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
                self.sol.solve(Control_readout_func, method=self.method)
            if self.sol.solved:
                grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
                # retrieve from co-system
                # normalise spin wave?
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_func = self.sol.counter_prop(np.flip(np.conj(Control), axis=0))
                self.sol.solve(Control_func, method=self.method)
                grad = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
                grads = np.array([grad, grad2])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'FLAME':
            [Control, Control_readout] = Controls
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(Control) # func(t, z) => (p)
            self.sol.solve(Control_func, method=self.method)
            # need to save P and S coherences for gradient calculation
            Pcopy = np.array(self.sol.P)
            Scopy = np.array(self.sol.S)
            # initial conditions for forward retreival
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval
            # use very strong gaussian control field - read out all spin wave
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_readout_func = self.sol.counter_prop(Control_readout)
            self.sol.solve(Control_readout_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
            total_eff = storage_eff * retrieval_eff
            Pcopy2 = np.array(self.sol.P)
            Scopy2 = np.array(self.sol.S)
            # need to flip Eout and set it as Einit
            # renormalise?
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
            self.sol.solve(Control_readout_func, method=self.method)
            grad2 = self.grad([Pcopy2, Scopy2], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control_readout/self.sol.gamma)
            # retrieve from co-system
            # normalise spin wave?
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func = self.sol.counter_prop(np.flip(np.conj(Control), axis=0))
            self.sol.solve(Control_func, method=self.method)
            grad = self.grad([Pcopy, Scopy], [np.conj(np.flip(np.flip(self.sol.P, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.P[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            grads = np.array([grad, grad2])
        elif self.sol.protocol == 'ORCA_GSM':
            [Control, M1, M2, Control_readout, M1_readout, M2_readout] = Controls
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_tzp = self.sol.counter_prop( Control, zdef=0.5, field=0)
            M1_tzp = self.sol.counter_prop( M1, zdef=0.5, field=1)
            M2_tzp = self.sol.co_prop( M2)
            Control_funcs = np.array([Control_tzp, M1_tzp, M2_tzp])
            self.sol.solve(Control_funcs, method=self.method)
            normed = 1 # kepp track of normalising spin waves for efficiency calculation
            # print("Storage: ", self.sol.solved)
            # print(self.sol.check_energy())
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Pgecopy = np.array(self.sol.Pge)
                Pge2copy = np.array(self.sol.Pge2)
                Sgscopy = np.array(self.sol.Sgs)
                Sgbcopy = np.array(self.sol.Sgb)
                # initial conditions for forward retreival
                storage_eff = self.sol.storage_efficiency(self.sol.Sgb, -1)
                ### Optional storage time
                if self.extra_params:
                    # no control pulses - propagate coherences for storage time
                    # save original tend
                    tend = self.sol.tbounds[-1]
                    # change time of simulation to storage time
                    self.sol.tbounds = np.array([0, self.extra_params[0]])
                    self.sol.tpoints = np.linspace(self.sol.tbounds[0]*self.sol.gamma, self.sol.tbounds[1]*self.sol.gamma, self.sol.m)
                    self.sol.tstep = self.sol.tpoints[1] - self.sol.tpoints[0] 
                    self.sol.t_grid, self.sol.z_grid = np.meshgrid(self.sol.tpoints, self.sol.zCheby)
                    # Normalise storage spin wave?
                    eff_bothS = self.sol.storage_efficiency(self.sol.Sgs, -1) + storage_eff
                    self.sol.E[:] = 0
                    self.sol.Pge[:] = 0
                    self.sol.Pge2[:] = 0
                    self.sol.Sgs[0] = self.sol.Sgs[-1]/np.sqrt(eff_bothS)
                    self.sol.Sgs[1:] = 0
                    self.sol.Sgb[0] = self.sol.Sgb[-1]/np.sqrt(eff_bothS)
                    self.sol.Sgb[1:] = 0
                    self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                    normed *= eff_bothS
                    # need to reset detunings
                    self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                    no_Control_tzp = self.sol.counter_prop( np.zeros(self.sol.m)[:, None]*np.array([1, 0]), zdef=0.5, field=0)
                    no_M1_tzp = self.sol.counter_prop( np.zeros(self.sol.m)[:, None]*np.array([1, 0]), zdef=0.5, field=1)
                    no_M2_tzp = self.sol.co_prop( np.zeros(self.sol.m)[:, None]*np.array([1, 0]) )
                    no_Control_funcs = np.array([no_Control_tzp, no_M1_tzp, no_M2_tzp])
                    self.sol.solve(no_Control_funcs, method=self.method)
                    # print("Optional storage time: ", self.sol.solved)
                    # print(self.sol.check_energy())
            if self.sol.solved:
                ### Retrieval
                # return to original tbounds
                self.sol.tbounds = np.array([0, tend])
                self.sol.tpoints = np.linspace(self.sol.tbounds[0]*self.sol.gamma, self.sol.tbounds[1]*self.sol.gamma, self.sol.m)
                self.sol.tstep = self.sol.tpoints[1] - self.sol.tpoints[0] 
                self.sol.t_grid, self.sol.z_grid = np.meshgrid(self.sol.tpoints, self.sol.zCheby)
                # Normalise storage spin wave?
                storage_eff2 = self.sol.storage_efficiency(self.sol.Sgb, -1) # maybe different to storage_eff if used extra_params to store for longer
                eff_bothS = self.sol.storage_efficiency(self.sol.Sgs, -1) + storage_eff2
                self.sol.E[:] = 0
                self.sol.Pge[:] = 0
                self.sol.Pge2[:] = 0
                self.sol.Sgs[0] = self.sol.Sgs[-1]/np.sqrt(eff_bothS)
                self.sol.Sgs[1:] = 0
                self.sol.Sgb[0] = self.sol.Sgb[-1]/np.sqrt(eff_bothS)
                self.sol.Sgb[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                normed *= eff_bothS
                # solve forward retrieval
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_tzp = self.sol.counter_prop( Control_readout, zdef=0.5, field=0)
                M1_tzp = self.sol.counter_prop( M1_readout, zdef=0.5, field=1)
                M2_tzp = self.sol.co_prop( M2_readout )
                Control_readout_funcs = np.array([Control_tzp, M1_tzp, M2_tzp])
                self.sol.solve(Control_readout_funcs, method=self.method)
                # print("Readout: ", self.sol.solved)
                # print(self.sol.check_energy())

            if self.sol.solved:
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1) # if spin wave normalised, this is retrieval efficiency
                total_eff = normed * retrieval_eff
                normed = 1
                # need to save P and S coherences for gradient calculation
                Pgecopy2 = np.array(self.sol.Pge)
                Pge2copy2 = np.array(self.sol.Pge2)
                Sgscopy2 = np.array(self.sol.Sgs)
                Sgbcopy2 = np.array(self.sol.Sgb)
                # need to flip Eout and set it as Einit
                # renormalise?
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.Pge[:] = 0
                self.sol.Pge2[:] = 0
                self.sol.Sgs[:] = 0
                self.sol.Sgb[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                # counter-propagating
                # need to reset detunings
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_tzp = self.sol.counter_prop( np.flip(np.conj(Control_readout), axis=0), zdef=0.5, field=0)
                M1_tzp = self.sol.counter_prop( np.flip(np.conj(M1_readout), axis=0), zdef=0.5, field=1)
                M2_tzp = self.sol.co_prop( np.flip(np.conj(M2_readout), axis=0))
                Control_readout_funcs = np.array([Control_tzp, M1_tzp, M2_tzp])
                self.sol.solve(Control_readout_funcs, method=self.method)
                # print("Co-Storage: ", self.sol.solved)
                # print(self.sol.check_energy())
                # readout grads
                readout_grad_control = self.grad([Pgecopy2, Sgscopy2], 
                                                [np.conj(np.flip(np.flip(self.sol.Pge, axis=0), axis=1)), 
                                                np.conj(np.flip(np.flip(self.sol.Sgs, axis=0), axis=1))], 
                                                Control_readout/self.sol.gamma, 
                                                field=0)
                readout_grad_M1 = self.grad([Pge2copy2, Sgscopy2], 
                                                [np.conj(np.flip(np.flip(self.sol.Pge2, axis=0), axis=1)), 
                                                np.conj(np.flip(np.flip(self.sol.Sgs, axis=0), axis=1))], 
                                                M1_readout/self.sol.gamma, 
                                                field=1)
                readout_grad_M2 = self.grad([Pge2copy2, Sgbcopy2], 
                                                [np.conj(np.flip(np.flip(self.sol.Pge2, axis=0), axis=1)), 
                                                np.conj(np.flip(np.flip(self.sol.Sgb, axis=0), axis=1))], 
                                                M2_readout/self.sol.gamma, 
                                                field=2)
                # print("Done readout grads")
                ### Optional storage time
                if self.extra_params:
                    # no control pulses - propagate coherences for storage time
                    # save original tend
                    tend = self.sol.tbounds[-1]
                    # change time of simulation to storage time
                    self.sol.tbounds = np.array([0, self.extra_params[0]])
                    self.sol.tpoints = np.linspace(self.sol.tbounds[0]*self.sol.gamma, self.sol.tbounds[1]*self.sol.gamma, self.sol.m)
                    self.sol.tstep = self.sol.tpoints[1] - self.sol.tpoints[0] 
                    self.sol.t_grid, self.sol.z_grid = np.meshgrid(self.sol.tpoints, self.sol.zCheby)
                    # Normalise storage spin wave?
                    eff_bothS = self.sol.storage_efficiency(self.sol.Sgs, -1) + self.sol.storage_efficiency(self.sol.Sgb, -1)
                    self.sol.E[:] = 0
                    self.sol.Pge[:] = 0
                    self.sol.Pge2[:] = 0
                    self.sol.Sgs[0] = self.sol.Sgs[-1]/np.sqrt(eff_bothS)
                    self.sol.Sgs[1:] = 0
                    self.sol.Sgb[0] = self.sol.Sgb[-1]/np.sqrt(eff_bothS)
                    self.sol.Sgb[1:] = 0
                    self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                    normed *= eff_bothS
                    # need to reset detunings
                    self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                    no_Control_tzp = self.sol.counter_prop( np.zeros(self.sol.m)[:, None]*np.array([1, 0]), zdef=0.5, field=0)
                    no_M1_tzp = self.sol.counter_prop( np.zeros(self.sol.m)[:, None]*np.array([1, 0]), zdef=0.5, field=1)
                    no_M2_tzp = self.sol.co_prop( np.zeros(self.sol.m)[:, None]*np.array([1, 0]) )
                    no_Control_funcs = np.array([no_Control_tzp, no_M1_tzp, no_M2_tzp])
                    self.sol.solve(no_Control_funcs, method=self.method)
                    # print("Co-optional storage time: ", self.sol.solved)
                    # print(self.sol.check_energy())
            if self.sol.solved:
                ### Retrieve from co-system
                # return to original tbounds
                self.sol.tbounds = np.array([0, tend])
                self.sol.tpoints = np.linspace(self.sol.tbounds[0]*self.sol.gamma, self.sol.tbounds[1]*self.sol.gamma, self.sol.m)
                self.sol.tstep = self.sol.tpoints[1] - self.sol.tpoints[0] 
                self.sol.t_grid, self.sol.z_grid = np.meshgrid(self.sol.tpoints, self.sol.zCheby)
                # Normalise spin wave?
                eff_bothS = self.sol.storage_efficiency(self.sol.Sgs, -1) + self.sol.storage_efficiency(self.sol.Sgb, -1)
                self.sol.E[:] = 0
                self.sol.Pge[:] = 0
                self.sol.Pge2[:] = 0
                self.sol.Sgs[0] = self.sol.Sgs[-1]/np.sqrt(eff_bothS)
                self.sol.Sgs[1:] = 0
                self.sol.Sgb[0] = self.sol.Sgb[-1]/np.sqrt(eff_bothS)
                self.sol.Sgb[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                normed *= eff_bothS
                self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
                Control_tzp = self.sol.counter_prop( np.flip(np.conj(Control), axis=0), zdef=0.5, field=0)
                M1_tzp = self.sol.counter_prop( np.flip(np.conj(M1), axis=0), zdef=0.5, field=1)
                M2_tzp = self.sol.co_prop( np.flip(np.conj(M2), axis=0))
                Control_funcs = np.array([Control_tzp, M1_tzp, M2_tzp])
                self.sol.solve(Control_funcs, method=self.method)
                # print("Co-retrieval: ", self.sol.solved)
                # print(self.sol.check_energy())
                # readin grads
                readin_grad_control = self.grad([Pgecopy, Sgscopy], 
                                                [np.conj(np.flip(np.flip(self.sol.Pge, axis=0), axis=1)), 
                                                np.conj(np.flip(np.flip(self.sol.Sgs, axis=0), axis=1))], 
                                                Control/self.sol.gamma, 
                                                field=0)
                readin_grad_M1 = self.grad([Pge2copy, Sgscopy], 
                                                [np.conj(np.flip(np.flip(self.sol.Pge2, axis=0), axis=1)), 
                                                np.conj(np.flip(np.flip(self.sol.Sgs, axis=0), axis=1))], 
                                                M1/self.sol.gamma, 
                                                field=1)
                readin_grad_M2 = self.grad([Pge2copy, Sgbcopy], 
                                                [np.conj(np.flip(np.flip(self.sol.Pge2, axis=0), axis=1)), 
                                                np.conj(np.flip(np.flip(self.sol.Sgb, axis=0), axis=1))], 
                                                M2/self.sol.gamma, 
                                                field=2)
                grads = np.array([readin_grad_control, readin_grad_M1, readin_grad_M2, readout_grad_control, readout_grad_M1, readout_grad_M2])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0

            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.Pge[:] = 0
            self.sol.Pge2[:] = 0
            self.sol.Sgs[:] = 0
            self.sol.Sgb[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            
        return storage_eff, total_eff, grads

    def forward_retrieval_opt_MZ_storage(self, Controls):
        [Control1, Control2, Control_readout] = Controls
        [eta_loop] = self.extra_params
        if self.sol.protocol == 'Raman':
            # co-propagating
            Control_func1 = self.sol.co_prop(Control1) # func(t, z) => (p)
            self.sol.solve(Control_func1, method=self.method)
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            # MZ
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            Control_func2 = self.sol.co_prop(Control2)
            self.sol.solve(Control_func2, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy2 = np.array(self.sol.E)
            Scopy2 = np.array(self.sol.S)
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # initial conditions for forward retreival
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval
            # use very strong gaussian control field - read out all spin wave
            Control_readout_func = self.sol.co_prop(Control_readout)
            self.sol.solve(Control_readout_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1)
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
            #norm[norm == 0] = 1 
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            Control_readout_func = self.sol.co_prop(np.flip(np.conj(Control_readout), axis=0))
            self.sol.solve(Control_readout_func, method=self.method)
            # retrieve from co-system
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            Control_func2 = self.sol.co_prop(np.conj(np.flip(Control2, axis=0)))
            self.sol.solve(Control_func2, method=self.method)            
            grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control2/self.sol.gamma)
            #MZ retrieve
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]#/np.sqrt(self.storage_efficiency(self.S, -1))
            self.sol.S[1:] = 0
            Control_func1 = self.sol.co_prop(np.conj(np.flip(Control1, axis=0)))
            self.sol.solve(Control_func1, method=self.method)
            grad1 = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control1/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'ORCA':
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(Control1) # func(t, z) => (p)
            self.sol.solve(Control_func1, method=self.method)
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            # MZ
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func2 = self.sol.counter_prop(Control2)
            self.sol.solve(Control_func2, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy2 = np.array(self.sol.E)
            Scopy2 = np.array(self.sol.S)
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # initial conditions for forward retreival
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval
            # use very strong gaussian control field - read out all spin wave
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_readout_func = self.sol.counter_prop(Control_readout)
            self.sol.solve(Control_readout_func, method=self.method)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1)
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
            #norm[norm == 0] = 1 
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_readout_func = self.sol.counter_prop(np.flip(np.conj(Control_readout), axis=0))
            self.sol.solve(Control_readout_func, method=self.method)
            # retrieve from co-system
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            Control_func2 = self.sol.counter_prop(np.conj(np.flip(Control2, axis=0)))
            self.sol.solve(Control_func2, method=self.method)            
            grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control2/self.sol.gamma)
            #MZ retrieve
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]#/np.sqrt(self.storage_efficiency(self.S, -1))
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(np.conj(np.flip(Control1, axis=0)))
            self.sol.solve(Control_func1, method=self.method)
            grad1 = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control1/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        grads = np.array([grad1, grad2, np.zeros((self.sol.m, 2))])
        return storage_eff, total_eff, grads

    def forward_opt_normal_storage_MZ_readout(self, Controls):
        # Control passed as array, shape (t, p)
        [Control1, Control2, Control3] = Controls
        [eta_loop] = self.extra_params
        if self.sol.protocol == 'Raman':
            # co-propagating
            Control_func1 = self.sol.co_prop(Control1) # func(t, z) => (p)
            self.sol.solve(Control_func1, method=self.method)
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # initial conditions for forward retreival
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval using MZ
            Control_func2 = self.sol.co_prop(Control2)
            self.sol.solve(Control_func2, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy2 = np.array(self.sol.E)
            Scopy2 = np.array(self.sol.S)
            # MZ
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            Control_func3 = self.sol.co_prop(Control3)
            self.sol.solve(Control_func3, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy3 = np.array(self.sol.E)
            Scopy3 = np.array(self.sol.S)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1)
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
            #norm[norm == 0] = 1 
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            Control_func3 = self.sol.co_prop(np.flip(np.conj(Control3), axis=0))
            self.sol.solve(Control_func3, method=self.method)
            grad3 = self.grad([Ecopy3, Scopy3], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control3/self.sol.gamma)
            # MZ store
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            Control_func2 = self.sol.co_prop(np.flip(np.conj(Control2), axis=0))
            self.sol.solve(Control_func2, method=self.method)
            grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control2/self.sol.gamma)
            # retrieve from co-system
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            #self.S[0] = np.conj(np.flip(Scopy[-1], axis=0)) 
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            Control_func1 = self.sol.co_prop(np.conj(np.flip(Control1, axis=0)))
            self.sol.solve(Control_func1, method=self.method)            
            grad1 = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control1/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'ORCA':
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(Control1) # func(t, z) => (p)
            self.sol.solve(Control_func1, method=self.method)
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # initial conditions for forward retreival
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval using MZ
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func2 = self.sol.counter_prop(Control2)
            self.sol.solve(Control_func2, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy2 = np.array(self.sol.E)
            Scopy2 = np.array(self.sol.S)
            # MZ
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func3 = self.sol.counter_prop(Control3)
            self.sol.solve(Control_func3, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy3 = np.array(self.sol.E)
            Scopy3 = np.array(self.sol.S)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1)
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
            #norm[norm == 0] = 1 
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func3 = self.sol.counter_prop(np.flip(np.conj(Control3), axis=0))
            self.sol.solve(Control_func3, method=self.method)
            grad3 = self.grad([Ecopy3, Scopy3], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control3/self.sol.gamma)
            # MZ store
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func2 = self.sol.counter_prop(np.flip(np.conj(Control2), axis=0))
            self.sol.solve(Control_func2, method=self.method)
            grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control2/self.sol.gamma)
            # retrieve from co-system
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            #self.S[0] = np.conj(np.flip(Scopy[-1], axis=0)) 
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(np.conj(np.flip(Control1, axis=0)))
            self.sol.solve(Control_func1, method=self.method)            
            grad1 = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control1/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        grads = np.array([grad1, grad2, grad3])
        return storage_eff, total_eff, grads
    
    def forward_opt_MZ_all4(self, Controls):
        # Control passed as array, shape (t, p)
        [Control1, Control2, Control3, Control4] = Controls
        [eta_loop] = self.extra_params
        if self.sol.protocol == 'Raman':
            # co-propagating
            Control_func1 = self.sol.co_prop(Control1) # func(t, z) => (p)
            self.sol.solve(Control_func1, method=self.method)
            if self.sol.solved:
                Ecopy = np.array(self.sol.E)
                Scopy = np.array(self.sol.S)
                # MZ
                self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]
                self.sol.S[1:] = 0
                Control_func2 = self.sol.co_prop(Control2)
                self.sol.solve(Control_func2, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy2 = np.array(self.sol.E)
                Scopy2 = np.array(self.sol.S)
                storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
                # initial conditions for forward retreival
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                # solve forward retrieval using MZ
                Control_func3 = self.sol.co_prop(Control3)
                self.sol.solve(Control_func3, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy3 = np.array(self.sol.E)
                Scopy3 = np.array(self.sol.S)
                # MZ
                self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]
                self.sol.S[1:] = 0
                Control_func4 = self.sol.co_prop(Control4)
                self.sol.solve(Control_func4, method=self.method)
            if self.sol.solved:
                # need to save P and S coherences for gradient calculation
                Ecopy4 = np.array(self.sol.E)
                Scopy4 = np.array(self.sol.S)
                retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1)
                total_eff = storage_eff * retrieval_eff
                # need to flip Eout and set it as Einit
                # renormalise?
                #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
                #norm[norm == 0] = 1 
                self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
                # initial condition for co-function run
                self.sol.E[:] = 0
                self.sol.S[:] = 0
                # solve co-function storage (time and space reversed), must time reverse control
                Control_func4 = self.sol.co_prop(np.flip(np.conj(Control4), axis=0))
                self.sol.solve(Control_func4, method=self.method)
                grad4 = self.grad([Ecopy4, Scopy4], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control4/self.sol.gamma)
            if self.sol.solved:
                # MZ store
                self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]
                self.sol.S[1:] = 0
                Control_func3 = self.sol.co_prop(np.flip(np.conj(Control3), axis=0))
                self.sol.solve(Control_func3, method=self.method)
                grad3 = self.grad([Ecopy3, Scopy3], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control3/self.sol.gamma)
            if self.sol.solved:
                # retrieve from co-system
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
                #self.S[0] = np.conj(np.flip(Scopy[-1], axis=0)) 
                self.sol.S[1:] = 0
                self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
                Control_func2 = self.sol.co_prop(np.conj(np.flip(Control2, axis=0)))
                self.sol.solve(Control_func2, method=self.method)            
                grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control2/self.sol.gamma)
            if self.sol.solved:    
                #MZ retrieve
                self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]#/np.sqrt(self.storage_efficiency(self.S, -1))
                self.sol.S[1:] = 0
                Control_func1 = self.sol.co_prop(np.conj(np.flip(Control1, axis=0)))
                self.sol.solve(Control_func1, method=self.method)
                grad1 = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control1/self.sol.gamma)
                grads = np.array([grad1, grad2, grad3, grad4])
            else:
                storage_eff = 0
                total_eff = 0
                grads = 0
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        elif self.sol.protocol == 'ORCA' or self.sol.protocol == 'TORCA':
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(Control1) # func(t, z) => (p)
            self.sol.solve(Control_func1, method=self.method)
            Ecopy = np.array(self.sol.E)
            Scopy = np.array(self.sol.S)
            # MZ
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func2 = self.sol.counter_prop(Control2)
            self.sol.solve(Control_func2, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy2 = np.array(self.sol.E)
            Scopy2 = np.array(self.sol.S)
            storage_eff = self.sol.storage_efficiency(self.sol.S, -1)
            # initial conditions for forward retreival
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(storage_eff)
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # solve forward retrieval using MZ
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func3 = self.sol.counter_prop(Control3)
            self.sol.solve(Control_func3, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy3 = np.array(self.sol.E)
            Scopy3 = np.array(self.sol.S)
            # MZ
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func4 = self.sol.counter_prop(Control4)
            self.sol.solve(Control_func4, method=self.method)
            # need to save P and S coherences for gradient calculation
            Ecopy4 = np.array(self.sol.E)
            Scopy4 = np.array(self.sol.S)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, 0, 0) + self.sol.retrieval_efficiency(self.sol.E, 0, 1)
            total_eff = storage_eff * retrieval_eff
            # need to flip Eout and set it as Einit
            # renormalise?
            #norm = np.trapz(np.conj(self.E[:, -1])*self.E[:, -1], x=self.tpoints, axis=0)
            #norm[norm == 0] = 1 
            self.sol.Einits = interp1d(self.sol.tpoints, np.conj(np.flip(-self.sol.E[:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            # initial condition for co-function run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            # solve co-function storage (time and space reversed), must time reverse control
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func4 = self.sol.counter_prop(np.flip(np.conj(Control4), axis=0))
            self.sol.solve(Control_func4, method=self.method)
            grad4 = self.grad([Ecopy4, Scopy4], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control4/self.sol.gamma)
            # MZ store
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func3 = self.sol.counter_prop(np.flip(np.conj(Control3), axis=0))
            self.sol.solve(Control_func3, method=self.method)
            grad3 = self.grad([Ecopy3, Scopy3], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control3/self.sol.gamma)
            # retrieve from co-system
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]/np.sqrt(self.sol.storage_efficiency(self.sol.S, -1))
            #self.S[0] = np.conj(np.flip(Scopy[-1], axis=0)) 
            self.sol.S[1:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func2 = self.sol.counter_prop(np.conj(np.flip(Control2, axis=0)))
            self.sol.solve(Control_func2, method=self.method)            
            grad2 = self.grad([Ecopy2, Scopy2], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control2/self.sol.gamma)
            #MZ retrieve
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol.E[:, -1]*np.sqrt(eta_loop), axis=0, fill_value="extrapolate", bounds_error=False)
            self.sol.E[:] = 0
            self.sol.S[0] = self.sol.S[-1]#/np.sqrt(self.storage_efficiency(self.S, -1))
            self.sol.S[1:] = 0
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(np.conj(np.flip(Control1, axis=0)))
            self.sol.solve(Control_func1, method=self.method)
            grad1 = self.grad([Ecopy, Scopy], [np.conj(np.flip(np.flip(-self.sol.E, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.S, axis=0), axis=1))], Control1/self.sol.gamma)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
        #grads = np.array([grad1, grad2, grad3, grad4])
        return storage_eff, total_eff, grads

    def forward_retrieval_opt_dressing_field(self, Controls):
        if self.sol.protocol == '4levelTORCAP':
            # Control1 should read in and read out
            # Control2 is for dephasing protection 
            [Control1, Control2] = Controls
            #self.field = 2 # optimise second field
            # counter-propagating
            # need to reset detunings
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(Control1) # func(t, z) => (p)
            Control_func2 = self.sol.co_prop(Control2) # func(t, z) => (p)
            self.sol.solve(np.array([Control_func1, Control_func2]), method=self.method)
            # need to save P and S coherences for gradient calculation
            Sgscopy = np.array(self.sol.Sgs)
            Sgbcopy = np.array(self.sol.Sgb)
            Pescopy = np.array(self.sol.Pes)
            Pebcopy = np.array(self.sol.Peb)
            # need readout window to define initial condition for cofunctions
            [tstart] = self.extra_params
            index = np.abs(self.sol.tpoints - tstart).argmin()
            storage_eff = self.sol.storage_efficiency(self.sol.Sgs, index)
            retrieval_eff = self.sol.retrieval_efficiency(self.sol.E, index, 0) + self.sol.retrieval_efficiency(self.sol.E, index, 1)
            self.sol.Einits = interp1d(self.sol.tpoints[0:self.sol.m - index], np.conj(np.flip(-self.sol.E[index:, -1], axis=0))/np.sqrt(retrieval_eff), axis=0, fill_value=0.0, bounds_error=False) # initial photon condition, in natural units
            self.sol.E[:] = 0
            self.sol.Sgs[:] = 0
            self.sol.Sgb[:] = 0
            self.sol.Pes[:] = 0
            self.sol.Peb[:] = 0
            self.sol.detunings(self.sol.deltaS, self.sol.deltaC)
            Control_func1 = self.sol.counter_prop(np.flip(np.conj(Control1), axis=0)) # func(t, z) => (p)
            Control_func2 = self.sol.co_prop(np.flip(np.conj(Control2), axis=0)) # func(t, z) => (p)
            self.sol.solve(np.array([Control_func1, Control_func2]), method=self.method)
            grad = self.grad([Sgscopy, Sgbcopy, Pescopy, Pebcopy], 
                        [np.conj(np.flip(np.flip(self.sol.Sgs, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.Sgb, axis=0), axis=1)),
                        np.conj(np.flip(np.flip(self.sol.Pes, axis=0), axis=1)), np.conj(np.flip(np.flip(self.sol.Peb, axis=0), axis=1))], Controls,
                        field=1)
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.Sgs[:] = 0
            self.sol.Sgb[:] = 0
            self.sol.Pes[:] = 0
            self.sol.Peb[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, self.sol._Einits/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False) # initial photon condition, in natural units
            grads = np.array([np.zeros((self.sol.m, 2)), grad]) # set gradient for control1 to be zero
        return storage_eff, retrieval_eff, grads
    
    def forward_retrieval_opt_perceptron(self, Einits_combined, Controls_combined, target_output):
        if self.sol.protocol == 'Raman':
            #[Controls, Control_readout] = Controls_combined
            Eforward_combined = np.zeros((len(Controls_combined), *self.sol.E.shape), dtype=complex) # keep track of all E
            Sforward_combined = np.zeros((len(Controls_combined), *self.sol.S.shape), dtype=complex) # keep track of all S
            Ebackward_combined = np.zeros((len(Controls_combined), *self.sol.E.shape), dtype=complex) # keep track of all E
            Sbackward_combined = np.zeros((len(Controls_combined), *self.sol.S.shape), dtype=complex) # keep track of all S
            for bin_i, Control in enumerate(Controls_combined):
                # co-propagating
                self.sol.Einits = interp1d(self.sol.tpoints, Einits_combined[bin_i]/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False)
                Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
                self.sol.solve(Control_func, method=self.method)
                # need to save P and S coherences for gradient calculation
                Eforward_combined[bin_i] = np.array(self.sol.E)
                Sforward_combined[bin_i] = np.array(self.sol.S)
                # initial conditions for next bin
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]#/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
            
            # after iterating through all bins, now do backpropagation
            # find cost - make more general for full manifold!
            stored = np.trapz(np.einsum('v, zv -> z', np.sqrt(self.sol.MB(self.sol.vs)*self.sol.dvs), 
                                               np.abs(self.sol.S[0, :, 0, 0, 0, 0, :])), x=self.sol.zCheby)
            cost = pow( target_output - np.trapz(np.einsum('v, zv -> z', np.sqrt(self.sol.MB(self.sol.vs)*self.sol.dvs), 
                                               np.abs(self.sol.S[0, :, 0, 0, 0, 0, :])), x=self.sol.zCheby), 2)
            # boundary condition for backpropagation
            # do we need to normalise somthing here? What bounds should be on size of target_output
            self.sol.S[0] = np.abs(np.flip(self.sol.S[0], axis=0) - 2*target_output) # reverse space

            Controls_reversed = np.flip(Controls_combined, axis=0) # reverse order and need to reverse each one in time in for loop
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            for bin_i, Control in enumerate(Controls_reversed):
                # co-propagating
                Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0)) # func(t, z) => (p)
                self.sol.solve(Control_func, method=self.method)
                # need to save P and S coherences for gradient calculation
                Ebackward_combined[bin_i] = np.array(self.sol.E)
                Sbackward_combined[bin_i] = np.array(self.sol.S)
                # initial conditions for next bin
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]#/np.sqrt(storage_eff)
                self.sol.S[1:] = 0

            # want gradient for control amplitudes - find gradient w.r.t. control field, then convolve with control field with amplitude=1
            #grad_amp = np.zeros((len(Controls_combined), 2)) # 2 is for polarsiations - assumed to be three level system for now
            grads = np.zeros((len(Controls_combined), self.sol.m, 2))
            Ebackward_combined = np.flip(Ebackward_combined, axis=0)
            Sbackward_combined = np.flip(Sbackward_combined, axis=0)
            for bin_i, Control in enumerate(Controls_combined):
                #Control_normed = Control[:, 0]/max(Control[:, 0])
                grad = self.grad([Eforward_combined[bin_i], Sforward_combined[bin_i]], 
                                 [np.conj(np.flip(np.flip(-Ebackward_combined[bin_i], axis=0), axis=1)), 
                                  np.conj(np.flip(np.flip(Sbackward_combined[bin_i], axis=0), axis=1))], Control/self.sol.gamma)
            
                #grad_amp[bin_i, 0] = -np.trapz(Control_normed*grad[:, 0], x=self.sol.tpoints) # negative as trying to minimise
                grads[bin_i] = -grad
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # Krotov method expects gradients of each control to be same length as time axis and have polarisation
            # assume three level system for now
            #grads = np.repeat(grad_amp, self.sol.m).reshape(len(grad_amp), 2, self.sol.m).transpose(0, 2, 1)
            return stored, cost, grads # set one of efficiencies to be returned as 0
        
    def forward_retrieval_opt_perceptron_amp(self, Einits_combined, Controls_combined, target_output):
        if self.sol.protocol == 'Raman':
            #[Controls, Control_readout] = Controls_combined
            Eforward_combined = np.zeros((len(Controls_combined), *self.sol.E.shape), dtype=complex) # keep track of all E
            Sforward_combined = np.zeros((len(Controls_combined), *self.sol.S.shape), dtype=complex) # keep track of all S
            Ebackward_combined = np.zeros((len(Controls_combined), *self.sol.E.shape), dtype=complex) # keep track of all E
            Sbackward_combined = np.zeros((len(Controls_combined), *self.sol.S.shape), dtype=complex) # keep track of all S
            for bin_i, Control in enumerate(Controls_combined):
                # co-propagating
                self.sol.Einits = interp1d(self.sol.tpoints, Einits_combined[bin_i]/np.sqrt(self.sol.gamma), axis=0, fill_value="extrapolate", bounds_error=False)
                Control_func = self.sol.co_prop(Control) # func(t, z) => (p)
                self.sol.solve(Control_func, method=self.method)
                # need to save P and S coherences for gradient calculation
                Eforward_combined[bin_i] = np.array(self.sol.E)
                Sforward_combined[bin_i] = np.array(self.sol.S)
                # initial conditions for next bin
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]#/np.sqrt(storage_eff)
                self.sol.S[1:] = 0
            
            # after iterating through all bins, now do backpropagation
            # find cost - make more general for full manifold!
            stored = np.trapz(np.einsum('v, zv -> z', np.sqrt(self.sol.MB(self.sol.vs)*self.sol.dvs), 
                                               np.abs(self.sol.S[0, :, 0, 0, 0, 0, :])), x=self.sol.zCheby)
            cost = pow( target_output - np.trapz(np.einsum('v, zv -> z', np.sqrt(self.sol.MB(self.sol.vs)*self.sol.dvs), 
                                               np.abs(self.sol.S[0, :, 0, 0, 0, 0, :])), x=self.sol.zCheby), 2)
            # boundary condition for backpropagation
            # do we need to normalise somthing here? What bounds should be on size of target_output
            self.sol.S[0] = np.abs(np.flip(self.sol.S[0], axis=0) - 2*target_output) # reverse space

            Controls_reversed = np.flip(Controls_combined, axis=0) # reverse order and need to reverse each one in time in for loop
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            for bin_i, Control in enumerate(Controls_reversed):
                # co-propagating
                Control_func = self.sol.co_prop(np.flip(np.conj(Control), axis=0)) # func(t, z) => (p)
                self.sol.solve(Control_func, method=self.method)
                # need to save P and S coherences for gradient calculation
                Ebackward_combined[bin_i] = np.array(self.sol.E)
                Sbackward_combined[bin_i] = np.array(self.sol.S)
                # initial conditions for next bin
                self.sol.E[:] = 0
                self.sol.S[0] = self.sol.S[-1]#/np.sqrt(storage_eff)
                self.sol.S[1:] = 0

            # want gradient for control amplitudes - find gradient w.r.t. control field, then convolve with control field with amplitude=1
            grad_amp = np.zeros((len(Controls_combined), 2)) # 2 is for polarsiations - assumed to be three level system for now
            Ebackward_combined = np.flip(Ebackward_combined, axis=0)
            Sbackward_combined = np.flip(Sbackward_combined, axis=0)
            for bin_i, Control in enumerate(Controls_combined):
                Control_normed = Control[:, 0]/max(Control[:, 0])
                grad = self.grad([Eforward_combined[bin_i], Sforward_combined[bin_i]], 
                                 [np.conj(np.flip(np.flip(-Ebackward_combined[bin_i], axis=0), axis=1)), 
                                  np.conj(np.flip(np.flip(Sbackward_combined[bin_i], axis=0), axis=1))], Control/self.sol.gamma)
            
                grad_amp[bin_i, 0] = -np.trapz(Control_normed*grad[:, 0], x=self.sol.tpoints) # negative as trying to minimise
            # set up boundary conditions for next run
            self.sol.E[:] = 0
            self.sol.S[:] = 0
            self.sol.Einits = interp1d(self.sol.tpoints, np.zeros(self.sol.m)[:, None]*np.array([1, 0]), axis=0, fill_value="extrapolate", bounds_error=False)
            # Krotov method expects gradients of each control to be same length as time axis and have polarisation
            # assume three level system for now
            grads = np.repeat(grad_amp, self.sol.m).reshape(len(grad_amp), 2, self.sol.m).transpose(0, 2, 1)
            return stored, cost, grads

    def Krotov(self, function, Controls, initial_step_sizes, tol, adaptive_step_size=False, max_energies=False, max_powers=np.array([False]), extra_params=False, max_step_size = False):
        count = 0
        satisfied = True # used for adaptive step size, otherwise should always remain true
        created_plots_bool = False
        effs = np.zeros(4)
        if self._complex:
            dtype = complex
        else:
            dtype = float
        Control_number = len(Controls)
        Control_hist = np.zeros((5, Control_number, self.sol.m, 2), dtype=dtype) # (count, number of controls, time steps, polarisation)
        Control_hist[0] = Controls
        grad_hist = np.zeros((4, Control_number, self.sol.m, 2), dtype=dtype)
        step_sizes = self.set_initial_step_sizes(initial_step_sizes, Control_number)
        self.arrays_for_counter_prop() # does nothing for co_prop
        if max_energies:
            adaptive_step_size = False
        self.extra_params = extra_params
        while count < 4:
            storage_eff, effs[count], grad_hist[count] = function(Control_hist[count])
            if count<1:
                grad_norms_hist = []
            grad_norms = np.linalg.norm(grad_hist[count], axis=1)
            grad_norms_hist.append(grad_norms)
            if self.verbose:
                grad_norms = np.linalg.norm(grad_hist[count], axis=1)
                print("Count: %d, Step sizes: %s, Storage efficiency: %f, Total efficiency: %f, Gradient norms: %s" % 
                        (count, step_sizes, storage_eff, effs[count], grad_norms))
            if self.live_plot or self.save_to_file:
                if count<1:
                    storage_eff_hist = []
                    total_eff_hist = []
                    #grad_norms_hist = []
                    step_sizes_hist = [] # step_sizes history
                    count_hist = []

                    if self.live_plot & ~created_plots_bool:
                        self.create_plots(Control_number)
                        created_plots_bool = True
                    if self.save_to_file:
                        filename = self.generate_filename(self.sol.metadata())

                
                # update history lists
                #grad_norms = np.linalg.norm(grad_hist[count], axis=1)
                storage_eff_hist.append(storage_eff)
                total_eff_hist.append(effs[count])
                #grad_norms_hist.append(grad_norms)
                step_sizes_hist.append(step_sizes)
                count_hist.append(count)

                if self.live_plot and self.sol.solved:
                    self.update_plots(Control_number, [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_hist[count]])


            if adaptive_step_size:
                if count<1:
                    self.initial_step_sizes = step_sizes
                    if self.adapt_method == 'Adam':
                        self.ma = np.zeros((Control_number, 2)) # first moment
                        self.va = np.zeros((Control_number, 2)) # second moment
                    elif self.adapt_method == 'Backtrack' or self.adapt_method == 'Wolfe':
                        step_sizes, satisfied = self.adapt_step(step_sizes, grad_hist, effs, grad_norms_hist, count, max_step_size)
                else:
                    # doesn't invoke the people's elbow on the first 4 steps
                    step_sizes, satisfied = self.adapt_step(step_sizes, grad_hist, effs, grad_norms_hist, count, max_step_size)
                    # adapt_step needs to deal with self.sol.solved == False
            if self.sol.solved == False:
                break        
            if satisfied:
                if max_energies:
                    #mu = np.sqrt(np.trapz(pow(np.abs(grad_hist[count][:, :, 0]) + np.abs(grad_hist[count][:, :, 1]),2)*self.sol.gamma, x=self.sol.tpoints, axis=1)/max_energies) # shape = (c,)
                    Control_hist[count+1] = Control_hist[count] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[count]*self.sol.gamma)
                    Ecs = np.trapz(pow(np.abs(Control_hist[count+1][:, :, 0]) + np.abs(Control_hist[count+1][:, :, 1]),2), x=self.sol.tpoints/self.sol.gamma, axis=1) # shape = (c,)
                    mu = 1-np.sqrt(max_energies/Ecs)
                    mu[mu < 0] = 0 # was energy of control less than max_energies => means control pulse is allowed to have less energy
                    Control_hist[count+1] = Control_hist[count+1] - np.einsum('ctp, c -> ctp', Control_hist[count+1], mu)
                    #grad_hist[count] = grad_hist[count] - np.einsum('ctp, c -> ctp', Control_hist[count+1]/self.sol.gamma, mu)
                    #Control_hist[count+1] = np.einsum('ctp, c -> ctp', grad_hist[count]*self.sol.gamma, 1/mu)
                else:
                    Control_hist[count+1] = Control_hist[count] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[count]*self.sol.gamma)
                count+=1
            else:
                Control_hist[count] = Control_hist[count-1] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[count-1]*self.sol.gamma)
            if max_powers.any():
                # if total power over both polarisations > max power 
                factors = np.einsum('ct, c -> ct', np.abs(np.sum(Control_hist[count], axis=-1)), 1/max_powers)
                mask = factors > 1
                Control_hist[count][mask, 0] = Control_hist[count][mask, 0]/factors[mask]
                Control_hist[count][mask, 1] = Control_hist[count][mask, 1]/factors[mask]


        # now have 5 initial controls, 4 effs and grads, count = 4
        eff_average = (effs[2] + effs[1] + effs[0])/3
        diff = (effs[3] - eff_average)/eff_average
        below_max_points = True         
        if self.sol.solved:
            while (diff > tol) and below_max_points:
                Control_hist = np.roll(Control_hist, -1, axis=0)
                eff0 = effs[0] # need to keep track of for averaging, in case fail Wolfe conditions
                effs = np.roll(effs, -1, axis=0)
                grad_hist = np.roll(grad_hist, -1, axis=0)
                storage_eff, effs[3], grad_hist[3] = function(Control_hist[3])
                grad_norms = np.linalg.norm(grad_hist[3], axis=1)
                grad_norms_hist.append(grad_norms)
                if self.verbose:
                    grad_norms = np.linalg.norm(grad_hist[3], axis=1)
                    print("Count: %d, Step sizes: %s, Storage efficiency: %f, Total efficiency: %f, Gradient norms: %s" % 
                            (count, step_sizes, storage_eff, effs[3], grad_norms))
                if self.live_plot or self.save_to_file:
                    # update history lists
                    #grad_norms = np.linalg.norm(grad_hist[3], axis=1)
                    storage_eff_hist.append(storage_eff)
                    total_eff_hist.append(effs[3])
                    #grad_norms_hist.append(grad_norms)
                    step_sizes_hist.append(step_sizes)
                    count_hist.append(count)

                    if self.live_plot and self.sol.solved == True:
                        self.update_plots(Control_number, [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_hist[3]])

                if adaptive_step_size:
                    step_sizes, satisfied = self.adapt_step(step_sizes, grad_hist, effs, grad_norms_hist, count, max_step_size)
                    # adapt_step needs to deal with self.sol.solved == False
                if self.sol.solved == False:
                    break   
                if satisfied:
                    if max_energies:
                        #mu = np.sqrt(np.trapz(pow(np.abs(grad_hist[3][:, :, 0]) + np.abs(grad_hist[3][:, :, 1]),2)*self.sol.gamma, x=self.sol.tpoints, axis=1)/max_energies) # shape = (c,)
                        Control_hist[4] = Control_hist[3] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[3]*self.sol.gamma)
                        Ecs = np.trapz(pow(np.abs(Control_hist[4][:, :, 0]) + np.abs(Control_hist[4][:, :, 1]),2), x=self.sol.tpoints/self.sol.gamma, axis=1) # shape = (c,)
                        mu = 1-np.sqrt(max_energies/Ecs)
                        mu[mu < 0] = 0 # was energy of control less than max_energies => means control pulse is allowed to have less energy
                        Control_hist[4] = Control_hist[4] - np.einsum('ctp, c -> ctp', Control_hist[4], mu)
                        #grad_hist[3] = grad_hist[3] - np.einsum('ctp, c -> ctp', Control_hist[4]/self.sol.gamma, mu)
                        #Control_hist[4] = np.einsum('ctp, c -> ctp', grad_hist[3]*self.sol.gamma, 1/mu)
                    else:
                        Control_hist[4] = Control_hist[3] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[3]*self.sol.gamma)
                    count+=1
                else:
                    Control_hist = np.roll(Control_hist, 1, axis=0)
                    effs = np.roll(effs, 1, axis=0)
                    effs[0] = eff0
                    grad_hist = np.roll(grad_hist, 1, axis=0)
                    Control_hist[4] = Control_hist[3] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[3]*self.sol.gamma)
                if max_powers.any():
                    # if total power over both polarisations > max power 
                    factors = np.einsum('ct, c -> ct', np.abs(np.sum(Control_hist[4], axis=-1)), 1/max_powers)
                    mask = factors > 1
                    Control_hist[4][mask, 0] = Control_hist[4][mask, 0]/factors[mask]
                    Control_hist[4][mask, 1] = Control_hist[4][mask, 1]/factors[mask]
                
                eff_average = (effs[2] + effs[1] + effs[0])/3
                diff = (effs[3] - eff_average)/eff_average

                if self.max_points:
                    if count >= self.max_points:
                        below_max_points = False
        
        index = np.argmax(effs)
        if self.live_plot: #?
            self.update_plots(Control_number, [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_hist[index]])
        if self.save_to_file:
            # Save the arrays to a file
            np.savez(filename, Control_number=Control_number, storage_eff_hist=storage_eff_hist, total_eff_hist=total_eff_hist, grad_norms_hist=grad_norms_hist, 
                    step_sizes_hist=step_sizes_hist, count_hist=count_hist, Control_best=Control_hist[index])
            
            # can now load using np.load and the variables will be saved under the identifiers above
        return effs[index], Control_hist[index], (count - 4 + index)
    
    def Krotov_perceptron(self, function, Einit_base, Controls, initial_step_sizes, tol, inputs, outputs, target_scaling = 1.0, adaptive_step_size=False, max_energies=False, max_powers=np.array([False]), extra_params=False):
        # input/output pairs
        # Einit_base is the base unit of input, for amplitude=1
        # from inputs need to make Einits
        # derive Einits from inputs and Einit_base
        # Controls are initialisations of the weights
        # inputs[index] and Controls need to be same length
        # from outputs, derive targets, some normalisation to be done here w.r.t. Einits
        
        count = 0
        inner_count = 4
        satisfied = True # used for adaptive step size, otherwise should always remain true
        effs = np.zeros(4)
        if self._complex:
            dtype = complex
        else:
            dtype = float
        Control_number = len(Controls)
        Control_hist = np.zeros((5, Control_number, self.sol.m, 2), dtype=dtype) # (count, number of controls, time steps, polarisation)
        Control_hist[0] = Controls
        grad_hist = np.zeros((4, Control_number, self.sol.m, 2), dtype=dtype)
        step_sizes = self.set_initial_step_sizes(initial_step_sizes, Control_number)
        self.arrays_for_counter_prop() # does nothing for co_prop
        if max_energies:
            adaptive_step_size = False
        self.extra_params = extra_params
        while count < 4:
            inner_storage_eff  = np.zeros(inner_count)
            inner_eff  = np.zeros(inner_count)
            inner_grad  = np.zeros((inner_count, Control_number, self.sol.m, 2), dtype=dtype)
            for i in range(0, inner_count):
                # randomly sample from inputs
                index = np.random.randint(len(inputs), size=1)
                Einits = np.einsum('i, tp -> itp', inputs[index][0], Einit_base)
                # scale target w.r.t. integrated amplitude of Einits
                area = np.einsum('ip -> ', np.trapz(Einits/np.sqrt(self.sol.gamma), x=self.sol.tpoints, axis=1))
                target = outputs[index]*area*target_scaling
                # find weight gradients
                inner_storage_eff[i], inner_eff[i], inner_grad[i] = function(Einits, Control_hist[count], target)
            
            storage_eff = np.mean(inner_storage_eff)
            effs[count] =  np.mean(inner_eff)
            grad_hist[count] = np.mean(inner_grad, axis=0)
            if count<1:
                grad_norms_hist = []
            grad_norms = np.linalg.norm(grad_hist[count], axis=1)
            grad_norms_hist.append(grad_norms)
            if self.verbose:
                grad_norms = np.linalg.norm(grad_hist[count], axis=1)
                print("Count: %d, Step sizes: %s, Storage efficiency: %f, Total efficiency: %f, Gradient norms: %s" % 
                        (count, step_sizes, storage_eff, effs[count], grad_norms))
            if self.live_plot or self.save_to_file:
                if count<1:
                    storage_eff_hist = []
                    total_eff_hist = []
                    #grad_norms_hist = []
                    step_sizes_hist = [] # step_sizes history
                    count_hist = []

                    if self.live_plot:
                        self.create_plots(Control_number)
                    if self.save_to_file:
                        filename = self.generate_filename(self.sol.metadata())

                
                # update history lists
                #grad_norms = np.linalg.norm(grad_hist[count], axis=1)
                storage_eff_hist.append(storage_eff)
                total_eff_hist.append(effs[count])
                #grad_norms_hist.append(grad_norms)
                step_sizes_hist.append(step_sizes)
                count_hist.append(count)

                if self.live_plot:
                    self.update_plots(Control_number, [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_hist[count]])


            if adaptive_step_size:
                if count<1:
                    self.initial_step_sizes = step_sizes
                    if self.adapt_method == 'Adam':
                        self.ma = np.zeros((Control_number, 2)) # first moment
                        self.va = np.zeros((Control_number, 2)) # second moment
                else:
                    # doesn't invoke the people's elbow on the first 4 steps
                    step_sizes, satisfied = self.adapt_step(step_sizes, [grad_hist[count], grad_hist[count-1]], [effs[count], effs[count-1]], grad_norms_hist)
                    
            if satisfied:
                if max_energies:
                    #mu = np.sqrt(np.trapz(pow(np.abs(grad_hist[count][:, :, 0]) + np.abs(grad_hist[count][:, :, 1]),2)*self.sol.gamma, x=self.sol.tpoints, axis=1)/max_energies) # shape = (c,)
                    Control_hist[count+1] = Control_hist[count] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[count]*self.sol.gamma)
                    Ecs = np.trapz(pow(np.abs(Control_hist[count+1][:, :, 0]) + np.abs(Control_hist[count+1][:, :, 1]),2), x=self.sol.tpoints/self.sol.gamma, axis=1) # shape = (c,)
                    mu = 1-np.sqrt(max_energies/Ecs)
                    mu[mu < 0] = 0 # was energy of control less than max_energies => means control pulse is allowed to have less energy
                    Control_hist[count+1] = Control_hist[count+1] - np.einsum('ctp, c -> ctp', Control_hist[count+1], mu)
                    #grad_hist[count] = grad_hist[count] - np.einsum('ctp, c -> ctp', Control_hist[count+1]/self.sol.gamma, mu)
                    #Control_hist[count+1] = np.einsum('ctp, c -> ctp', grad_hist[count]*self.sol.gamma, 1/mu)
                else:
                    Control_hist[count+1] = Control_hist[count] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[count]*self.sol.gamma)
                count+=1
            else:
                Control_hist[count] = Control_hist[count-1] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[count]*self.sol.gamma)
            if max_powers.any():
                # if total power over both polarisations > max power 
                factors = np.einsum('ct, c -> ct', np.abs(np.sum(Control_hist[count], axis=-1)), 1/max_powers)
                mask = factors > 1
                Control_hist[count][mask, 0] = Control_hist[count][mask, 0]/factors[mask]
                Control_hist[count][mask, 1] = Control_hist[count][mask, 1]/factors[mask]


        # now have 5 initial controls, 4 effs and grads, count = 4
        eff_average = (effs[2] + effs[1] + effs[0])/3
        diff = tol*1.1 #(effs[3] - eff_average)/eff_average
        below_max_points = True            
        while (diff > tol) and below_max_points:
            Control_hist = np.roll(Control_hist, -1, axis=0)
            eff0 = effs[0] # need to keep track of for averaging, in case fail Wolfe conditions
            effs = np.roll(effs, -1, axis=0)
            grad_hist = np.roll(grad_hist, -1, axis=0)

            inner_storage_eff  = np.zeros(inner_count)
            inner_eff  = np.zeros(inner_count)
            inner_grad  = np.zeros((inner_count, Control_number, self.sol.m, 2), dtype=dtype)
            for i in range(0, inner_count):
                # randomly sample from inputs
                index = np.random.randint(len(inputs), size=1)
                Einits = np.einsum('i, tp -> itp', inputs[index][0], Einit_base)
                # scale target w.r.t. integrated amplitude of Einits
                area = np.einsum('ip -> ', np.trapz(Einits/np.sqrt(self.sol.gamma), x=self.sol.tpoints, axis=1))
                target = outputs[index]*area*target_scaling
                # find weight gradients
                inner_storage_eff[i], inner_eff[i], inner_grad[i] = function(Einits, Control_hist[3], target)
            
            storage_eff = np.mean(inner_storage_eff)
            effs[3] =  np.mean(inner_eff)
            grad_hist[3] = np.mean(inner_grad, axis=0)

            grad_norms = np.linalg.norm(grad_hist[3], axis=1)
            grad_norms_hist.append(grad_norms)
            if self.verbose:
                grad_norms = np.linalg.norm(grad_hist[3], axis=1)
                print("Count: %d, Step sizes: %s, Storage efficiency: %f, Total efficiency: %f, Gradient norms: %s" % 
                        (count, step_sizes, storage_eff, effs[3], grad_norms))
            if self.live_plot or self.save_to_file:
                # update history lists
                #grad_norms = np.linalg.norm(grad_hist[3], axis=1)
                storage_eff_hist.append(storage_eff)
                total_eff_hist.append(effs[3])
                #grad_norms_hist.append(grad_norms)
                step_sizes_hist.append(step_sizes)
                count_hist.append(count)

                if self.live_plot:
                    self.update_plots(Control_number, [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_hist[3]])

            if adaptive_step_size:
                step_sizes, satisfied = self.adapt_step(step_sizes, [grad_hist[3], grad_hist[2]], [effs[3], effs[2]], grad_norms_hist)
            if satisfied:
                if max_energies:
                    #mu = np.sqrt(np.trapz(pow(np.abs(grad_hist[3][:, :, 0]) + np.abs(grad_hist[3][:, :, 1]),2)*self.sol.gamma, x=self.sol.tpoints, axis=1)/max_energies) # shape = (c,)
                    Control_hist[4] = Control_hist[3] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[3]*self.sol.gamma)
                    Ecs = np.trapz(pow(np.abs(Control_hist[4][:, :, 0]) + np.abs(Control_hist[4][:, :, 1]),2), x=self.sol.tpoints/self.sol.gamma, axis=1) # shape = (c,)
                    mu = 1-np.sqrt(max_energies/Ecs)
                    mu[mu < 0] = 0 # was energy of control less than max_energies => means control pulse is allowed to have less energy
                    Control_hist[4] = Control_hist[4] - np.einsum('ctp, c -> ctp', Control_hist[4], mu)
                    #grad_hist[3] = grad_hist[3] - np.einsum('ctp, c -> ctp', Control_hist[4]/self.sol.gamma, mu)
                    #Control_hist[4] = np.einsum('ctp, c -> ctp', grad_hist[3]*self.sol.gamma, 1/mu)
                else:
                    Control_hist[4] = Control_hist[3] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[3]*self.sol.gamma)
                count+=1
            else:
                Control_hist = np.roll(Control_hist, 1, axis=0)
                effs = np.roll(effs, 1, axis=0)
                effs[0] = eff0
                grad_hist = np.roll(grad_hist, 1, axis=0)
                Control_hist[4] = Control_hist[3] + np.einsum('cp, ctp -> ctp', step_sizes, grad_hist[3]*self.sol.gamma)
            if max_powers.any():
                # if total power over both polarisations > max power 
                factors = np.einsum('ct, c -> ct', np.abs(np.sum(Control_hist[4], axis=-1)), 1/max_powers)
                mask = factors > 1
                Control_hist[4][mask, 0] = Control_hist[4][mask, 0]/factors[mask]
                Control_hist[4][mask, 1] = Control_hist[4][mask, 1]/factors[mask]
            
            eff_average = (effs[2] + effs[1] + effs[0])/3
            diff = tol*1.1 #(effs[3] - eff_average)/eff_average

            if self.max_points:
                if count >= self.max_points:
                    below_max_points = False
        
        index = np.argmax(effs)
        if self.live_plot: #?
            self.update_plots(Control_number, [storage_eff_hist, total_eff_hist, grad_norms_hist, step_sizes_hist, count_hist, Control_hist[index]])
        if self.save_to_file:
            # Save the arrays to a file
            np.savez(filename, Control_number=Control_number, storage_eff_hist=storage_eff_hist, total_eff_hist=total_eff_hist, grad_norms_hist=grad_norms_hist, 
                     step_sizes_hist=step_sizes_hist, count_hist=count_hist, Control_best=Control_hist[index])
            
            # can now load using np.load and the variables will be saved under the identifiers above

        return effs[index], Control_hist[index], (count - 4 + index)

    def set_initial_step_sizes(self, initial_step_sizes, Control_number):
        step_sizes = np.zeros((Control_number, 2))
        if isinstance(initial_step_sizes, int) or isinstance(initial_step_sizes, float):
            step_sizes[:] = initial_step_sizes
        elif len(initial_step_sizes) == 1:
            step_sizes[:] = initial_step_sizes[0]
        elif len(initial_step_sizes) == 2:
            step_sizes[:, 0] = initial_step_sizes[0]
            step_sizes[:, 1] = initial_step_sizes[1]
        elif len(initial_step_sizes) == Control_number:
            step_sizes[:, 0] = initial_step_sizes
            step_sizes[:, 1] = initial_step_sizes
        elif np.array(initial_step_sizes).shape == step_sizes.shape:
            step_sizes = initial_step_sizes
        return step_sizes

    def adapt_step(self, step_sizes, grads, effs, grad_local_hist, count, max_step_size = False):
        if self.adapt_method == 'Adam':
            step_sizes, satisfied = self.adapt_step_Adam(step_sizes, grads, effs, grad_local_hist, count)
        elif self.adapt_method == 'Wolfe':
            step_sizes, satisfied = self.adapt_step_Wolfe(step_sizes, grads, effs, grad_local_hist, count)
        elif self.adapt_method == 'Backtrack':
            step_sizes, satisfied = self.adapt_step_backtrack(step_sizes, grads, effs, grad_local_hist, count)
        if max_step_size:
            step_sizes[step_sizes > max_step_size] = max_step_size
        return step_sizes, satisfied

    def adapt_step_Adam(self, step_sizes, grads, effs, grad_local_hist, count):
        """
        Adapt the step size using an Adam-based momentum approach.
        """
        satisfied = True
        eff_current = effs[count]
        eff_prev = effs[count-1]
        grad_current = grads[count]
        grad_prev = grads[count-1]
        # [eff_current, eff_prev] = effs
        # [grad_current, grad_prev] = grads
        step_sizes_out = np.array(step_sizes)
        if len(grad_local_hist) == 2:
            hist = [np.zeros(grad_local_hist[0].shape), grad_local_hist[0], grad_local_hist[1]]
        elif len(grad_local_hist) == 1:
            hist = [np.zeros(grad_local_hist[0].shape), np.zeros(grad_local_hist[0].shape), grad_local_hist[0]]
        else: # length of 3
            hist = grad_local_hist
        # check if step size too big
        # use two conditions
        # the people's elbow
        diff0 = hist[-2] - hist[-3]
        diff1 = hist[-1] - hist[-2]
        mask = (diff0 < 0) * (diff1 > 0)
        # 1st Wolfe condition
        eff_current_array = np.full(step_sizes.shape, eff_current)
        eff_prev_array = np.full(step_sizes.shape, eff_prev)
        too_large = (eff_current_array < (eff_prev_array + self.c1*step_sizes*np.linalg.norm(grad_prev, axis=1))) # check efficiency reduced sufficiently -> places upper bound
        if mask.any() or too_large.any():
            step_sizes_out[mask] = self.down_factor * step_sizes[mask]
            step_sizes_out[too_large] = self.down_factor * step_sizes[too_large]
            satisfied = False
        if satisfied:
            # Compute gradient norms
            grad_norm = np.linalg.norm(grad_current, axis=1) # shape =  (len(Controls), p)

            # Update first and second moments
            self.ma = self.beta1 * self.ma + (1 - self.beta1) * grad_norm
            self.va = self.beta2 * self.va + (1 - self.beta2) * pow(grad_norm, 2)

            # Bias correction
            m_hat = self.ma / (1 - self.beta1)
            v_hat = self.va / (1 - self.beta2)

            # Compute step size
            step_sizes_out = step_sizes * m_hat / (np.sqrt(v_hat) + self.epsilon)
        if (step_sizes_out < self.step_tol*self.initial_step_sizes).all(): # step size has gotten below tolerance
            satisfied = True
        return step_sizes_out, satisfied

    # def adapt_step_Wolfe(self, step_sizes, grads, effs, grad_local_hist, count):
    #     """
    #     Adapt the step size using an Wolfe conditions based approach.
    #     """
    #     eff_current = effs[count]
    #     eff_prev = effs[count-1]
    #     grad_current = grads[count]
    #     grad_prev = grads[count-1]
    #     # [eff_current, eff_prev] = effs
    #     # [grad_current, grad_prev] = grads
    #     satisfied = True
    #     if self.sol.solved:
    #         step_sizes_out = np.array(step_sizes)
    #         if len(grad_local_hist) == 2:
    #             hist = [np.zeros(grad_local_hist[0].shape), grad_local_hist[0], grad_local_hist[1]]
    #         elif len(grad_local_hist) == 1:
    #             hist = [np.zeros(grad_local_hist[0].shape), np.zeros(grad_local_hist[0].shape), grad_local_hist[0]]
    #         else: # length of 3
    #             hist = grad_local_hist
    #         # the people's elbow
    #         diff0 = hist[-2] - hist[-3]
    #         diff1 = hist[-1] - hist[-2]
    #         mask = (diff0 < 0) * (diff1 > 0)
    #         # 1st Wolfe condition
    #         eff_current_array = np.full(step_sizes.shape, eff_current)
    #         eff_prev_array = np.full(step_sizes.shape, eff_prev)
    #         too_large = (eff_current_array < (eff_prev_array + self.c1*step_sizes*np.linalg.norm(grad_prev, axis=1))) # check efficiency reduced sufficiently -> places upper bound
    #         # combine with the people's elbow
    #         too_large = np.logical_or(too_large, mask)
    #         # print(too_large)
    #         # print()
    #         # 2nd Wolfe condition
    #         too_small = (np.linalg.norm(grad_current, axis=1) > (self.c2*np.linalg.norm(grad_prev, axis=1))) # check gradient reduced sufficiently -> places lower bound
    #         both = too_large*too_small # if step size is both too large and too small, reduce step size?
    #         too_large[both] = True
    #         too_small[both] = False
    #         step_sizes_out[too_large] = step_sizes[too_large] * self.down_factor
    #         step_sizes_out[too_small] = step_sizes[too_small] * self.up_factor
    #         # if too_large.any() or too_small.any():
    #         #     satisfied = False
    #         if too_large.any() or too_small.any():
    #             satisfied = False
    #         if (step_sizes_out < self.step_tol*self.initial_step_sizes).all(): # step size has gotten below tolerance
    #             satisfied = True
    #         if satisfied:
    #             step_sizes_out = self.initial_step_sizes
            
    #     else:
    #         satisfied = False
    #         step_sizes_out = step_sizes * self.down_factor
    #         self.sol.solved = True # to continue in while loop
    #     return step_sizes_out, satisfied
    
    def adapt_step_Wolfe(self, step_sizes, grads, effs, grad_local_hist, count):
        """
        Adapt the step size using an Wolfe conditions based approach.
        """
        if count < 4:
            eff_current = effs[count]
            eff_prev = effs[count-1]
            grad_current = grads[count]
            grad_prev = grads[count-1]
        else:
            eff_current = effs[-1]
            eff_prev = effs[-2]
            grad_current = grads[-1]
            grad_prev = grads[-2]
        # [eff_current, eff_prev] = effs
        # [grad_current, grad_prev] = grads
        satisfied = True
        if self.sol.solved:
            step_sizes_out = np.array(step_sizes)
            # 1st Wolfe condition - reduce step size
            eff_current_array = np.full(step_sizes.shape, eff_current)
            eff_prev_array = np.full(step_sizes.shape, eff_prev)
            too_large = (eff_current_array < (eff_prev_array + self.c1*step_sizes*np.linalg.norm(grad_prev, axis=1))) # check efficiency reduced sufficiently -> places upper bound
            # # if gradient has decreased, don't reduce step size
            # undo_too_large = (np.linalg.norm(grad_current, axis=1) < (np.linalg.norm(grad_prev, axis=1)))
            # too_large[undo_too_large] = False
            # 2nd Wolfe condition - increase step size
            too_small = (np.linalg.norm(grad_current, axis=1) > (self.c2*np.linalg.norm(grad_prev, axis=1))) & (np.linalg.norm(grad_current, axis=1) < (np.linalg.norm(grad_prev, axis=1)))  # check gradient reduced sufficiently -> places lower bound
            both = too_large*too_small # if step size is both too large and too small, reduce step size?
            too_large[both] = True
            too_small[both] = False
            step_sizes_out[too_large] = step_sizes[too_large] * self.down_factor
            step_sizes_out[too_small] = step_sizes[too_small] * self.up_factor
            # if too_large.any() or too_small.any():
            #     satisfied = False
            if too_large.any() or too_small.any():
                satisfied = False
            if (step_sizes_out < self.step_tol*self.initial_step_sizes).all(): # step size has gotten below tolerance
                satisfied = True
            if satisfied:
                step_sizes_out = self.initial_step_sizes
            
        else:
            satisfied = False
            step_sizes_out = step_sizes * self.down_factor
            self.sol.solved = True # to continue in while loop
        return step_sizes_out, satisfied
    
    def adapt_step_backtrack(self, step_sizes, grads, effs, grad_local_hist, count):
        """
        Adapt the step size using an Wolfe conditions based approach.
        """
        if count < 4:
            eff_current = effs[count]
            eff_prev = effs[count-1]
            grad_current = grads[count]
            grad_prev = grads[count-1]
        else:
            eff_current = effs[-1]
            eff_prev = effs[-2]
            grad_current = grads[-1]
            grad_prev = grads[-2]
        # [eff_current, eff_prev] = effs
        # [grad_current, grad_prev] = grads
        satisfied = True
        if self.sol.solved:
            step_sizes_out = np.array(step_sizes)
            # 1st Wolfe condition
            eff_current_array = np.full(step_sizes.shape, eff_current)
            eff_prev_array = np.full(step_sizes.shape, eff_prev)
            too_large = (eff_current_array < (eff_prev_array + self.c1*step_sizes*np.linalg.norm(grad_prev, axis=1))) # check efficiency reduced sufficiently -> places upper bound
            
            step_sizes_out[too_large] = step_sizes[too_large] * self.down_factor
            
            if too_large.any():
                satisfied = False
            if (step_sizes_out < self.step_tol*self.initial_step_sizes).all(): # step size has gotten below tolerance
                satisfied = True
        else:
            satisfied = False
            step_sizes_out = step_sizes * self.down_factor
            self.sol.solved = True # to continue in while loop

        if satisfied:
            step_sizes_out = self.initial_step_sizes
        return step_sizes_out, satisfied
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.express as px
#-------------------------

class ReservoirManagement():

    def __init__(self, Country, Company, State):
        ''''
        Name of Country, Company and State of producing asset
        '''
        self.country = Country
        self.company = Company
        self.state = State

    def load_production_pressure_pvt_data(self, filelocation_production, filelocation_pressure=None, filelocation_pvt=None):

        self.filelocation_production = filelocation_production
        self.filelocation_pressure = filelocation_pressure
        self.filelocation_pvt = filelocation_pvt

    def production_data(self):
        self.load_production_pressure_pvt_data(filelocation_production=self.filelocation_production,filelocation_pressure=self.filelocation_pressure,filelocation_pvt=self.filelocation_pvt)
        ''''
        Load production data from xlsx data
        '''
        production_data_df = pd.read_excel(self.filelocation_production, header=None)
        self.production_data_df = production_data_df

        row_for_well_name = self.production_data_df.iloc[0,:] #read in the row that has well names
        count = -1
        for i in row_for_well_name:
            if (pd.isna(i) != True):
                count+=1
        self.total_no_wells = count #total number of wells
        print(self.total_no_wells)

    def bhp_data(self, upper_gauge_file=None,bottom_gauge_file=None, first_line_data_upper=None, first_line_data_bottom=None):
        self.upper_gauge_file_location = upper_gauge_file
        self.bottom_gauge_file_location = bottom_gauge_file
        self.first_line_data_upper = first_line_data_upper
        self.first_line_data_bottom = first_line_data_bottom
        self.upper_gauge_df = pd.read_csv(filepath_or_buffer=self.upper_gauge_file_location, delimiter='\s+', header=None, skiprows=self.first_line_data_upper)
        self.bottom_gauge_df = pd.read_csv(filepath_or_buffer=self.bottom_gauge_file_location, delimiter='\s+', header=None, skiprows=self.first_line_data_bottom)
        self.upper_gauge_df.columns = ('Date','Time','Elapsed Time','Pressure','Temperature')
        self.bottom_gauge_df.columns = ('Date', 'Time', 'Elapsed Time', 'Pressure', 'Temperature')

    def plot_upp_gauge_bhp(self):
        self.bhp_data(upper_gauge_file=self.upper_gauge_file_location,bottom_gauge_file=self.bottom_gauge_file_location, first_line_data=self.first_line_data)
        fig = px.line(self.upper_gauge_df, x=self.upper_gauge_df['Elapsed Time'], y=self.upper_gauge_df['Pressure'])
        fig.update_layout(title_text="Upper Gauge Pressure")
        fig.update_xaxes(title_text="<b>Elapsed Time<b>")
        fig.update_yaxes(title_text="<b>Pressure (psia)</b>")
        fig.show()

    def plot_upp_gauge_bhp_temp(self):
        self.bhp_data(upper_gauge_file=self.upper_gauge_file_location,bottom_gauge_file=self.bottom_gauge_file_location, first_line_data=self.first_line_data)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=self.upper_gauge_df['Elapsed Time'], y=(self.upper_gauge_df['Pressure']), name = 'Pressure'), secondary_y=False)
        fig.add_trace(go.Scatter(x=self.upper_gauge_df['Elapsed Time'], y=(self.upper_gauge_df['Temperature']), name='Temperature'), secondary_y=True)
        fig.update_layout(title_text="Upper Gauge Pressure")
        fig.update_xaxes(title_text="Elapsed Time")
        fig.update_yaxes(title_text="<b>Pressure (psia)</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Temperature deg F</b>", secondary_y=True)
        fig.show()

    def plot_bott_gauge_bhp(self):
        self.bhp_data(upper_gauge_file=self.upper_gauge_file_location,bottom_gauge_file=self.bottom_gauge_file_location, first_line_data_bottom=self.first_line_data_bottom)
        fig = px.line(self.bottom_gauge_df, x=self.bottom_gauge_df['Elapsed Time'], y=self.bottom_gauge_df['Pressure'])
        fig.update_layout(title_text="Bottom Gauge Pressure")
        fig.update_xaxes(title_text="<b>Elapsed Time<b>")
        fig.update_yaxes(title_text="<b>Pressure (psia)</b>")
        fig.show()

    def plot_bott_gauge_bhp_temp(self):
        self.bhp_data(upper_gauge_file=self.upper_gauge_file_location,bottom_gauge_file=self.bottom_gauge_file_location, first_line_data_bottom=self.first_line_data_bottom, first_line_data_upper=self.first_line_data_upper)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=self.bottom_gauge_df['Elapsed Time'], y=(self.bottom_gauge_df['Pressure']), name = 'Pressure'), secondary_y=False)
        fig.add_trace(go.Scatter(x=self.bottom_gauge_df['Elapsed Time'], y=(self.bottom_gauge_df['Temperature']), name='Temperature'), secondary_y=True)
        fig.update_layout(title_text="Bottom Gauge Pressure")
        fig.update_xaxes(title_text="<b>Elapsed Time<b>")
        fig.update_yaxes(title_text="<b>Pressure (psia)</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Temperature deg F</b>", secondary_y=True)
        fig.show()

    def plot_compare_bott_upp_gauge(self):
        self.bhp_data(upper_gauge_file=self.upper_gauge_file_location, bottom_gauge_file=self.bottom_gauge_file_location,
                  first_line_data_upper=self.first_line_data_upper, first_line_data_bottom=self.first_line_data_bottom)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
        go.Scatter(x=self.bottom_gauge_df['Elapsed Time'], y=(self.bottom_gauge_df['Pressure']), name='Bottom Pressure'),
        secondary_y=False)
        fig.add_trace(
        go.Scatter(x=self.upper_gauge_df['Elapsed Time'], y=(self.upper_gauge_df['Pressure']), name='Upper Pressure'),
        secondary_y=True)
        fig.update_layout(title_text="Gauge Pressures")
        fig.update_xaxes(title_text="<b>Elapsed Time<b>")
        fig.update_yaxes(title_text="<b> Bottom Pressure (psia)</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b> Upper Pressure (psia) </b>", secondary_y=True)
        fig.show()

    def plot_fluid_gradient(self,file_location):
        self.gradient_file_location = file_location
        self.fluid_gradient = pd.read_csv(filepath_or_buffer=self.gradient_file_location, delimiter='\s+', header=None, skiprows=self.first_line_data)

    def get_index_drawdown(self):
        listofPOS = list(self.bottom_gauge_df['Elapsed Time'])
        listofPOS = [float(format(num,'.5f')) for num in listofPOS]
        self.index_start = listofPOS.index(self.start_time)
        self.index_stop = listofPOS.index(self.stop_time)

        self.bottom_gauge_drawdown_df = self.bottom_gauge_df.iloc[self.index_start:self.index_stop].copy()  # create a dataframe of pressure vs time to store the drawdown information using start and stop time
        self.bottom_gauge_drawdown_df['Elapsed Time Normalized'] = self.bottom_gauge_drawdown_df['Elapsed Time'] - self.bottom_gauge_drawdown_df['Elapsed Time'].iloc[0]

        listofPOS1 = list(self.bottom_gauge_drawdown_df['Elapsed Time Normalized'])  # make a list of the elapsed normalized time
        listofPOS1 = [float(format(num, '.5f')) for num in listofPOS1]  # change to 5 decimal places
        self.index_start1 = listofPOS1.index(self.IARF_start)  # the beginning of the IARF straight line
        self.index_stop1 = listofPOS1.index(self.IARF_stop)  # the end of the IARF straight line
        self.x = self.bottom_gauge_drawdown_df['Elapsed Time Normalized'].iloc[self.index_start1:self.index_stop1].copy().to_numpy()  # the x component of the IARF straight line

        listofPOS1_press = list(self.bottom_gauge_drawdown_df['Pressure'])  # make a list of the elapsed normalized time
        listofPOS1_press = [float(format(num1, '.5f')) for num1 in listofPOS1_press]  # change to 5 decimal places
        self.y = self.bottom_gauge_drawdown_df['Pressure'].iloc[self.index_start1:self.index_stop1].copy().to_numpy()  # the x component of the IARF straight line

    def drawdown(self,start_time, stop_time,IARF_start=None, IARF_stop=None, Qo=None,Bo=None,ViscO=None,h=None,P1hour = None,Ct=None,Rw=None, print_plot=True):
        '''
        The start_time and stop_time are determined from the initial plot of bottom hole gauge pressure...
        The IARF_start and IARF_stop are determined from the normalized plot of pressure vs normalized time...
        '''
        self.Qo = Qo
        self.Bo = Bo
        self.ViscO = ViscO
        self.h = h
        self.P1hour = P1hour
        self.Ct = Ct
        self.Rw = Rw

        self.start_time = start_time
        self.stop_time = stop_time
        self.IARF_start = IARF_start #determined from the normalized plot of pressure vs normalized time...
        self.IARF_stop = IARF_stop #determined from the normalized plot of pressure vs normalized time...

        self.get_index_drawdown()

        self.m,self.b = np.polyfit(x=self.x,y=self.y,deg=1)

        if (self.Bo is not None) & (self.ViscO is not None) & (self.Qo is not None) & (self.h is not None) & (self.m is not None):
            self.K = (162.6*self.Qo*self.Bo*self.ViscO)/(self.h*self.m)

        print('Perk K is {}md'.format(abs(self.K)))

        if print_plot != False:
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(go.Scatter(x=self.bottom_gauge_drawdown_df['Elapsed Time Normalized'], y=self.bottom_gauge_drawdown_df['Pressure'], name='Drawdown Bottom Pressure'), secondary_y=False)
            fig.add_trace(go.Scatter(x=self.x, y=self.m * self.x + self.b, name='IARF'), secondary_y=False)
            fig.update_layout(title_text="Drawdown Test")
            fig.update_xaxes(title_text="<b>Elapsed Time Normalized<b>", type='log')
            fig.update_yaxes(title_text="<b>Pressure (psia)</b>")
            fig.show()

    def get_index_log_log_buildup(self):

        self.L =0.01/2
        listofPOS_bu = list(self.bottom_gauge_df['Elapsed Time'])
        listofPOS_bu = [float(format(num, '.5f')) for num in listofPOS_bu]
        self.index_tp_bu_log = listofPOS_bu.index(self.tp_bu_log) #the self.tp_bu is selected by the user from the log_log
        self.index_stop_bu_log = listofPOS_bu.index(self.stop_time_bu_log) #also selected by the user

        self.bottom_gauge_buildup_df = self.bottom_gauge_df.iloc[self.index_tp_bu_log:self.index_stop_bu_log].copy()  # create a dataframe of pressure vs time to store the build up information using start and stop time

        self.bottom_gauge_buildup_df['Elapsed Time Normalized L'] = self.bottom_gauge_buildup_df['Elapsed Time'] - self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0]

        self.bottom_gauge_buildup_df['Elapsed Time Normalized C'] = self.bottom_gauge_buildup_df['Elapsed Time Normalized L'].shift(2, axis=0)

        self.bottom_gauge_buildup_df['Elapsed Time Normalized R'] = self.bottom_gauge_buildup_df['Elapsed Time Normalized L'].shift(3, axis=0)

        self.bottom_gauge_buildup_df['Pressure L'] = self.bottom_gauge_buildup_df['Pressure']

        self.bottom_gauge_buildup_df['Pressure C'] = self.bottom_gauge_buildup_df['Pressure'].shift(2, axis=0)

        self.bottom_gauge_buildup_df['Pressure R'] = self.bottom_gauge_buildup_df['Pressure'].shift(3, axis=0)

        #resample the time index of the build up pressure
        self.bottom_gauge_buildup_df['Normalized_Time'] = self.bottom_gauge_buildup_df['Elapsed Time'] - self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0] #this will be used for smooth time

        t_last = max(self.bottom_gauge_buildup_df['Normalized_Time']) #the maximum time that will be used for the data

        t = self.bottom_gauge_buildup_df['Normalized_Time'].iloc[0] #this is to initialize t to a starting time a.k.a the first build up period t

        start_index = self.bottom_gauge_buildup_df['Pressure'].index[0] #the row number that starts the data

        smooth_time_stamps = []

        smooth_pressure_stamps = []

        while t < t_last:

            exactmatch = self.bottom_gauge_buildup_df[self.bottom_gauge_buildup_df['Normalized_Time']==t] #look for the row in the main dataframe where the time matches

            if not exactmatch.empty: #this means it has the same exact timestamp match
                #smooth_time_stamps.append(exactmatch.index) #this puts the index that we found into a list
                smooth_pressure_stamps.append(self.bottom_gauge_buildup_df['Pressure'].iloc[int(exactmatch.index.values) - start_index])

            else:
                closest_lower = self.bottom_gauge_buildup_df[self.bottom_gauge_buildup_df['Normalized_Time'] < t]['Normalized_Time'].idxmax()
                #smooth_time_stamps.append(closest_lower)
                smooth_pressure_stamps.append(self.bottom_gauge_buildup_df['Pressure'].iloc[closest_lower-start_index])

            smooth_time_stamps.append(t)

            t = round((t + self.L),3) #this is too round the time to 3 dp then loop it


        #print(smooth_time_stamps)
        #print(smooth_pressure_stamps) #the pressure points

        #convert the lists into a data frame
        smooth_times = smooth_time_stamps
        smooth_pressures = smooth_pressure_stamps
        data = {'Elapsed Time Smooth': smooth_times, 'Pressure Smooth':smooth_pressures}
        Smooth_BU_df = pd.DataFrame(data, columns=['Elapsed Time Smooth','Pressure Smooth'])
        self.Smooth_BU_df = Smooth_BU_df

        Smooth_BU_df['Elapsed Time Smooth L'] = Smooth_BU_df['Elapsed Time Smooth']
        Smooth_BU_df['Elapsed Time Smooth C'] = Smooth_BU_df['Elapsed Time Smooth L'].shift(1, axis=0)
        Smooth_BU_df['Elapsed Time Smooth R'] = Smooth_BU_df['Elapsed Time Smooth C'].shift(1, axis=0)

        Smooth_BU_df['Pressure Smooth L'] = Smooth_BU_df['Pressure Smooth']
        Smooth_BU_df['Pressure Smooth C'] = Smooth_BU_df['Pressure Smooth L'].shift(1, axis=0)
        Smooth_BU_df['Pressure Smooth R'] = Smooth_BU_df['Pressure Smooth C'].shift(1, axis=0)

        #the is the derivative that will be plot
        Smooth_BU_df['t Delta P/Delta t'] = ((np.log(
            Smooth_BU_df['Elapsed Time Smooth R']) - np.log(
            Smooth_BU_df['Elapsed Time Smooth C'])) * (((
                    Smooth_BU_df['Pressure Smooth C'] - Smooth_BU_df['Pressure Smooth L'])) / (np.log(
            Smooth_BU_df['Elapsed Time Smooth C']) - np.log(
            Smooth_BU_df['Elapsed Time Smooth L']))) + (np.log(
            Smooth_BU_df['Elapsed Time Smooth C']) - np.log(
            Smooth_BU_df['Elapsed Time Smooth L'])) * (((
                    Smooth_BU_df['Pressure Smooth R'] - Smooth_BU_df['Pressure Smooth C'])) / (np.log(
            Smooth_BU_df['Elapsed Time Smooth R']) - np.log(
            Smooth_BU_df['Elapsed Time Smooth C'])))) / (np.log(
            Smooth_BU_df['Elapsed Time Smooth R']) - np.log(
            Smooth_BU_df['Elapsed Time Smooth L']))


        self.bottom_gauge_buildup_df['Pws - Pwf(tp)'] = self.bottom_gauge_buildup_df['Pressure'] - self.bottom_gauge_buildup_df['Pressure'].iloc[0]

        self.bottom_gauge_buildup_df['t Delta P/Delta t'] = ((np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized R'])-np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized C']))*(((self.bottom_gauge_buildup_df['Pressure C']-self.bottom_gauge_buildup_df['Pressure L']))/(np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized C'])-np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized L']))) + (np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized C'])-np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized L']))*(((self.bottom_gauge_buildup_df['Pressure R']-self.bottom_gauge_buildup_df['Pressure C']))/(np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized R'])-np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized C'])))) / (np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized R']) - np.log(self.bottom_gauge_buildup_df['Elapsed Time Normalized L']))

    def buildup_log_log(self,tp_bu_log,stop_time_bu_log, show_plot=True):
        ''''
        build up analysis portion
        '''
        self.tp_bu_log = tp_bu_log #the total producing time before shut in
        self.stop_time_bu_log = stop_time_bu_log

        self.get_index_log_log_buildup()

        if show_plot != False:
            self.bottom_gauge_buildup_df.sort_values(by='Elapsed Time Normalized L')
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(go.Scatter(x=self.bottom_gauge_buildup_df['Elapsed Time Normalized L'], y=self.bottom_gauge_buildup_df['Pws - Pwf(tp)'], name='P', mode='markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=self.Smooth_BU_df['Elapsed Time Smooth'], y=self.Smooth_BU_df['t Delta P/Delta t'], name="P'", mode='markers'), secondary_y=False)
            #fig.add_trace(go.Scatter(x=self.bottom_gauge_buildup_df['Elapsed Time Normalized L'], y=self.bottom_gauge_buildup_df['t Delta P/Delta t'], name="P'", mode='markers'), secondary_y=False)
            fig.update_layout(title_text="Buildup Log Log")
            fig.update_xaxes(title_text="<b>Delta t<b>", type='log')
            fig.update_yaxes(title_text="<b>Pws - Pwf(tp), P' (psia)</b>", type='log')
            fig.show()

    def get_index_buildup(self):

        listofPOS_bu = list(self.bottom_gauge_df['Elapsed Time'])
        listofPOS_bu = [float(format(num, '.5f')) for num in listofPOS_bu]
        self.index_tp_bu = listofPOS_bu.index(self.tp_bu)
        self.index_stop_bu = listofPOS_bu.index(self.stop_time_bu)

        self.bottom_gauge_buildup_df = self.bottom_gauge_df.iloc[self.index_tp_bu:self.index_stop_bu].copy()  # create a dataframe of pressure vs time to store the build up information using start and stop time

        self.bottom_gauge_buildup_df['Elapsed Time Normalized'] = self.bottom_gauge_buildup_df['Elapsed Time'] - self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0]

        print("Producing time tp is {} hours".format(self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0]))
        print("Build up time end is {} hours".format(self.bottom_gauge_buildup_df['Elapsed Time'].iloc[-1]))
        print("Build up time duration is {} hours".format(self.bottom_gauge_buildup_df['Elapsed Time'].iloc[-1] - self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0]))

        print("HTR1hour is {}".format((self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0]+1)/1))

        self.bottom_gauge_buildup_df['[tp + delta t] / [delta t]'] = (self.bottom_gauge_buildup_df['Elapsed Time'].iloc[0] + self.bottom_gauge_buildup_df['Elapsed Time Normalized'])/self.bottom_gauge_buildup_df['Elapsed Time Normalized']

    def buildup(self,tp_bu,stop_time_bu,m,b,show_plot=True):
        ''''
        build up analysis portion
        '''
        self.tp_bu = tp_bu
        self.stop_time_bu = stop_time_bu

        self.m = m
        self.b = b

        self.get_index_buildup()

        self.bottom_gauge_buildup_df['Line'] = (self.m * np.log(self.bottom_gauge_buildup_df['[tp + delta t] / [delta t]'])) + self.b

        if show_plot != False:
            self.bottom_gauge_buildup_df.sort_values(by='[tp + delta t] / [delta t]')
            #df = px.data.iris()
            #fig = px.scatter(df, x=self.bottom_gauge_buildup_df['[tp + delta t] / [delta t]'], y=[self.bottom_gauge_buildup_df['Pressure'], self.bottom_gauge_buildup_df['Line']])
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(go.Scatter(x=self.bottom_gauge_buildup_df['[tp + delta t] / [delta t]'], y=self.bottom_gauge_buildup_df['Pressure'], name='Horner Buildup Pressure'), secondary_y=False)
            fig.add_trace(go.Scatter(x=self.bottom_gauge_buildup_df['[tp + delta t] / [delta t]'], y=self.bottom_gauge_buildup_df['Line'], name='IARF'), secondary_y=False)
            fig.update_layout(title_text="Buildup Test")
            fig.update_xaxes(title_text="<b>HTR<b>", type='log')
            fig.update_yaxes(title_text="<b>Pws</b>")
            fig.show()

Top_gauge_office = 'TOP_GAUGE.TXT'
Bottom_gauge_office = 'BOTTOM_GAUGE.TXT'

test_project = ReservoirManagement(Country='USA',Company='CompanyA',State='Texas')

test_project.bhp_data(upper_gauge_file= Top_gauge_office ,bottom_gauge_file= Bottom_gauge_office , first_line_data_upper=19, first_line_data_bottom=21)

test_project.buildup(tp_bu=11.18833,stop_time_bu=20.1675,m=-1.5,b=3220.1,show_plot=True)

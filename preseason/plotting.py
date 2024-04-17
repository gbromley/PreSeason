import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker

import cartopy.io.shapereader as shapereader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def plotOnsetTS(p_anom, onset_data, demise_data, time_slice, loc = None, iloc = None):
    ### Take precip data, calculated onsets, make a nice TS figure
    #TODO error handle on the other arguments
    ### Argument Handling ###
    if (loc is None and iloc is None) or (loc is not None and iloc is not None):
        raise ValueError("Exactly one location argument must be provided.")
    
    
    if (loc is not None):
        lat = loc[0]
        lon = loc[1]
        
    if (iloc is not None):
        lat = p_anom.lat[iloc[0]]
        lon = p_anom.lon[iloc[1]]
    
    
    ### Data Wrangling ###
    
    
    
    p_anom_ts = p_anom.sel(time=time_slice, lat=lat, lon=lon)
    
    time = p_anom_ts.time.values
    year_array = pd.to_datetime(np.unique(p_anom_ts.time.dt.year), format="%Y")
    
    onset_ts = onset_data.sel(year=time_slice, lat=lat, lon=lon)
    demise_ts = demise_data.sel(year=time_slice, lat=lat, lon=lon)
    

    
    ### Onset/Demise plotting section ###
    
    # create empty array to hold dates
    onset_dates = np.empty_like(onset_ts, dtype=np.dtype('M8[D]'))
    onset_dates[:] = np.datetime64("NaT")

    demise_dates = np.empty_like(demise_ts, dtype=np.dtype('M8[D]'))
    demise_dates[:] = np.datetime64("NaT")

    # convert onset days to labels
    for i,j in enumerate(onset_ts.values):
        
        if not np.isnan(j):
            
            onset_dates[i] = year_array[i] + pd.Timedelta(int(j), unit='day')
            
    for i,j in enumerate(demise_ts.values):
        if not np.isnan(j):
            
            demise_dates[i] = year_array[i] + pd.Timedelta(int(j), unit='day')

    ticks_str = year_array

    ticks_to_add = [int(mdates.date2num(item)) for item in ticks_str]
    
    additional_labels = [mdates.num2date(year).strftime('%j-%y') for year in ticks_to_add]

    
    
    ### Plotting Parameters ###
    plt.rcParams['font.size'] = 20
    fs = plt.rcParams['font.size']
    pre_color = '#30638E'
    cumul_color = '#193C16'
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()
    fig.set_size_inches((16,9))
    
    # Plot the first data set with the first y-axis
    anom_line = ax1.plot(time, p_anom_ts, pre_color, label=r'P$_{anom}$', alpha=0.8)  # 'g-' is for green solid line
    
    ax1.set_xlabel('Day of Year')
    ax1.set_ylabel(r'P$_{anom}$ (mm)', color=pre_color, fontsize=fs+4)  # Set the color of the y-axis label to green
    ax1.tick_params(axis='y', labelcolor='black')
    
    ymin, ymax = plt.ylim()
    ax1.set_ylim(ymin,ymax+25)
    
    # Color in the area above and below zero
    ax1.fill_between(time, p_anom_ts, 0, alpha=0.5, where= p_anom_ts > 0, facecolor = '#30638E')
    ax1.fill_between(time, p_anom_ts, 0, alpha=0.5, where= p_anom_ts < 0, facecolor='#857E61')


    for i in onset_dates:
        if not np.isnan(i):
            onset_line = plt.axvline(x=i, ymin=0, ymax=1, color='#6AA078', linewidth=2, linestyle='--')

    for j in demise_dates:
        if not np.isnan(j):
            
            demise_line = plt.axvline(x=j, ymin=0, ymax=1, color='#C5C392', linewidth=2, linestyle='--')
    

    onset_line.set_label(r'Onset$_{ wet}$')
    demise_line.set_label(r'Onset$_{ dry}$')
    
        
    # make the ticks look pretty
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    current_ticks = plt.gca().get_xticks()[1:]
    current_labels = [mdates.num2date(tick).strftime('%-j') for tick in current_ticks]
    
    # Combine current and additional ticks and labels
    all_ticks = np.concatenate((current_ticks, ticks_to_add))
    all_labels = current_labels + additional_labels
    
    plt.gca().set_xticks(all_ticks)
    plt.gca().set_xticklabels(all_labels)
    plt.gcf().autofmt_xdate(rotation=30)
    
    # adjust the legend
    legend_artists = [anom_line[0], demise_line, onset_line]
    legend_labels = [r'P$_{anom}$', r'Onset$_{ dry}$', r'Onset$_{ wet}$']
    fig.legend(legend_artists, legend_labels, loc='lower left', bbox_to_anchor=(.1175, 0.802, 0.79, .25), ncols=3, mode='expand', frameon=True, fancybox=False, framealpha=1, edgecolor='white')

    # set title to lat/lon
    ax1.set_title('Location: '+str(np.abs(lat.values))+' S, '+str(lon.values)+' E'+' (Peru)', x=0.5, y=1.05)



    #out_path = '/Users/gbromley/Dropbox/OU/Peru/'
    #plt.savefig(out_path+'Peru_Example.png', dpi=600, transparent=True)
    plt.show()
    
    return None


def plot_spatial_data(dataarray, projection=ccrs.PlateCarree(), cmap ='twilight', vmax = 365, vmin = 1, title='Spatial Data Plot', var='data_to_plot'):
    """
    Plots a spatial figure of a variable from an xarray DataArray.

    :param dataarray: xarray DataArray containing the geospatial data to be plotted.
    :param projection: Cartopy CRS projection. Defaults to PlateCarree.
    :param title: Title of the plot.
    """
    
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m')
    map_proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=45)
    #cmap = mpl.cm.RdBu_r

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1, 1, figsize=(10, 13), dpi=600,  subplot_kw={'projection': projection})
    p = dataarray.plot(ax=ax,transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap,alpha = 0.8, norm=norm)


    ### Setting 1st plot parameters ###
    ax.coastlines(edgecolor='black', linewidth=2)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.LAKES, alpha=0.5, edgecolor='blue')
    ax.add_feature(cfeature.RIVERS, color='blue')
    #ax1.set_xticks(np.arange(-180,181, 40))
    #ax1.set_yticks(np.arange(-90,91,15))
    
    ###TODO Change shape files to be from internet
    drainage = shapereader.Reader('/Users/gbromley/Downloads/major_basins_of_the_world_0_0_0/Major_Basins_of_the_World.shp')
    #for feature in drainage.records():
    #    geometry = feature.geometry
    #    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=2, linestyle='--')
    #at = AnchoredText("a",
    #                    loc='upper left', prop=dict(size=8), frameon=True,)
    #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    #ax1.add_artist(at)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    plt.colorbar(p, cax=cax, label=var)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=2, color='black', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([280-360, 285-360, 290-360])
    gl.ylocator = mticker.FixedLocator([-5, -10, -15])
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = True

    # Add a title
    ax.set_title(title, loc='center')

    # Show the plot
    plt.show()
    
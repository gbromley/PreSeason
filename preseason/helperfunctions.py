import intake_esm
url = intake_esm.tutorial.get_url('google_cmip6')
cat = intake_esm.esm_datastore(url)
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df = df[(df['variable_id']=='pr') & (df['source_id']=='CESM2-WACCM') & (df['activity_id']=='CMIP') & (df['experiment_id']=='historical') & (df['table_id']=='day') & (df['grid_label']=='gn')]
df
ds = open_dsets(df)

dsets = [xr.open_zarr(fsspec.get_mapper(ds_url), consolidated=True)
             .pipe(drop_all_bounds)
             for ds_url in df.zstore]
ds = xr.concat(dsets, join='outer', dim='member')

### Peru Domain ###
min_lon = -83+360
min_lat = -20.0
max_lon = -67+360
max_lat = 0.0

ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon, max_lon))
pr = ds['pr'].load()
pr.to_netcdf('CESM2_CMIP6_historical.nc')
pr = xr.open_dataset('CESM2_CMIP6_historical.nc')
pr['time']
ssp5 = xr.open_dataset('pr_day_CESM2_ssp585_r4i1p1f1_gn_20500101-20991231_v20200528.nc')
pd.date_range('1850-01-01', freq='D', periods=60228)
pr['time'] = pd.date_range('1850-01-01', freq='D', periods=60228)



states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')
map_proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=45)
#cmap = mpl.cm.RdBu_r


f, ax1 = plt.subplots(1, 1, figsize=(10, 13), dpi=600, subplot_kw={'projection': map_proj})
p = pr.where(pr['time.month']==1).mean(dim='time').isel(member=0).plot.pcolormesh(ax=ax1,transform=ccrs.PlateCarree(), add_colorbar=False, cmap='viridis')


### Setting 1st plot parameters ###
ax1.coastlines(color='grey')
ax1.add_feature(cartopy.feature.BORDERS, color='black')
ax1.add_feature(cfeature.STATES, edgecolor='black')
#ax1.set_xticks(np.arange(-180,181, 40))
#ax1.set_yticks(np.arange(-90,91,15))
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
at = AnchoredText("a",
                      loc='upper left', prop=dict(size=8), frameon=True,)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
plt.colorbar(p, cax=cax)

def drop_all_bounds(ds):
    """Drop coordinates like 'time_bounds' from datasets,
    which can lead to issues when merging."""
    drop_vars = [vname for vname in ds.coords
                 if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop(drop_vars)

def open_dsets(df):
    """Open datasets from cloud storage and return xarray dataset."""
    dsets = [xr.open_zarr(fsspec.get_mapper(ds_url), consolidated=True)
             .pipe(drop_all_bounds)
             for ds_url in df.zstore]
    try:
        ds = xr.merge(dsets, join='exact')
        return ds
    except ValueError:
        return None

def open_delayed(df):
    """A dask.delayed wrapper around `open_dsets`.
    Allows us to open many datasets in parallel."""
    return dask.delayed(open_dsets)(df)


def toDOY(time):
    y_ar = (y_ar - 1970).astype('M8[Y]')
    m_ar = (m_ar - 1).astype('m8[M]')
    d_ar = (d_ar - 1).astype('m8[D]')

    date_ar = y_ar + m_ar + d_ar  # full date
    julians = date_ar - y_ar + 1
    
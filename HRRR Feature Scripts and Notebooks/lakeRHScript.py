from herbie import Herbie
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import urllib
from pathlib import Path
import requests


def herbieDateTime(dateTime=None):
    if dateTime is None:
        dt = pd.Timestamp.utcnow().floor("h") - pd.Timedelta(hours=2)
    else:
        dt = pd.Timestamp(dateTime)
    
    if dt.tzinfo is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt


def fetch_lake_geojson(lake_name="Cayuga Lake", overwrite=False, timeout=60):
    out_dir = Path("finger_lakes_geojson")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fname = lake_name.lower().replace(" ", "_") + ".geojson"
    out_path = out_dir / fname
    
    if out_path.exists() and not overwrite:
        return out_path
    
    base = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12/query"
    where = f"GNIS_NAME = '{lake_name}' AND FTYPE = 390"
    params = {
        "where": where,
        "outFields": "GNIS_NAME,FTYPE",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }
    
    url = base + "?" + urllib.parse.urlencode(params)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    geojson = r.json()
    
    features = geojson.get("features", [])
    if len(features) == 0:
        raise ValueError(f"No features returned for lake '{lake_name}'")
    if len(features) > 1:
        raise ValueError(
            f"Multiple features returned for lake '{lake_name}' "
            f"({len(features)} features). Refine query."
        )
    
    out_path.write_bytes(r.content)
    print(f"Saved {lake_name}: {len(r.content):,} bytes → {out_path}")
    
    return out_path



def getLakeGrid(lake_name="Cayuga Lake", spacing=0.02):
    lake_geoJson = gpd.read_file(fetch_lake_geojson(lake_name))
    lakeGeom = lake_geoJson.geometry.iloc[0]
    minLon, minLat, maxLon, maxLat = lakeGeom.bounds
    
    candidate_lons = np.arange(minLon, maxLon, spacing)
    candidate_lats = np.arange(minLat, maxLat, spacing)
    
    interior_points = []
    for lat in candidate_lats:
        for lon in candidate_lons:
            p = Point(lon, lat)
            if lakeGeom.contains(p):
                interior_points.append((lat, lon))
    
    return interior_points


def rh850Grid(lake_name="Cayuga Lake", dateTime=None, spacing=0.02):
    dateTime = herbieDateTime(dateTime)

    interior_points = getLakeGrid(lake_name)
    inLats = [p[0] for p in interior_points]
    inLons = [p[1] for p in interior_points]
    #Specify the points to be passed to Herbie's pick_points() function
    points = pd.DataFrame(
        {
            "latitude" : inLats,
            "longitude" : inLons,
        }
    )

    ds850 = rh850(dateTime)
    sampled = ds850.herbie.pick_points(points)

    rh_vals= sampled["r"].values

    projected = gpd.GeoDataFrame(
        {
            "lake": lake_name,
            "timestamp": pd.Timestamp(dateTime),
            "latitude": inLats,
            "longitude": inLons,
            "rh850": rh_vals,
        },
        geometry=gpd.points_from_xy(inLons, inLats),
        crs="EPSG:4326",
    )
    return projected

def rh850(dateTime=None):
    #Convert time entered to UTC
    dt = herbieDateTime(dateTime)
    #HRRR PRS Herbie variable
    H = Herbie(
        dt,
        model="hrrr",
        product="prs",      #3D pressure level   
        fxx=0,                  
        save_dir=Path("herbie_cache"),
        overwrite=False
    )

    #GRIB2 subset only RH at 850 mb 
    pat = r":RH:850 mb:"
    H.download(pat, verbose=False)

    #Read GRIB2 in xarray.Dataset
    ds = H.xarray(pat, remove_grib=True)

    return ds

def avgRH850(lake_name="Cayuga Lake", dateTime = None, spacing = 0.02):
    dateTime = herbieDateTime(dateTime)
    grid = rh850Grid(lake_name, dateTime, spacing)
    if grid.empty:
        return pd.DataFrame(columns=["lake", "dateTime", "meanRH850"])
    return pd.DataFrame(
        {
            "lake": [grid["lake"].iloc[0]],
            "dateTime" : [grid["timestamp"].iloc[0]],
            "meanRH850" : [grid["rh850"].mean()],
        }
    )

def batch_avgRH850(lake_names, dateTimes, spacing=0.02):
    results = []
    
    for lake in lake_names:
        for dt in dateTimes:
            try:
                result = avgRH850(lake, dt, spacing)
                results.append(result)
            except Exception as e:
                print(f"Error processing {lake} at {dt}: {e}")
                continue
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["lake", "dateTime", "meanRH850"])


def range_avgRH850(lake_name, start_dateTime, end_dateTime, spacing=0.02):
    start = herbieDateTime(start_dateTime)
    end = herbieDateTime(end_dateTime)
    
    if start > end:
        raise ValueError("Start dateTime must be before end dateTime")
    
    # Generate hourly timestamps
    time_range = pd.date_range(start=start, end=end, freq='h')
    print(f"Processing {len(time_range)} hours from {start} to {end}")
    
    results = []
    for dt in time_range:
        try:
            result = avgRH850(lake_name, dt, spacing)
            results.append(result)
        except Exception as e:
            print(f"Error processing {lake_name} at {dt}: {e}")
            continue
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["lake", "dateTime", "meanRH850"])


def interactive_main():
    """Interactive command-line interface for lake relative humidity at 850mb analysis."""
    print("=" * 60)
    print("Lake Relative Humidity at 850mb")
    print("=" * 60)
    
    # Get lake name
    lake_name = input("\nEnter lake name (default: Cayuga Lake): ").strip()
    if not lake_name:
        lake_name = "Cayuga Lake"
    
    # Get download type
    print("\nDownload options:")
    print("  1. Single datetime")
    print("  2. Range of datetimes (hourly)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Get spacing preference
    spacing_input = input("\nEnter grid spacing in degrees (default: 0.02 ≈ 2km): ").strip()
    spacing = float(spacing_input) if spacing_input else 0.02
    
    try:
        if choice == "1":
            # Single datetime
            datetime_str = input("\nEnter datetime (e.g., '2026-01-01 00:00'): ").strip()
            if not datetime_str:
                print("Using current time minus 2 hours...")
                datetime_str = None
            
            print("\nProcessing...")
            result = avgRH850(lake_name, datetime_str, spacing)
            
        else:
            # Range of datetimes
            start_str = input("\nEnter start datetime (e.g., '2026-01-01 00:00'): ").strip()
            end_str = input("Enter end datetime (e.g., '2026-01-01 23:00'): ").strip()
            
            if not start_str or not end_str:
                raise ValueError("Both start and end datetime must be provided for range download")
            
            print("\nProcessing...")
            result = range_avgRH850(lake_name, start_str, end_str, spacing)
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(result.to_string(index=False))
        
        # Option to save
        save = input("\nSave results to CSV? (y/n): ").strip().lower()
        if save == 'y':
            if choice == "1":
                # Single datetime - use the datetime in filename
                dt = result['dateTime'].iloc[0]
                timestamp = dt.strftime('%Y%m%d_%H%M')
                filename = f"{lake_name.lower().replace(' ', '_')}_{timestamp}.csv"
            else:
                # Range - use start and end in filename
                start_dt = result['dateTime'].iloc[0]
                end_dt = result['dateTime'].iloc[-1]
                start_str = start_dt.strftime('%Y%m%d_%H%M')
                end_str = end_dt.strftime('%Y%m%d_%H%M')
                filename = f"{lake_name.lower().replace(' ', '_')}_{start_str}_to_{end_str}.csv"
            result.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        
    except Exception as e:
        print(f"\nError: {e}")
        return


# Example usage
if __name__ == "__main__":
    # Run interactive mode
    interactive_main()
    
    # Or use functions directly:
    # result = avgRH850("Cayuga Lake", "2026-01-03 05:00")
    # print(result)
    
    # range_result = range_avgRH850("Cayuga Lake", "2026-01-01 00:00", "2026-01-01 23:00")
    # print(range_result)
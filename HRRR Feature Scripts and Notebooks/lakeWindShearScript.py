from herbie import Herbie
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import urllib
from pathlib import Path
import requests
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=True)


def herbieDateTime(dateTime=None):
    """
    Normalize user-provided datetime to a naive UTC pandas.Timestamp.

    If dateTime is None, uses current UTC time floored to the hour minus 2 hours.
    """
    if dateTime is None:
        dt = pd.Timestamp.utcnow().floor("h") - pd.Timedelta(hours=2)
    else:
        dt = pd.Timestamp(dateTime)

    if dt.tzinfo is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt


def fetch_lake_geojson(lake_name="Cayuga Lake", overwrite=False, timeout=60):
    """
    Download (or reuse cached) GeoJSON for a named lake from The National Map (NHD).
    """
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
    print(f"Saved {lake_name}: {len(r.content):,} bytes \u2192 {out_path}")

    return out_path


def getLakeGrid(lake_name="Cayuga Lake", spacing=0.02):
    """
    Generate a list of (lat, lon) interior points within the lake polygon.
    """
    lake_geojson = gpd.read_file(fetch_lake_geojson(lake_name))
    lake_geom = lake_geojson.geometry.iloc[0]
    minLon, minLat, maxLon, maxLat = lake_geom.bounds

    candidate_lons = np.arange(minLon, maxLon, spacing)
    candidate_lats = np.arange(minLat, maxLat, spacing)

    interior_points = []
    for lat in candidate_lats:
        for lon in candidate_lons:
            p = Point(lon, lat)
            if lake_geom.contains(p):
                interior_points.append((lat, lon))

    return interior_points


def surfaceAnd850(dateTime=None):
    """
    Fetch HRRR sfc product (fxx=0) and return two xarray datasets:
      - Surface: UGRD/VGRD at 10 m AGL (heightAboveGround)
      - 850 mb: UGRD/VGRD at 850 mb (isobaricInhPa)
    """
    dateTime = herbieDateTime(dateTime)
    H = Herbie(
        dateTime,
        model="hrrr",
        product="sfc",
        fxx=0,
        save_dir=Path("herbie_cache"),
        overwrite=False,
    )

    var_regex = r"[U|V]GRD:10 m|[U|V]GRD:850 mb"
    H.download(var_regex, verbose=False)

    dMixed = H.xarray(var_regex, remove_grib=True)
    d850mb = next(ds for ds in dMixed if "isobaricInhPa" in ds.coords)
    dSurface = next(ds for ds in dMixed if "heightAboveGround" in ds.coords)
    return dSurface, d850mb


def deltaWindGrid(lake_name="Cayuga Lake", dateTime=None, spacing=0.02):
    """
    Compute directional wind shear magnitude (smallest angle difference, degrees)
    between surface 10m winds and 850 mb winds for points inside the lake.
    Returns a GeoDataFrame with a point geometry for each sampled location.
    """
    dateTime = herbieDateTime(dateTime)

    interior_points = getLakeGrid(lake_name, spacing)
    if not interior_points:
        return gpd.GeoDataFrame(
            columns=["lake", "timestamp", "latitude", "longitude", "delta_wind_shear", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    lakeSurface, lake850mb = surfaceAnd850(dateTime)

    inLats = [p[0] for p in interior_points]
    inLons = [p[1] for p in interior_points]

    points = pd.DataFrame({"latitude": inLats, "longitude": inLons})

    lakeSurfaceWind = lakeSurface.herbie.pick_points(points)
    lake850Wind = lake850mb.herbie.pick_points(points)

    # Meteorological wind direction ("from", degrees clockwise from north).
    # Using arctan2(-u, -v) is a common conversion for (u, v) to "from" direction.
    lakeSurfaceWindTheta = (np.degrees(np.arctan2(-lakeSurfaceWind["u10"].values,
                                                 -lakeSurfaceWind["v10"].values)) + 360) % 360
    lake850WindTheta = (np.degrees(np.arctan2(-lake850Wind["u"].values,
                                             -lake850Wind["v"].values)) + 360) % 360

    windDirectionDiff = np.abs(lakeSurfaceWindTheta - lake850WindTheta)
    windDirectionDiff = np.minimum(windDirectionDiff, 360 - windDirectionDiff)

    projectedLakePoints = gpd.GeoDataFrame(
        {
            "lake": lake_name,
            "timestamp": pd.Timestamp(dateTime),
            "latitude": inLats,
            "longitude": inLons,
            "delta_wind_shear": windDirectionDiff,
        },
        geometry=gpd.points_from_xy(inLons, inLats),
        crs="EPSG:4326",
    )

    return projectedLakePoints


def avgWindShear(lake_name="Cayuga Lake", dateTime=None, spacing=0.02):
    """
    Return a 1-row DataFrame with mean directional shear for the lake at a datetime.
    """
    dateTime = herbieDateTime(dateTime)
    windGrid = deltaWindGrid(lake_name, dateTime, spacing)

    if getattr(windGrid, "empty", True):
        return pd.DataFrame(columns=["lake", "dateTime", "meanDeltaWindShear"])

    return pd.DataFrame(
        {
            "lake": [windGrid["lake"].iloc[0]],
            "dateTime": [windGrid["timestamp"].iloc[0]],
            "meanDeltaWindShear": [windGrid["delta_wind_shear"].mean()],
        }
    )


def batch_avgWindShear(lake_names, dateTimes, spacing=0.02):
    """
    Compute avgWindShear for multiple lakes and datetimes.
    """
    results = []
    for lake in lake_names:
        for dt in dateTimes:
            try:
                results.append(avgWindShear(lake, dt, spacing))
            except Exception as e:
                print(f"Error processing {lake} at {dt}: {e}")
                continue

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=["lake", "dateTime", "meanDeltaWindShear"])


def range_avgWindShear(lake_name, start_dateTime, end_dateTime, spacing=0.02):
    """
    Compute avgWindShear hourly over a datetime range [start, end].
    """
    start = herbieDateTime(start_dateTime)
    end = herbieDateTime(end_dateTime)

    if start > end:
        raise ValueError("Start dateTime must be before end dateTime")

    time_range = pd.date_range(start=start, end=end, freq="h")
    print(f"Processing {len(time_range)} hours from {start} to {end}")

    results = []
    for dt in time_range:
        try:
            results.append(avgWindShear(lake_name, dt, spacing))
        except Exception as e:
            print(f"Error processing {lake_name} at {dt}: {e}")
            continue

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=["lake", "dateTime", "meanDeltaWindShear"])


def interactive_main():
    """Interactive command-line interface for lake wind shear analysis."""
    print("=" * 60)
    print("Lake Directional Wind Shear Analysis (10m vs 850mb)")
    print("=" * 60)

    lake_name = input("\nEnter lake name (default: Cayuga Lake): ").strip() or "Cayuga Lake"

    print("\nDownload options:")
    print("  1. Single datetime")
    print("  2. Range of datetimes (hourly)")

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")

    spacing_input = input("\nEnter grid spacing in degrees (default: 0.02 \u2248 2km): ").strip()
    spacing = float(spacing_input) if spacing_input else 0.02

    try:
        if choice == "1":
            datetime_str = input("\nEnter datetime (e.g., '2026-01-01 00:00'): ").strip()
            if not datetime_str:
                print("Using current time minus 2 hours...")
                datetime_str = None

            print("\nProcessing...")
            result = avgWindShear(lake_name, datetime_str, spacing)

        else:
            start_str = input("\nEnter start datetime (e.g., '2026-01-01 00:00'): ").strip()
            end_str = input("Enter end datetime (e.g., '2026-01-01 23:00'): ").strip()
            if not start_str or not end_str:
                raise ValueError("Both start and end datetime must be provided for range download")

            print("\nProcessing...")
            result = range_avgWindShear(lake_name, start_str, end_str, spacing)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(result.to_string(index=False))

        save = input("\nSave results to CSV? (y/n): ").strip().lower()
        if save == "y":
            if choice == "1":
                dt = result["dateTime"].iloc[0]
                timestamp = pd.Timestamp(dt).strftime("%Y%m%d_%H%M")
                filename = f"{lake_name.lower().replace(' ', '_')}_{timestamp}_wind_shear.csv"
            else:
                start_dt = pd.Timestamp(result["dateTime"].iloc[0])
                end_dt = pd.Timestamp(result["dateTime"].iloc[-1])
                filename = (
                    f"{lake_name.lower().replace(' ', '_')}_"
                    f"{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}_wind_shear.csv"
                )

            result.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

    except Exception as e:
        print(f"\nError: {e}")
        return


if __name__ == "__main__":
    interactive_main()

    # Example direct usage:
    # print(avgWindShear("Cayuga Lake", "2026-01-03 05:00"))
    # print(range_avgWindShear("Cayuga Lake", "2026-01-01 00:00", "2026-01-01 23:00"))

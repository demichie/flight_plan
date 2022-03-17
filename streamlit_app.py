import os, sys
import streamlit as st
from pyproj import Proj, transform
from pyproj import Transformer
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points
from scipy import interpolate
import datetime
from matplotlib.pyplot import cm
import pandas as pd
import pydeck as pdk
 
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
#import plugins 
from folium import plugins 
 
st.set_page_config(layout="wide")

 
step_to_home = 100 # m 
    
# when set to True, do not compute
# ground elevation from DEM to 
# obtain absolute elevation 
agl_flag = True

# Add custom base maps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True,
        show = False
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True,
        show = False
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True,
        show = False
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True,
        show = False
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    )
}




def save_csv(x_in,y_in,z_in,x_home,y_home,z_home,waypoints_file,max_wpt):

    csv_output_files = []

    i_csv = 0    
    filename = waypoints_file + '_{0:03}'.format(i_csv) + '.csv'    
    
    csv_output_files.append(filename) 
    f_out = open(filename, 'w')
    print('Open file ',filename) 
    
    
    lat2, lon2 = getGeoCoordinates(x_home, y_home)
    if ( agl_flag ):
        f_out.write(",".join([str(lat2),str(lon2),str(h_photo),str(heading)]))
    else:
        f_out.write(",".join([str(lat2),str(lon2),str(float(z_home)),str(heading)]))    
    f_out.write(str_nophoto)

    dist_to_home = np.sqrt( (x_in[0]-x_home)**2 + (y_in[0]-y_home)**2 )
        
    # add intermediate point along the path to new first point 
    # this is done to follow to topography
    for j,part_dist in enumerate(np.arange(step_to_home,dist_to_home,step_to_home)):
            
        dx_to_first = ( x_in[0] - x_home ) / dist_to_home 
        dy_to_first = ( y_in[0] - y_home ) / dist_to_home 
                
        x_part_to_home = x_home + part_dist*dx_to_first
        y_part_to_home = y_home + part_dist*dy_to_first
        
        if ( agl_flag ):
            z_part_to_home = h_photo
        else:
            z_part_to_home = f(x_part_to_home,y_part_to_home)
        
        lat2, lon2 = getGeoCoordinates(x_part_to_home, y_part_to_home)

        # add new point
        f_out.write(",".join([str(lat2),str(lon2),str(float(z_part_to_home)),str(heading)]))            
        f_out.write(str_nophoto)

                   
    dist = dist_to_home  
    
    i_wpt = 0
        
    for i, (x, y,z) in enumerate(zip(x_in[1:-1],y_in[1:-1],z_in[1:-1])):
    
        i_wpt += 1
                    
        dist += np.sqrt( (x_in[i+1]-x_in[i])**2 + (y_in[i+1]-y_in[i])**2 )
        time = dist / flight_speed + hovering_time/1000 * i_wpt
        dist_to_home = np.sqrt( (x-x_home)**2 + (y-y_home)**2 )
        # print('dist,dist_to_home',dist,dist_to_home)    
        time_to_home = dist_to_home / flight_speed
        # print('time,time_to_home',time,time_to_home)
                        
        if ( time + time_to_home > battery_time * 60 ) or ( i_wpt > max_wpt ):
        
            # add intermediate point along the path to home 
            # this is done to follow to topography
            
            dx_to_home = ( x_home - x_in[i] )  
            dy_to_home = ( y_home - y_in[i] ) 
                
            l = np.sqrt( dx_to_home**2 + dy_to_home**2 )
                
            dx_to_home = dx_to_home/ l
            dy_to_home = dy_to_home/ l
                
            for j,part_dist in enumerate(np.arange(step_to_home,l,step_to_home)):
                            
                x_part_to_home = x_in[i] + part_dist*dx_to_home
                y_part_to_home = y_in[i] + part_dist*dy_to_home
                if ( agl_flag ):
                    z_part_to_home = h_photo
                else:    
                    z_part_to_home = Z_grid = f(x_part_to_home,y_part_to_home)
                    
                lat2, lon2 = getGeoCoordinates(x_part_to_home, y_part_to_home)

                # add new point
                f_out.write(",".join([str(lat2),str(lon2),str(float(z_part_to_home)),str(heading)]))                    
                f_out.write(str_nophoto)
            
            
            # add home as last point and close file
            lat2, lon2 = getGeoCoordinates(x_home, y_home)
            
            f_out.write(",".join([str(lat2),str(lon2),str(float(z_home)),str(heading)]))
            f_out.write(str_photo)
            
            f_out.close()
            print('Close file ',filename)
            print('Flight distance',dist+dist_to_home)
            print('Flight time',str(datetime.timedelta(seconds=int(time+time_to_home))) ) 
            print('Number of waypoints',i_wpt)
            print('')

            # initialize new waipoint file    
            i_wpt = 1
            i_csv +=1     
            filename = waypoints_file + '_{0:03}'.format(i_csv) + '.csv'    
            csv_output_files.append(filename) 
    
            print('Open file ',filename) 
            f_out = open(filename, 'w')
            # add home point
            if ( agl_flag):
                f_out.write(",".join([str(lat2),str(lon2),str(h_photo),str(heading)]))
            else:    
                f_out.write(",".join([str(lat2),str(lon2),str(float(z_home)),str(heading)]))
            f_out.write(str_nophoto)
            # initialize distance           
            dist = np.sqrt( (x-x_home)**2 + (y-y_home)**2 )
            
            
            dist_to_home = np.sqrt( (x_in[i+1]-x_home)**2 + (y_in[i+1]-y_home)**2 )
        
            # add intermediate point along the path to new first point 
            # this is done to follow to topography
            for j,part_dist in enumerate(np.arange(step_to_home,dist_to_home,step_to_home)):
            
                dx_to_first = ( x_in[i+1] - x_home ) / dist_to_home 
                dy_to_first = ( y_in[i+1] - y_home ) / dist_to_home 
                
                x_part_to_home = x_home + part_dist*dx_to_first
                y_part_to_home = y_home + part_dist*dy_to_first
                if ( agl_flag ):
                    z_part_to_home = h_photo
                else:    
                    z_part_to_home = Z_grid = f(x_part_to_home,y_part_to_home)
                    
                lat2, lon2 = getGeoCoordinates(x_part_to_home, y_part_to_home)

                # add new point
                f_out.write(",".join([str(lat2),str(lon2),str(float(z_part_to_home)),str(heading)]))
                f_out.write(str_nophoto)
            
            
            xa = x_home
            ya = y_home
        
                    
        else:
        
            xa = x_in[i] 
            ya = y_in[i]
            
        # get lat,lon of new point    
        lat2, lon2 = getGeoCoordinates(x, y)
        # print(lat2,lon2,z_in[-1])
            
        # add new point
        if ( agl_flag ):
            f_out.write(",".join([str(lat2),str(lon2),str(h_photo),str(heading)]))
        else:
            f_out.write(",".join([str(lat2),str(lon2),str(z),str(heading)]))
            
        f_out.write(str_photo)
           
    # add intermediate point along the path to home 
    # this is done to follow to topography
    for j,part_dist in enumerate(np.arange(step_to_home,dist_to_home,step_to_home)):
            
        dx_to_home = ( x_home - x ) / dist_to_home 
        dy_to_home = ( y_home - y ) / dist_to_home 
                
        x_part_to_home = x + part_dist*dx_to_home
        y_part_to_home = y + part_dist*dy_to_home
        if ( agl_flag ):
            z_part_to_home = h_photo
        else:    
            z_part_to_home = Z_grid = f(x_part_to_home,y_part_to_home)
            
        lat2, lon2 = getGeoCoordinates(x_part_to_home, y_part_to_home)

        # add new point
        if ( agl_flag ):
            f_out.write(",".join([str(lat2),str(lon2),str(h_photo),str(heading)]))
        else:
            f_out.write(",".join([str(lat2),str(lon2),str(float(z_part_to_home)),str(heading)]))
        f_out.write(str_nophoto)
    
    lat2, lon2 = getGeoCoordinates(x_home, y_home)
    if ( agl_flag ):
        f_out.write(",".join([str(lat2),str(lon2),str(h_photo),str(heading)]))
    else:    
        f_out.write(",".join([str(lat2),str(lon2),str(float(z_home)),str(heading)]))    
    f_out.write(str_nophoto)
    
    f_out.close()
                    
    print('Close file ',filename)
    print('Flight time',str(datetime.timedelta(seconds=int(time))))
    print('Number of waypoints',i_wpt)
    
    return csv_output_files


def eval_dist(x_home,y_home,z_home,X1D,Y1D,Z1D,x_photo,y_photo,polygon):

    xTemp_in = []
    yTemp_in = []
    zTemp_in = []

    distTemp = 0.0  
    
    first = True
    
    for i, (x, y, z) in enumerate(zip(X1D,Y1D,Z1D)):
    
        dist = Point(x,y).distance(polygon)
    
        if dist < 0.5*np.minimum(x_photo,y_photo):
        
            xTemp_in.append(x)
            yTemp_in.append(y)
            zTemp_in.append(z)
            
            if first:
            
                # print('First point',x,y)
                first = False
        
            else:
              
                distTemp += np.sqrt( (xTemp_in[-1]-xTemp_in[-2])**2 + 
                               (yTemp_in[-1]-yTemp_in[-2])**2 )

    return distTemp, xTemp_in, yTemp_in, zTemp_in                   

def create_grid(polygon,x_home,y_home,z_home,X_grid,Y_grid,Z_grid,x_photo,y_photo,h_flag,v_flag,first):

    nx = X_grid.shape[1]
    ny = X_grid.shape[0]

    if h_flag:

        for j in np.arange(first,ny,2):

                X_grid[j,:] = np.flip(X_grid[j,:])
                Z_grid[j,:] = np.flip(Z_grid[j,:])

        distTemp, xTemp_in, yTemp_in, zTemp_in  = eval_dist(x_home,y_home,z_home,
                                  X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel(),
                                  x_photo,y_photo,polygon)
     
    if v_flag:
         
         # horiz. lines, starting from top_left
    
        for i in np.arange(first,nx,2):

                Y_grid[:,i] = np.flip(Y_grid[:,i])
                Z_grid[:,i] = np.flip(Z_grid[:,i])

        distTemp, xTemp_in, yTemp_in, zTemp_in  = eval_dist(x_home,y_home,z_home,
                                X_grid.ravel(order='F'), Y_grid.ravel(order='F'),
                                Z_grid.ravel(order='F'),x_photo,y_photo,polygon)
    

    return distTemp,xTemp_in,yTemp_in,zTemp_in

def select_grid(distH1,xH1,yH1,zH1,distH2,xH2,yH2,zH2,
                distV1,xV1,yV1,zV1,distV2,xV2,yV2,zV2,
                x_home,y_home,z_home,double_grid):
                
                
    if double_grid:
    
        # H1 + V1
        dist1 = np.sqrt( (xH1[0]-x_home)**2 + (yH1[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH1[-1]-xV1[0])**2 + (yH1[-1]-yV1[0])**2 )        
        dist3 = np.sqrt( (xV1[-1]-x_home)**2 + (yV1[-1]-y_home)**2 )
        
        distOpt = dist1+distH1+dist2+distV1+dist3
        
        #print('Total distance with H1+V1',distOpt)
        #print(dist1,dist2,dist3)

        x_in = [x_home]+xH1+xV1+[x_home]
        y_in = [y_home]+yH1+yV1+[y_home]
        z_in = [z_home]+zH1+zV1+[z_home]
        
        # H1 + V1.reverse        
        dist1 = np.sqrt( (xH1[0]-x_home)**2 + (yH1[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH1[-1]-xV1[-1])**2 + (yH1[-1]-yV1[-1])**2 )        
        dist3 = np.sqrt( (xV1[0]-x_home)**2 + (yV1[0]-y_home)**2 )
        
        distTemp = dist1+distH1+dist2+distV1+dist3
        
        #print('Total distance with H1+V1.reverse',distTemp)
        #print(dist1,dist2,dist3)
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH1+xV1[::-1]+[x_home]
            y_in = [y_home]+yH1+yV1[::-1]+[y_home]
            z_in = [z_home]+zH1+zV1[::-1]+[z_home]
            distOpt = distTemp
        
        # H1 + V2    
        dist1 = np.sqrt( (xH1[0]-x_home)**2 + (yH1[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH1[-1]-xV2[0])**2 + (yH1[-1]-yV2[0])**2 )        
        dist3 = np.sqrt( (xV2[-1]-x_home)**2 + (yV2[-1]-y_home)**2 )
            
        distTemp = dist1+distH1+dist2+distV2+dist3

        #print('Total distance with H1+V2',distTemp)
        #print(dist1,dist2,dist3)
        
        if ( distTemp < distOpt ):

            x_in = [x_home]+xH1+xV2+[x_home]
            y_in = [y_home]+yH1+yV2+[y_home]
            z_in = [z_home]+zH1+zV2+[z_home]
            distOpt = distTemp
        
        # H1 + V2.reverse    
        dist1 = np.sqrt( (xH1[0]-x_home)**2 + (yH1[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH1[-1]-xV2[-1])**2 + (yH1[-1]-yV2[-1])**2 )        
        dist3 = np.sqrt( (xV2[0]-x_home)**2 + (yV2[0]-y_home)**2 )
        
        distTemp = dist1+distH1+dist2+distV2+dist3
        
        #print('Total distance with H1+V2.reverse',distTemp)
        #print(dist1,dist2,dist3)

        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH1+xV2[::-1]+[x_home]
            y_in = [y_home]+yH1+yV2[::-1]+[y_home]
            z_in = [z_home]+zH1+zV2[::-1]+[z_home]
            distOpt = distTemp

        # H2 + V1
        dist1 = np.sqrt( (xH2[0]-x_home)**2 + (yH2[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH2[-1]-xV1[0])**2 + (yH2[-1]-yV1[0])**2 )        
        dist3 = np.sqrt( (xV1[-1]-x_home)**2 + (yV1[-1]-y_home)**2 )
        
        distTemp = dist1+distH2+dist2+distV1+dist3
        
        #print('Total distance with H2+V1',distTemp)
        #print(dist1,dist2,dist3)
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH2+xV1+[x_home]
            y_in = [y_home]+yH2+yV1+[y_home]
            z_in = [z_home]+zH2+zV1+[z_home]
            distOpt = distTemp

        # H2 + V1.reverse        
        dist1 = np.sqrt( (xH2[0]-x_home)**2 + (yH2[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH2[-1]-xV1[-1])**2 + (yH2[-1]-yV1[-1])**2 )        
        dist3 = np.sqrt( (xV1[0]-x_home)**2 + (yV1[0]-y_home)**2 )
        
        distTemp = dist1+distH2+dist2+distV1+dist3
        
        #print('Total distance with H2+V1.reverse',distTemp)
        #print(dist1,dist2,dist3)
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH2+xV1[::-1]+[x_home]
            y_in = [y_home]+yH2+yV1[::-1]+[y_home]
            z_in = [z_home]+zH2+zV1[::-1]+[z_home]
            distOpt = distTemp
        
        # H2 + V2    
        dist1 = np.sqrt( (xH2[0]-x_home)**2 + (yH2[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH2[-1]-xV2[0])**2 + (yH2[-1]-yV2[0])**2 )        
        dist3 = np.sqrt( (xV2[-1]-x_home)**2 + (yV2[-1]-y_home)**2 )
            
        distTemp = dist1+distH2+dist2+distV2+dist3
        
        #print('Total distance with H2+V2',distTemp)
        #print(dist1,dist2,dist3)
        
        if ( distTemp < distOpt ):

            x_in = [x_home]+xH2+xV2+[x_home]
            y_in = [y_home]+yH2+yV2+[y_home]
            z_in = [z_home]+zH2+zV2+[z_home]
            distOpt = distTemp
        
        # H2 + V2.reverse    
        dist1 = np.sqrt( (xH2[0]-x_home)**2 + (yH2[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH2[-1]-xV2[-1])**2 + (yH2[-1]-yV2[-1])**2 )        
        dist3 = np.sqrt( (xV2[0]-x_home)**2 + (yV2[0]-y_home)**2 )
        
        distTemp = dist1+distH2+dist2+distV2+dist3
        
        #print('Total distance with H2+V2.reverse',distTemp)
        #print(dist1,dist2,dist3)
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH2+xV2[::-1]+[x_home]
            y_in = [y_home]+yH2+yV2[::-1]+[y_home]
            z_in = [z_home]+zH2+zV2[::-1]+[z_home]
            distOpt = distTemp

    else:
    
        distOpt = 1.e+10
    
        #H1
        dist1 = np.sqrt( (xH1[0]-x_home)**2 + (yH1[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH1[-1]-x_home)**2 + (yH1[-1]-y_home)**2 )        
        
        distTemp = dist1+distH1+dist2
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH1+[x_home]
            y_in = [y_home]+yH1+[y_home]
            z_in = [z_home]+zH1+[z_home]
            distOpt = distTemp
        
        #H2
        dist1 = np.sqrt( (xH2[0]-x_home)**2 + (yH2[0]-y_home)**2 )        
        dist2 = np.sqrt( (xH2[-1]-x_home)**2 + (yH2[-1]-y_home)**2 )        
        
        distTemp = dist1+distH2+dist2
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xH2+[x_home]
            y_in = [y_home]+yH2+[y_home]
            z_in = [z_home]+zH2+[z_home]
            distOpt = distTemp
        
        #V1
        dist1 = np.sqrt( (xV1[0]-x_home)**2 + (yV1[0]-y_home)**2 )        
        dist2 = np.sqrt( (xV1[-1]-x_home)**2 + (yV1[-1]-y_home)**2 )        
        
        distTemp = dist1+distV1+dist2
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xV1+[x_home]
            y_in = [y_home]+yV1+[y_home]
            z_in = [z_home]+zV1+[z_home]
            distOpt = distTemp
        
        #V2
        dist1 = np.sqrt( (xV2[0]-x_home)**2 + (yV2[0]-y_home)**2 )        
        dist2 = np.sqrt( (xV2[-1]-x_home)**2 + (yV2[-1]-y_home)**2 )        
        
        distTemp = dist1+distV2+dist2
        
        if ( distTemp < distOpt ):
        
            x_in = [x_home]+xV2+[x_home]
            y_in = [y_home]+yV2+[y_home]
            z_in = [z_home]+zV2+[z_home]
            distOpt = distTemp
                            
    return x_in,y_in,z_in            


def read_csv(csv_file):

    print('csv_file',csv_file)
    array2 = np.genfromtxt(csv_file, delimiter=',',skip_header=1)
    
    points = []
    
    for j in range(array2.shape[0]):
        
        x,y,z = t2.transform(array2[j,1],array2[j,0],0.0)
        points.append((x,y))
        
    points.append(points[0])
    
    print('Number of points read',len(points))
    
    return points  

def getGeoCoordinates(x, y):
    # lat, lon, depth = transform(fleast_m,wgs84,x*FT2M,y*FT2M,0.0)
    lat, lon, depth = t1.transform(x,y,0.0)
    return lon, lat


# convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
def convert_wgs_to_utm(lon: float, lat: float):

    import math
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code


def main(array,csv_file,option,dx_perc_overlap,
                 dy_perc_overlap,cm_per_pixel,battery_time,
                 flight_speed,hovering_time,heading,home_side,
                 res_x,res_y,fov,max_wpt):
    
    points = []
    
    for j in range(array.shape[0]):
        
        x,y,z = t2.transform(array[j,1],array[j,0],0.0)
        points.append((x,y))
        
    points.append(points[0])
    
    print('Number of points read',len(points))
    
    if option == 'Double grid':
    
        double_grid = True
        
    else:
    
        double_grid = False
                   
        
    ln = LineString(points)
    minx, miny, maxx, maxy = ln.bounds
    
    if (home_side == 'N'):
    
        x_home = 0.5*(minx+maxx)
        y_home = maxy + 10.0
        maxy = y_home + 10.0
        
    elif (home_side == 'S'):
    
        x_home = 0.5*(minx+maxx)
        y_home = miny - 10.0     
        miny = y_home - 10.0

    elif (home_side == 'W'):
    
        x_home = minx - 10.0
        y_home = 0.5*(miny+maxy)
        minx = x_home - 10.0
        
    elif (home_side == 'E'):
    
        x_home = maxx + 10.0
        y_home = 0.5*(miny+maxy)
        maxx = x_home + 10.0

    elif (home_side == 'C'):
    
        x_home = 0.5 * ( maxx + minx )
        y_home = 0.5 * ( miny + maxy )

                
    print('Area Bounding Box',minx, miny, maxx, maxy)

    xp = []
    yp = []
    
    
    for p in points:
        xp.append(p[0])
        yp.append(p[1])
    
    
    polygon = Polygon(points[0:-1])
    
    x_grid = np.arange(minx-2.0*dx_photo,maxx+dx_photo,dx_photo)
    y_grid = np.arange(miny-2.0*dy_photo,maxy+dy_photo,dy_photo)
    
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    
    X_grid,Y_grid = np.meshgrid(x_grid,y_grid)
    
    if ( agl_flag ):
    
        Z_grid = np.zeros_like(X_grid) + h_photo
        z_home = h_photo
       
    else:
    
        # build the interpolating function for the elevation      
        xc = X[0,:]
        yc = Y[:,0]
    
        f = interpolate.interp2d(xc, yc, h, kind='linear')
    
        Z_grid = f(X_grid[0,:],Y_grid[:,0])
        z_home = f(x_home,y_home)
        
        
    waypoints_file = csv_file.replace('.csv','_')+'waypoint'
                
    distH1,xH1_in,yH1_in,zH1_in = create_grid(polygon,x_home,y_home,z_home,X_grid,Y_grid,Z_grid,
                                 x_photo,y_photo,h_flag=True,v_flag=False,first=0)    

    distH2,xH2_in,yH2_in,zH2_in = create_grid(polygon,x_home,y_home,z_home,X_grid,Y_grid,Z_grid,
                                 x_photo,y_photo,h_flag=True,v_flag=False,first=1)    

    X_grid += 0.5*dx_photo
    Y_grid += 0.5*dy_photo
    
    if ( agl_flag ):
    
        Z_grid = np.zeros_like(X_grid) + h_photo
        
    else:
        
        Z_grid = f(X_grid[0,:],Y_grid[:,0])
        
    distV1,xV1_in,yV1_in,zV1_in = create_grid(polygon,x_home,y_home,z_home,X_grid,Y_grid,Z_grid,
                                 x_photo,y_photo,h_flag=False,v_flag=True,first=0)    

    distV2,xV2_in,yV2_in,zV2_in = create_grid(polygon,x_home,y_home,z_home,X_grid,Y_grid,Z_grid,
                                 x_photo,y_photo,h_flag=False,v_flag=True,first=1)   
                                 
    
    x_in,y_in,z_in = select_grid(distH1,xH1_in,yH1_in,zH1_in,
                                 distH2,xH2_in,yH2_in,zH2_in,
                                 distV1,xV1_in,yV1_in,zV1_in,                 
                                 distV2,xV2_in,yV2_in,zV2_in,
                                 x_home,y_home,z_home,double_grid)

    csv_output_files = save_csv(x_in,y_in,z_in,x_home,y_home,z_home,waypoints_file,max_wpt)
        
    lat = []
    lon = []
    
    for i, (x, y) in enumerate(zip(x_in,y_in)):
        
        lat_i, lon_i = getGeoCoordinates(x, y)
        lat.append(lat_i)
        lon.append(lon_i)
        
    df = pd.DataFrame(np.column_stack((lat,lon)),columns=['lat', 'lon'])

    lat = []
    lon = []

    for i, (x, y) in enumerate(zip(xp,yp)):
        
        lat_i, lon_i = getGeoCoordinates(x, y)
        lat.append(lat_i)
        lon.append(lon_i)
        
    df2 = pd.DataFrame(np.column_stack((lat,lon)),columns=['lat', 'lon'])
    
    # Adding code so we can have map default to the center of the data
    midpoint = (np.average(df['lat']), np.average(df['lon']))
    
    c = []

    n = len(csv_output_files)
    import matplotlib
    cmap = cm.prism(np.linspace(0, 1, n))
    
    layers = []
    
    i_file = 0
    for filename in csv_output_files:
    
        my_data = np.genfromtxt(filename, delimiter=',')
        
        path = []
        name = []
        path_new = []

        for i in range(my_data.shape[0]):
        
            path_new.append([my_data[i,1],my_data[i,0]])
        
              
        path.append(path_new)
        name.append(filename)
        
        
        data = {'name':name,
            'path':path}
        df_path = pd.DataFrame(data)
    
        ci = cmap[i_file]
        ci_str = str((np.round_(255 * ci[0:4], decimals=0)).astype(int))
        
        print(ci_str)
        i_file +=1

        path_layer = pdk.Layer(
            type="PathLayer",
            data=df_path,
            pickable=True,
            get_color=ci_str,
            width_scale=1,
            width_min_pixels=1,
            get_path="path",
            get_width=1,
        )
    
        layers.append(path_layer)

    scatter_layer1 = pdk.Layer(
                  'ScatterplotLayer',
                  data=df,
                  get_position='[lon, lat]',
                  get_color='[200, 200, 200, 250]',
                  get_radius=2,)

    layers.append(scatter_layer1)
                  
    scatter_layer2 = pdk.Layer(
                  'ScatterplotLayer',
                  data=df2,
                  get_position='[lon, lat]',
                  get_color='[230, 230, 230, 255]',
                  get_radius=4,)

    layers.append(scatter_layer2)

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-streets-v11',
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=14,
            pitch=0,
            height=600, width=900),
            layers=layers
                ,         ))


    for filename in csv_output_files:

        with open(filename) as f:
        
           st.download_button('Download '+filename, f, file_name=filename)  # Defaults to 'text/plain'

    
if __name__ == '__main__':

   
    m = folium.Map(location=[42.81233, 10.31297], zoom_start=12)

    # Add custom basemaps
    basemaps['Google Maps'].add_to(m)
    basemaps['Google Satellite Hybrid'].add_to(m)
    basemaps['Google Terrain'].add_to(m)

    # Add a layer control panel to the map.
    m.add_child(folium.LayerControl())

    #fullscreen
    plugins.Fullscreen().add_to(m)

    Draw(export=False,draw_options={'polygon': {'allowIntersection': False},
                                'polyline': False,'rectangle':False,
                                'circle': False, 'marker': False, 
                                'circlemarker': False},
                      edit_options={'edit': False},
        ).add_to(m)

    output_map = st_folium(m, width=900, height=600)


    coords = []
    area = 0.0

    if output_map is not None:

        if output_map.get("all_drawings") is not None:

            # print( 'len', len(output.get("all_drawings")) )

            if (output_map.get("all_drawings")[0]).get("features"):
         
                features = (output_map.get("all_drawings")[0]).get("features")
    
                for i in range(len(features)):
        
                    typeGeo = features[i].get("geometry").get("type") 
          
                    if typeGeo == "Polygon":
             
                        # print('i',i)
        
                        coords = features[i].get("geometry").get("coordinates")[0]


    if coords:

        lat_coord = []
        lon_coord = []

        array = np.zeros((len(coords),2))

        for coord in coords:

            lat_coord.append(float(coord[0]))
            lon_coord.append(float(coord[1]))


        array[:,0] = lon_coord
        array[:,1] = lat_coord

        utm_code = convert_wgs_to_utm(lat_coord[0],lon_coord[0])    
        proj = 'EPSG:'+utm_code
        print('Projection',proj)   
        
        t1 = Transformer.from_proj(
                proj,'+proj=longlat +datum=WGS84 +no_defs +type=crs',
                always_xy=True,
            )

        t2 = Transformer.from_proj(
                '+proj=longlat +datum=WGS84 +no_defs +type=crs',
                proj,
                always_xy=True,
            )
            
        pts = []    
            
        for j in range(array.shape[0]):
        
            x,y,z = t2.transform(array[j,1],array[j,0],0.0)
            pts.append((x,y))
        
        pts.append(pts[0])

        polygon = Polygon(pts[0:-1])
        area = polygon.area
        (minx, miny, maxx, maxy) = polygon.bounds


    # csv_file = st.sidebar.file_uploader("Select a .csv file", type='csv', accept_multiple_files=False)

    csv_name = st.sidebar.text_input('Flight plan name', 'myFlight')
    
    csv_file = csv_name.replace(' ','_')+'.csv'

    option = st.sidebar.radio('Select the grid type:',
                  ['Single grid',
                   'Double grid'])
                   
    dx_perc_overlap = st.sidebar.slider(
     "Horizontal overlap percentage",1,99,50)

    dy_perc_overlap = st.sidebar.slider(
     "Vertical overlap percentage",1,99,50)

    cm_per_pixel = st.sidebar.number_input("Centimeters per pixel", min_value=0.1, max_value=None, value=2.0, step=0.1)

    max_wpt = st.sidebar.number_input("Maximum number of waypoints for flight", min_value=10, max_value=None, value=95, step=1)


    battery_time = st.sidebar.slider(
     "Maximum duration of single flight",1,60,20)

    flight_speed = st.sidebar.slider(
     "Flight speed (m/s)",0.1,20.0,5.0)

    hovering_time = st.sidebar.slider(
     "Hovering time (ms)",0,5000,1000)

    # camera parameters
    heading = st.sidebar.slider(
     "Camera heading (angle)",0,359,0)

    home_side = st.sidebar.radio('Select home location:',
                  ['E','W','N','S','C'])

    # camera parameters
    res_x = st.sidebar.slider(
     "Camera horizontal resolution (pixels)",1000,10000,4000)

    res_y = st.sidebar.slider(
     "Camera vertical resolution (pixels)",1000,10000,3000)

    fov = st.sidebar.slider(
     "Camera field of view (pixels)",1.0,180.0,83.0)

    fov_rad = fov/180.0*np.pi
    photo_ratio = res_x/res_y
        
    x_photo = res_x * ( cm_per_pixel / 100.0)
    y_photo = res_y * ( cm_per_pixel / 100.0)
    
    h_photo = 0.5*x_photo / np.tan(0.5*fov_rad)
    
    dx_photo = x_photo * ( 1.0 - dx_perc_overlap / 100.0 )
    dy_photo = y_photo * ( 1.0 - dy_perc_overlap / 100.0 )
    
    dxy_photo = dx_photo*dy_photo
    
    if ( area > 0 ):
    
        area = np.abs((maxx-minx+x_photo)*(maxy-miny+y_photo))

        n_photo = area / dxy_photo
        if option == 'Double grid':
            n_photo *=2
        
        print('n_photo',n_photo)
        st.text('Approx. number of photos: '+str(int(np.floor(n_photo))))
    
    x_pic = [ -0.5*x_photo,0.5*x_photo,0.5*x_photo,-0.5*x_photo,-0.5*x_photo]
    y_pic = [ -0.5*y_photo,-0.5*y_photo,0.5*y_photo,0.5*y_photo,-0.5*y_photo]


    if ( hovering_time > 0 ):

        str_photo_asl = ',0,0,2,-90,0,'+str(int(hovering_time))+\
                        ',1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,'+str(int(flight_speed))+',0,0,0,0,-1,-1\n'
        
    else:
    
        str_photo_asl = ',0,0,2,-90,1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,'+str(int(flight_speed))+',0,0,0,0,-1,-1\n'
        
    str_nophoto_asl = ',0,0,2,-90,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,'+str(int(flight_speed))+',0,0,0,0,-1,-1\n'

    if ( hovering_time > 0 ):

        str_photo_agl = ',0,0,2,-90,0,'+str(int(hovering_time))+\
                        ',1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,1,'+str(int(flight_speed))+',0,0,0,0,-1,-1\n'
                        
    else:
    
        str_photo_agl = ',0,0,2,-90,1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,1,'+str(int(flight_speed))+',0,0,0,0,-1,-1\n'    
                        
    str_nophoto_agl = ',0,0,2,-90,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,1,'+str(int(flight_speed))+',0,0,0,0,-1,-1\n'
    
    if agl_flag:

        str_photo = str_photo_agl
        str_nophoto = str_nophoto_agl
    
    else:
        
        str_photo = str_photo_asl
        str_nophoto = str_nophoto_asl
    
        X,Y,h,x_min,x_max,y_min,y_max = read_dem(source)
        delta_x = X[0,1]-X[0,0]
        delta_y = Y[1,0]-Y[0,0]


    if st.sidebar.button('Run'):
    
        if n_photo == 0:
        
            st.text('Draw a polygon on the map with the tool')

        elif n_photo < 500:


            t1 = Transformer.from_proj(
                proj,'+proj=longlat +datum=WGS84 +no_defs +type=crs',
                always_xy=True,
            )

            t2 = Transformer.from_proj(
                '+proj=longlat +datum=WGS84 +no_defs +type=crs',
                proj,
                always_xy=True,
            )

       
            main(array,csv_file,option,dx_perc_overlap,
                 dy_perc_overlap,cm_per_pixel,battery_time,
                 flight_speed,hovering_time,heading,home_side,
                 res_x,res_y,fov,max_wpt)
            
        else:
        
            st.text('The number of photo for the selected area is too large. Increase cm per pixel')    
    
    
    
    

## ----setup, include=FALSE-------------------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,collapse=TRUE)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library(sf)      # vector data package
library(terra)   # raster data package
library(dplyr)   # tidyverse package for data frame manipulation
library(spData)  # loads datasets used here
library(here)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elev = rast(system.file("raster/elev.tif", package = "spData"))
grain = rast(system.file("raster/grain.tif", package = "spData"))


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# filter out Canterbury
canterbury = nz %>% filter(Name == "Canterbury")
# subset the high points that "intersect" the above.
(canterbury_height = nz_height[canterbury, ])


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

tmap::tm_shape(canterbury) + tmap::tm_borders() +
tmap::tm_shape(canterbury_height) + tmap::tm_symbols(shape = 17, col = "blue", size = .2)




## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# intersect heights and Canterbury
sel_sgbp = st_intersects(x = nz_height, y = canterbury)

class(sel_sgbp)

sel_sgbp

# transform this into a logical
# lengths: applied to each element in a list

sel_logical = lengths(sel_sgbp) > 0

# carry out the subsetting

canterbury_height2 = nz_height[sel_logical, ]


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
canterbury_height3 = nz_height %>%
  st_filter(y = canterbury, .predicate = st_intersects)


## ----fig1, fig.asp= .5----------------------------------------------------------------------------------------------------------------------------------------------------------------
polygon_matrix = cbind(
  x = c(0, 0, 1, 1,   0),
  y = c(0, 1, 1, 0.5, 0)
)
polygon_sfc = st_sfc(st_polygon(list(polygon_matrix)))

tmap::tm_shape(polygon_sfc) + tmap::tm_polygons() + tmap::tm_grid(lines = FALSE)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
line_matrix = cbind(
  x = c(0.4, 1),
  y = c(0.2, 0.5))

line_sfc = st_sfc(st_linestring(line_matrix))

# create a data frame of points
(point_df = data.frame(
  x = c(0.2, 0.7, 0.4),
  y = c(0.1, 0.2, 0.8)
)) 
point_sf = st_as_sf(point_df, coords = c("x", "y")) %>% 
  tibble::rowid_to_column("ID") %>% mutate(ID=as.character(ID))



## ----fig2, fig.asp= .5----------------------------------------------------------------------------------------------------------------------------------------------------------------

oldw <- getOption("warn")
options(warn = -1)

tmap::tm_shape(polygon_sfc) + tmap::tm_polygons() + 
  tmap::tm_shape(line_sfc) + tmap::tm_lines(scale = 10) +
  tmap::tm_shape(point_sf) + 
  tmap::tm_dots(scale=5, legend.show = F,col = "turquoise", alpha = .7, size= .1) +
  tmap::tm_text("ID", size = .5) +
  tmap::tm_grid(lines = FALSE)

options(warn = oldw)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# The code below sets sparse=FALSE to coerce the output 
# into a logical vector, instead of a sparse matrix.

st_intersects(point_sf, polygon_sfc, sparse = FALSE)

# A sparse matrix is a list of vectors with
# empty elements where a match doe not exists.



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st_within(point_sf, polygon_sfc)
st_touches(point_sf, polygon_sfc)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# note [, 1] converts the result into a vector:

st_disjoint(point_sf, polygon_sfc, sparse = FALSE)[, 1]



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st_is_within_distance(point_sf, polygon_sfc,
                      dist = 0.2, sparse = FALSE)[, 1]


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(2018) # set seed for reproducibility
(bb = st_bbox(world)) # the world's bounds
random_df = data.frame(
  x = runif(n = 10, min = bb[1], max = bb[3]),
  y = runif(n = 10, min = bb[2], max = bb[4])
)
random_points = random_df %>% 
  st_as_sf(coords = c("x", "y")) %>% # set coordinates
  st_set_crs("EPSG:4326") # set geographic CRS


## ----fig5, fig.asp= .5----------------------------------------------------------------------------------------------------------------------------------------------------------------

st_crs(world) <- 4326

tmap::tm_shape(world) + tmap::tm_borders() +
  tmap::tm_shape(random_points) + tmap::tm_symbols(shape = 17, col = "blue", size = .2)




## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# find the countries "touched" by random points
(world_random = world[random_points,])



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# find the points that touch a country.
(random_joined = 
   st_join(random_points, select(world,name_long),
           join = st_intersects))


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tmap::tm_shape(world)+tmap::tm_borders() +
  tmap::tm_shape(world_random) + tmap::tm_polygons("name_long") +
  tmap::tm_shape(random_points) + tmap::tm_symbols(shape = 17, col = "blue", size = .2)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(st_geometry(cycle_hire), col = "blue", main = "London Cycle points: official-blue, OpenStreetMap-red")
plot(st_geometry(cycle_hire_osm), add = TRUE, pch = 3, col = "red")


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st_touches(cycle_hire, cycle_hire_osm, sparse = FALSE) %>% 
any()



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
head(cycle_hire)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
head(cycle_hire_osm)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

(sel = st_is_within_distance(cycle_hire, cycle_hire_osm, dist = 20))



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
summary(lengths(sel) > 0)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

aux = st_join(cycle_hire, select(cycle_hire_osm,capacity), join = st_is_within_distance,
            dist = 20)
nrow(cycle_hire)
nrow(aux)
head(aux)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
aux = aux %>% 
  group_by(id) %>% 
  summarize(capacity = mean(capacity))
nrow(aux) == nrow(cycle_hire)
#> [1] TRUE


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(cycle_hire_osm["capacity"], main="actual capacity")


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(aux["capacity"], main= "estimated capacity")


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(nz_agg2 = st_join(x = nz, y = nz_height, join = st_intersects))


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
nz_agg2 = nz_agg2 %>%
  group_by(Name) %>%
  summarize(elevation = mean(elevation, na.rm = TRUE)) 
  head(nz_agg2)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

tmap::tm_shape(nz)+tmap::tm_borders() +
  tmap::tm_shape(nz_agg2) + tmap::tm_polygons("elevation")




## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
iv = incongruent %>% select(value) # keep only the values to be transferred
agg_aw = st_interpolate_aw(iv, aggregating_zones,
                           ext = TRUE)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(iv)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plot(agg_aw)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# with respect to elevation,
# take the top 1 observation.

nz_heighest = nz_height %>% top_n(n = 1, wt = elevation)
canterbury_centroid = st_centroid(canterbury)
st_distance(nz_heighest, canterbury_centroid)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
co = filter(nz, Name=="Canterbury" | Name=="Otago")
st_distance(nz_height[1:3, ], co)


## ----fig4, fig.asp= .5----------------------------------------------------------------------------------------------------------------------------------------------------------------
tmap::tm_shape(st_geometry(co)[2]) +tmap::tm_borders() +
tmap::tm_shape(st_geometry(nz_height)[2:3]) + tmap::tm_symbols(shape = 14, col = "blue", size = 2, alpha = .3)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
id = cellFromXY(elev, xy = matrix(c(0.1, 0.1), ncol = 2))
elev[id]
# the same as
terra::extract(elev, matrix(c(0.1, 0.1), ncol = 2))


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# raster with only 1s across 9 cells

clip = rast(xmin = 0.9, xmax = 1.8, ymin = -0.45, ymax = 0.45,
            resolution = 0.3, vals = rep(1, 9))
elev[clip]
# we can also use extract
# terra::extract(elev, ext(clip))
#plot(elev)
#plot(clip, add=T,col="blue")


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(elev)
plot(clip, add=T,col="blue")


## ----fig6, fig.asp= .2----------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(elev)
plot(elev[1:2, drop = FALSE] )


## ----fig7, fig.asp= .5----------------------------------------------------------------------------------------------------------------------------------------------------------------

# create raster mask
rmask = elev
values(rmask) = sample(c(NA, TRUE), 36, replace = TRUE)

# spatial subsetting

plot(mask(elev, rmask))                   # with mask()



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elev[elev < 20] = 100
plot(elev)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data(elev)
elev_sum = elev + elev
elev_square = elev^2
elev_log = log(elev)
elev_5 = elev > 5



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pal <- colorRampPalette(c("white","red"))
plot(elev_sum, col=pal(15))



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pal <- colorRampPalette(c("white","red"))

plot(elev_square,col=pal(15))
plot(elev_5, col=pal(7))



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pal <- colorRampPalette(c("white","red"))

plot(elev_5, col=pal(7))



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rcl = matrix(c(0, 12, 1, 12, 24, 2, 24, 36, 3),
             ncol = 3, byrow = TRUE)
rcl


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
recl = classify(rast(elev), rcl = rcl)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
multi_raster_file = system.file("raster/landsat.tif", package = "spDataLarge")
multi_rast = rast(multi_raster_file)

# The raster object has four satellite bands - blue, green, red,
# and near-infrared (NIR).
# Our next step should be to implement the NDVI formula into an R function:

# create a function that takes the two
# types of light and computes NVDI

ndvi_fun = function(nir, red){
  (nir - red) / (nir + red)
}


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ndvi_rast = lapp(multi_rast[[c(4, 3)]], fun = ndvi_fun)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(ndvi_rast)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# First construct the kernel (AKA mooving window)

(w = matrix(1, nrow = 3, ncol = 3))

# Now apply focal to the elevation data. 

r_focal = focal(elev, w, fun = min)


## ---- figures-side, fig.show="hold", out.width="25%"----------------------------------------------------------------------------------------------------------------------------------

par(mar = c(4, 4, .3, .3))
plot(elev)
plot(r_focal)
# Use terra's values() to visualize the output.
matrix(terra::values(elev),nrow = 6, ncol = 6)
matrix(terra::values(r_focal),nrow = 6, ncol = 6)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# first let's check the grain's structure
dim(grain)
matrix(values(grain),nrow = 6,ncol = 6)
cats(grain)
# let's apply the zonal function to elev using grain as a filter provider.
# it tell us about the elevation per grain-type/size
z = zonal(rast(elev), grain, fun = "mean")
z


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library(maptools) # for data below
data(wrld_simpl)

# Here we use the raster package, but could have used terra instaed.
# Create a raster template. 
# (set the desired grid resolution with res)

r <- raster::raster(xmn=-180, xmx=180, ymn=-90, ymx=90, res=1)

# Rasterize the countries polygons: 1 for land cells
# Nas for water.

r2 <- raster::rasterize(wrld_simpl, r, 1)



## ----fig20, fig.asp= .5---------------------------------------------------------------------------------------------------------------------------------------------------------------

# the condition below is a land-sea indicator
# cond = is.na(r2)

# set land pixels to NA # water-pixels to sea
# maskvalue: what we want for water.
# updatevalue: what we want for land

r3 <- mask(is.na(r2), r2, maskvalue=1, updatevalue=NA)

# Calculate distance to nearest non-NA pixel
# don't run it, it takes too long

d <- raster::distance(r3)

plot(d)



## ---- fig12, figures-side, fig.show="hold", out.width="25%"---------------------------------------------------------------------------------------------------------------------------
aut = geodata::elevation_30s(country = "AUT", path = tempdir())
ch = geodata::elevation_30s(country = "CHE", path = tempdir())
aut_ch = merge(aut, ch)
par(mar = c(4, 4, .3, .3))
plot(aut)
plot(ch)
plot(aut_ch)
# terra’s merge() command combines two images,
# and in case they overlap, it uses the value of the first raster.


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library(raster)
# find out the ISO_3 code of Spain
dplyr::filter(ccodes(), NAME %in% "Spain")
# retrieve a data -elevation-model of Spain
dem = getData("alt", country = "ESP", mask = FALSE)
# change the resolution to decrease computing time
agg = aggregate(dem, fact = 5)
#spain polygons
esp = getData("GADM", country = "ESP", level = 1)



## ---- fig30, figures-side, fig.show="hold", out.width="25%"---------------------------------------------------------------------------------------------------------------------------

par(mar = c(4, 4, .3, .3))
plot(dem)
# visualize NAs
plot(is.na(agg))



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# construct a distance input raster
# we have to set the land cells to NA 
# and the sea cells to an arbitrary value since 
# raster::distance computes the distance to
# the nearest non-NA cell
dist = is.na(agg)
cellStats(dist, summary)
# convert land cells into NAs and sea cells into 1s
dist[dist == FALSE] = NA
dist[dist == TRUE] = 1
#plot(dist)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(dist)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# compute distance to nearest non-NA cell
dist = raster::distance(dist)
# erase cells' contents that are not mainland. 
dist = mask(dist, esp)
agg = mask(agg, esp)
# convert distance into km
dist = dist / 1000
# now let's weight each 100 altitudinal m.
# by an additional distance of 10 km.
agg[agg < 0] = 0 # only positive elevation.
# now create a raster with 
# elev-weighted distance data.
weight = dist + agg / 100 * 10 
#plot(weight - dist)



## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plot(weight - dist)


# ##################################################################################################################
# 
# 
# 
#                                        IntELiMon_1_1_1 Automated 
#                                      Lidar Point Cloud Processing
#                       
# 
# 
# Products: ---------------------------------------------------------------------------------------------------
# Canopy Height Model (TIF) - Raster image of the height of the normalized and cropped scan
#
# Digital Terrain Model (DTM) - Raster image of the digital terrain model 
#
# ---------------------------------------------------------------------------------------------------
# Inventory Table (CSV)- Table used to identify location of each tree (center of point cloud
#                        is projected at (0,0)), the radius of the trunk at 1.3m height (DBH) in meters,
#                        and the tree height (H) in meters
#
# Shrub Inventory Table (CSV)- Table used to identify the location of each shrub, it's convex hull area
# ---------------------------------------------------------------------------------------------------
# Segmented understory (shrub) fuels raster (TIF) - canopy height model of shrubs segmented from the 0-3m fuel height 
#                                point cloud. Segmented using the treetops algorithm.
#  
# ---------------------------------------------------------------------------------------------------
# Metrics file (CSV)- Table of directly measured and applied predictive modeled metrics 
# ---------------------------------------------------------------------------------------------------
# Contacts:  Scott Pokswinski, Aaron Maxwell (UWV), Emily Link (USFWS)
#
# Install all packages through CRAN except TreeLS 
# The required TreeLS branch removes error handling when no trees are found 
#
# Install TreeLS and leafR using remotes package through Github
#       Install Rtools and specify path (Newer versions of Rtools no longer require a path specification)
#       Install remotes package 
#       paste into console:
#            remotes::install_github('spokswinski/TreeLS')
#
# IntELiMon Version 1.1.1 updates scripts to remove all packages that require rgdal including the raster package
# 
####################################################################################################################

# User Inputs:  

####################################################################################################################
{
  # Process command-line arguments
  args <- commandArgs(trailingOnly=TRUE)
  
  # Get current running script name
  script_name <- basename(sub(".*=", "", commandArgs()[4]))
  
  if (length(args) != 2) {
    stop(paste0("usage: ", script_name, " <input_file or input_dir> <output_dir>"))
  }
  
  inputData <- args[1]
  outDir <- args[2]
  
  if (file_test('-f', inputData)) {
    inDir <- dirname(inputData)
    files <- c(basename(inputData))
  } else if (file_test('-d', inputData)) {
    inDir <- inputData
    files <- list.files(pattern="*.ptx", inputData)
  } else {
    stop("not a valid input file or directory")
  }
  setwd(inDir)
  
  # Set parameters for processing      
  clipRadius <- 15
  sphMax <- 40
  b1 <- 0.5
  b2 <- 1
  b3 <- 1.5
  b4 <- 2
  bMax <- 30
  spacing <- .5
  scHght <- 2
  h1 <- 0.5
  h2 <- 1
  h3 <- 1.5
  h4 <- 2
  binMax <- 40
  binRes <- 1
  agg_factor <- 10  
  x <- 0
  script_name<-'IntELiMon_1.1.1'
  characters<- 21
  
  ####################################################################################################################
  
  # preliminary steps:  
  
  ####################################################################################################################  
  # Load required libraries
  library(nabor)# required
  library(dplyr)# required
  library(data.table)# required
  library(terra)#required
  library(rlas)# required
  library(lidR)# required
  library(e1071)# required
  library(geometry)# required
  library(sf)# required
  library(Morpho)# required
  library(TreeLS)# required
  # Create output folders
  
  # dir.create(file.path(outDir, "las"), recursive=TRUE)  
  dir.create(file.path(outDir, "metrics"), recursive=TRUE)
  dir.create(file.path(outDir, "dtm"), recursive=TRUE)
  dir.create(file.path(outDir, "chm"), recursive=TRUE)
  dir.create(file.path(outDir, "Fuels"), recursive=TRUE)
  dir.create(file.path(outDir, "Shrubs"), recursive=TRUE)
  dir.create(file.path(outDir, "Inventory"), recursive=TRUE)
  
  # Create a LAS header
  lasdata = data.frame(X = c(0.001, 0.001, 0.001),
                       Y = c(0.001, 0.001, 0.001),
                       Z = c(0.001, 0.001, 0.001),
                       R = c(0.001, 0.001, 0.001),
                       G = c(0.001, 0.001, 0.001),
                       B = c(0.001, 0.001, 0.001),
                       I = c(0.001, 0.001, 0.001),
                       gpstime = c(0L, 0L, 0L),
                       Intensity = c(0L, 0L, 0L),
                       ReturnNumber = c(0L, 0L, 0L),
                       NumberOfReturns = c(0L, 0L, 0L),
                       ScanDirectionFlag = c(0L, 0L, 0L),
                       EdgeOfFlightline = c(0L, 0L, 0L),
                       Classification = c(0L, 0L, 0L),
                       ScanAngleRank = c(0L, 0L, 0L),
                       UserData = c(0L, 0L, 0L),
                       PointSourceID = c(0L, 0L, 0L))
  
  
  lhead = header_create(lasdata)
  
  # Create a temporary location for the ephemeral LAS file
  tempPath  = file.path(tempdir(), "temp.las")
  tempPath2 = file.path(tempdir(), "cyl.las")
}

#################################################################################################################### 

#================================Functions================================

#################################################################################################################### 
  
{
  # Convert PTX files to LAS files
  ptxToLas <- function(ptx, outfile, header, radius) {
    dat <- fread(ptx, skip = 10, header = FALSE)
    names(dat) <- c("X", "Y", "Z", "Intensity", "R", "G", "B")
    dat2 <- dat %>% filter(X != 0 & Y != 0 & Z != 0)
    min_value <- 0
    dat2$R <- as.integer(dat2$R * 257)
    dat2$G <- as.integer(dat2$G * 257)
    dat2$B <- as.integer(dat2$B * 257)
    max_value <- max(dat2$Intensity)
    dat2$Intensity <- round((dat2$Intensity - min_value) / (max_value - min_value) * 255)
    dat2$intensity <- pmax(0, pmin(255, dat2$Intensity))
    dat2$Intensity <- as.integer(dat2$Intensity)
    dat2$dist <- sqrt(dat2$X ^ 2 + dat2$Y ^ 2)
    dat3 <- dat2 %>% filter(dist <= radius)
    dat4 <- dat3[, 1:ncol(dat3) - 1]
    write.las(outfile, header, dat3)
  }
  
  # convert las data to a matrix
  las2xyz = function(las){
    if(class(las)[1] != "LAS")
      stop("las must be a LAS object")
    
    las = las@data[,c('X','Y','Z')] %>% as.matrix
    return(las)
  }
  
  # noise filter
  nnFilter2 = function(las, d = 0.05, n = 2){
    rnn = knn(las %>% las2xyz, k = n+1)$nn.dists[,-1]
    keep = rep(T, nrow(las@data))
    for(i in 1:ncol(rnn)){
      keep = keep & rnn[,i] < d
    }
    las = filter_poi(las, keep)
    return(las)
  }
  
  # Create CWD noise function to remove noise
  nnFilter3 = function(las, d = 0.05, n = 15){
    rnn = knn(las %>% las2xyz, k = n+1)$nn.dists[,-1]
    keep = rep(T, nrow(las@data))
    for(i in 1:ncol(rnn)){
      keep = keep & rnn[,i] < d
    }
    las = filter_poi(las, keep)
    return(las)
  }

  # toMetrics function to establish output metrics DF and output initial metrics
  toMetrics = function(l1, prefix){
    l1_cnt <- length(l1@data$Z)
    l1_mean <- mean(l1@data$Z) 
    l1_median <- median(l1@data$Z) 
    l1_std <- sd(l1@data$Z) 
    l1_tgi <- mean(((670-480)*(as.double(l1@data$R)-as.double(l1@data$G)))-
                     ((670-550)*(as.double(l1@data$R)-as.double(l1@data$B)))/2, na.rm=TRUE)/(2^15)
    l1_vari <- mean((as.double(l1@data$G)-as.double(l1@data$R))/
                      (as.double(l1@data$G)+as.double(l1@data$R)-as.double(l1@data$B)+0.0001))
    l1_skew <- skewness(l1@data$Z)
    l1_kurt <- kurtosis(l1@data$Z)
    
    # Merge metrics to dataframe
    all_metrics <- as.data.frame(cbind(l1_cnt, l1_mean, l1_median, l1_std, l1_tgi, l1_vari, l1_skew,l1_kurt))
    
    # Rename all columns
    cNames <- names(all_metrics)
    cNames2 <- paste0(prefix, cNames)
    names(all_metrics) <- cNames2
    return(all_metrics)
  }
  
  # Define function to calculate metrics for height bins
  heightBinMetrics = function(la, circleRad, b1, b2, b3, b4, bMax){
    #Input ground classified and height normalized data
    #Clip to plot radius extent
    laClip <- clip_circle(la, 0, 0, circleRad)
    #Filter out ground classified points
    lg <- filter_poi(laClip, Classification == 2) 
    #Filter out not ground classified points
    lng <- filter_poi(laClip, Classification != 2)
    if (object.size(lng)> 20000L){
      #Break data into five height bins (defined using b1 through b4 variables)
      l1 <- filter_poi(lng, Z > 0 & Z <= b1) 
      l2 <- filter_poi(lng, Z > b1 & Z <= b2) 
      l3 <- filter_poi(lng, Z > b2 & Z <= b3) 
      l4 <- filter_poi(lng, Z > b3 & Z <= b4) 
      l5 <- filter_poi(lng, Z > b4 & Z <= bMax)
      GC <- filter_poi(lng, Z > 0 & Z <= 1)
      US <- filter_poi(lng, Z > 1 & Z <= 3)
      MS <- filter_poi(lng, Z > 3 & Z <= 9)
      OS <- filter_poi(lng, Z > 9 & Z <= bMax)
      
      #Get plot scale metrics
      #Count of ground points
      ground_cnt <- length(lg@data$Z) 
      #Count of not ground points
      not_ground_cnt <- length(lng@data$Z) 
      #Percent of returns that are ground
      per_ground <- (ground_cnt/(not_ground_cnt+ground_cnt))*100 
      #Calculate triangular greenness index from RGB data
      ng_tgi <- mean(((670-480)*(as.double(lng@data$R)-as.double(lng@data$G)))-
                       ((670-550)*(as.double(lng@data$R)-as.double(lng@data$B)))/2, na.rm=TRUE)/(2^15)
      #Calculate Visual Atmospheric Resistance Index from RGB data
      ng_vari <- mean((as.double(lng@data$G)-as.double(lng@data$R))/
                        (as.double(lng@data$G)+as.double(lng@data$R)-as.double(lng@data$B)+.0001))
      
      #Get l1 metrics
      l1_cnt <- length(l1@data$Z)
      l1_per <- (l1_cnt/(not_ground_cnt))*100 
      l1_mean <- mean(l1@data$Z) 
      l1_median <- median(l1@data$Z) 
      l1_std <- sd(l1@data$Z) 
      l1_tgi <- mean(((670-480)*(as.double(l1@data$R)-as.double(l1@data$G)))-
                       ((670-550)*(as.double(l1@data$R)-as.double(l1@data$B)))/2, na.rm=TRUE)/(2^15)
      l1_vari <- mean((as.double(l1@data$G)-as.double(l1@data$R))/
                        (as.double(l1@data$G)+as.double(l1@data$R)-as.double(l1@data$B)+0.0001))
      l1_skew <- skewness(l1@data$Z)
      l1_kurt <- kurtosis(l1@data$Z)
      
      #Get l2 metrics
      l2_cnt <- length(l2@data$Z)
      l2_per <- (l2_cnt/(not_ground_cnt))*100 
      l2_mean <- mean(l2@data$Z) 
      l2_median <- median(l2@data$Z) 
      l2_std <- sd(l2@data$Z) 
      l2_tgi <- mean(((670-480)*(as.double(l2@data$R)-as.double(l2@data$G)))-
                       ((670-550)*(as.double(l2@data$R)-as.double(l2@data$B)))/2, na.rm=TRUE)/(2^15)
      l2_vari <- mean((as.double(l2@data$G)-as.double(l2@data$R))/
                        (as.double(l2@data$G)+as.double(l2@data$R)-as.double(l2@data$B)+0.0001))
      l2_skew <- skewness(l2@data$Z)
      l2_kurt <- kurtosis(l2@data$Z)
      
      #Get l3 metrics
      l3_cnt <- length(l3@data$Z) 
      l3_per <- (l3_cnt/(not_ground_cnt))*100
      l3_mean <- mean(l3@data$Z) 
      l3_median <- median(l3@data$Z) 
      l3_std <- sd(l3@data$Z) 
      l3_tgi <- mean(((670-480)*(as.double(l3@data$R)-as.double(l3@data$G)))-
                       ((670-550)*(as.double(l3@data$R)-as.double(l3@data$B)))/2, na.rm=TRUE)/(2^15)
      l3_vari <- mean((as.double(l3@data$G)-as.double(l3@data$R))/
                        (as.double(l3@data$G)+as.double(l3@data$R)-as.double(l3@data$B)+0.0001))
      l3_skew <- skewness(l3@data$Z)
      l3_kurt <- kurtosis(l3@data$Z)
      
      #Get L4 metrics
      l4_cnt <- length(l4@data$Z) 
      l4_per <- (l4_cnt/(not_ground_cnt))*100 
      l4_mean <- mean(l4@data$Z) 
      l4_median <- median(l4@data$Z) 
      l4_std <- sd(l4@data$Z)
      l4_tgi <- mean(((670-480)*(as.double(l4@data$R)-as.double(l4@data$G)))-
                       ((670-550)*(as.double(l4@data$R)-as.double(l4@data$B)))/2, na.rm=TRUE)/(2^15)
      l4_vari <- mean((as.double(l4@data$G)-as.double(l4@data$R))/
                        (as.double(l4@data$G)+as.double(l4@data$R)-as.double(l4@data$B)+0.0001))
      l4_skew <- skewness(l4@data$Z)
      l4_kurt <- kurtosis(l4@data$Z)
      
      #Get L5 metrics
      l5_cnt <- length(l5@data$Z) 
      l5_per <- (l5_cnt/(not_ground_cnt))*100 
      l5_mean <- mean(l5@data$Z) 
      l5_median <- median(l5@data$Z) 
      l5_std <- sd(l5@data$Z) 
      l5_tgi <- mean(((670-480)*(as.double(l5@data$R)-as.double(l5@data$G)))-
                       ((670-550)*(as.double(l5@data$R)-as.double(l5@data$B)))/2, na.rm=TRUE)/(2^15)
      l5_vari <- mean((as.double(l5@data$G)-as.double(l5@data$R))/
                        (as.double(l5@data$G)+as.double(l5@data$R)-as.double(l5@data$B)+0.0001))
      l5_skew <- skewness(l5@data$Z)
      l5_kurt <- kurtosis(l5@data$Z)
      
      #Get Ground Cover metrics
      GC_cnt <- length(GC@data$Z) 
      GC_per <- (GC_cnt/(not_ground_cnt))*100 
      GC_mean <- mean(GC@data$Z) 
      GC_median <- median(GC@data$Z) 
      GC_std <- sd(GC@data$Z) 
      GC_tgi <- mean(((670-480)*(as.double(GC@data$R)-as.double(GC@data$G)))-
                       ((670-550)*(as.double(GC@data$R)-as.double(GC@data$B)))/2, na.rm=TRUE)/(2^15)
      GC_vari <- mean((as.double(GC@data$G)-as.double(GC@data$R))/
                        (as.double(GC@data$G)+as.double(GC@data$R)-as.double(GC@data$B)+0.0001))
      GC_skew <- skewness(GC@data$Z)
      GC_kurt <- kurtosis(GC@data$Z)
      
      #Get Understory metrics
      US_cnt <- length(US@data$Z) 
      US_per <- (US_cnt/(not_ground_cnt))*100 
      US_mean <- mean(US@data$Z) 
      US_median <- median(US@data$Z) 
      US_std <- sd(US@data$Z) 
      US_tgi <- mean(((670-480)*(as.double(US@data$R)-as.double(US@data$G)))-
                       ((670-550)*(as.double(US@data$R)-as.double(US@data$B)))/2, na.rm=TRUE)/(2^15)
      US_vari <- mean((as.double(US@data$G)-as.double(US@data$R))/
                        (as.double(US@data$G)+as.double(US@data$R)-as.double(US@data$B)+0.0001))
      US_skew <- skewness(US@data$Z)
      US_kurt <- kurtosis(US@data$Z)
      
      #Get Midstory metrics
      MS_cnt <- length(MS@data$Z) 
      MS_per <- (MS_cnt/(not_ground_cnt))*100 
      MS_mean <- mean(MS@data$Z) 
      MS_median <- median(MS@data$Z) 
      MS_std <- sd(MS@data$Z) 
      MS_tgi <- mean(((670-480)*(as.double(MS@data$R)-as.double(MS@data$G)))-
                       ((670-550)*(as.double(MS@data$R)-as.double(MS@data$B)))/2, na.rm=TRUE)/(2^15)
      MS_vari <- mean((as.double(MS@data$G)-as.double(MS@data$R))/
                        (as.double(MS@data$G)+as.double(MS@data$R)-as.double(MS@data$B)+0.0001))
      MS_skew <- skewness(MS@data$Z)
      MS_kurt <- kurtosis(MS@data$Z)
      
      #Get Overstory metrics
      OS_cnt <- length(OS@data$Z) 
      OS_per <- (OS_cnt/(not_ground_cnt))*100 
      OS_mean <- mean(OS@data$Z) 
      OS_median <- median(OS@data$Z) 
      OS_std <- sd(OS@data$Z) 
      OS_tgi <- mean(((670-480)*(as.double(OS@data$R)-as.double(OS@data$G)))-
                       ((670-550)*(as.double(OS@data$R)-as.double(OS@data$B)))/2, na.rm=TRUE)/(2^15)
      OS_vari <- mean((as.double(OS@data$G)-as.double(OS@data$R))/
                        (as.double(OS@data$G)+as.double(OS@data$R)-as.double(OS@data$B)+0.0001))
      OS_skew <- skewness(OS@data$Z)
      OS_kurt <- kurtosis(OS@data$Z)
      
      #Cloud-level metrics from lidR package cloud_metrics function
      standard_metrics<-cloud_metrics(lng,.stdmetrics_z) %>% do.call('cbind',.) 
      
    } else {
      l1 <- filter_poi(laClip, Z > 0 & Z <= b1) 
      l2 <- filter_poi(laClip, Z > b1 & Z <= b2) 
      l3 <- filter_poi(laClip, Z > b2 & Z <= b3) 
      l4 <- filter_poi(laClip, Z > b3 & Z <= b4) 
      l5 <- filter_poi(laClip, Z > b4 & Z <= bMax)
      GC <- filter_poi(laClip, Z > 0 & Z <= 1)
      US <- filter_poi(laClip, Z > 1 & Z <= 3)
      MS <- filter_poi(laClip, Z > 3 & Z <= 9)
      OS <- filter_poi(laClip, Z > 9 & Z <= bMax)
      
      #Get plot scale metrics
      #Count of ground points
      ground_cnt <- length(lg@data$Z) 
      #Count of not ground points
      not_ground_cnt <- length(lng@data$Z) 
      #Percent of returns that are ground
      per_ground <- (ground_cnt/(not_ground_cnt+ground_cnt))*100 
      #Calculate triangular greenness index from RGB data
      ng_tgi <- 0L
      #Calculate Visual Atmospheric Resistance Index from RGB data
      ng_vari <- 0L
      
      #Get l1 metrics
      l1_cnt <- 0L
      l1_per <- 0L
      l1_mean <- 0L
      l1_median <- 0L
      l1_std <- 0L
      l1_tgi <- mean(((670-480)*(as.double(l1@data$R)-as.double(l1@data$G)))-
                       ((670-550)*(as.double(l1@data$R)-as.double(l1@data$B)))/2, na.rm=TRUE)/(2^15)
      l1_vari <- mean((as.double(l1@data$G)-as.double(l1@data$R))/
                        (as.double(l1@data$G)+as.double(l1@data$R)-as.double(l1@data$B)+0.0001))
      l1_skew <- 0L
      l1_kurt <- 0L
      
      #Get l2 metrics
      l2_cnt <- 0L
      l2_per <- 0L
      l2_mean <- 0L
      l2_median <- 0L
      l2_std <- 0L
      l2_tgi <- 0L
      l2_vari <- 0L
      l2_skew <- 0L
      l2_kurt <- 0L
      
      #Get l3 metrics
      l3_cnt <- 0L
      l3_per <- 0L
      l3_mean <- 0L
      l3_median <- 0L 
      l3_std <- 0L
      l3_tgi <- 0L
      l3_vari <- 0L
      l3_skew <- 0L
      l3_kurt <- 0L
      
      #Get L4 metrics
      l4_cnt <- 0L
      l4_per <- 0L
      l4_mean <- 0L
      l4_median <- 0L
      l4_std <- 0L
      l4_tgi <- 0L
      l4_vari <- 0L
      l4_skew <- 0L
      l4_kurt <- 0L
      
      #Get L5 metrics
      l5_cnt <- 0L
      l5_per <- 0L
      l5_mean <- 0L
      l5_median <- 0L
      l5_std <- 0L
      l5_tgi <- 0L
      l5_vari <- 0L
      l5_skew <- 0L
      l5_kurt <- 0L
      
      #Get Ground Cover metrics
      GC_cnt <- 0L
      GC_per <- 0L
      GC_mean <- 0L
      GC_median <- 0L
      GC_std <- 0L
      GC_tgi <- mean(((670-480)*(as.double(GC@data$R)-as.double(GC@data$G)))-
                       ((670-550)*(as.double(GC@data$R)-as.double(GC@data$B)))/2, na.rm=TRUE)/(2^15)
      GC_vari <- mean((as.double(GC@data$G)-as.double(GC@data$R))/
                        (as.double(GC@data$G)+as.double(GC@data$R)-as.double(GC@data$B)+0.0001))
      GC_skew <- 0L
      GC_kurt <- 0L
      
      #Get Understory metrics
      US_cnt <- 0L
      US_per <- 0L
      US_mean <- 0L
      US_median <- 0L 
      US_std <- 0L 
      US_tgi <- 0L
      US_vari <- 0L
      US_skew <- 0L
      US_kurt <- 0L
      
      #Get Midstory metrics
      MS_cnt <- 0L
      MS_per <- 0L
      MS_mean <- 0L
      MS_median <- 0L
      MS_std <- 0L 
      MS_tgi <- 0L
      MS_vari <- 0L
      MS_skew <- 0L
      MS_kurt <- 0L
      
      #Get Overstory metrics
      OS_cnt <- 0L
      OS_per <- 0L
      OS_mean <- 0L
      OS_median <- 0L
      OS_std <- 0L
      OS_tgi <- 0L
      OS_vari <- 0L
      OS_skew <- 0L
      OS_kurt <- 0L
      #Cloud-level metrics from lidR package cloud_metrics function
      standard_metrics<-cloud_metrics(lg,.stdmetrics_z) %>% do.call('cbind',.) 
    }            
    
    
    #Merge metrics to dataframe
    all_metrics <- base::as.data.frame(cbind(filename, script_name, ground_cnt, not_ground_cnt, per_ground, ng_tgi, ng_vari,
                                             l1_cnt, l1_per, l1_mean, l1_median, l1_std, l1_tgi, l1_vari, l1_skew,l1_kurt,
                                             l2_cnt, l2_per, l2_mean, l2_median, l2_std, l2_tgi, l2_vari, l2_skew,l2_kurt,
                                             l3_cnt, l3_per, l3_mean, l3_median, l3_std, l3_tgi, l3_vari, l3_skew,l3_kurt,
                                             l4_cnt, l4_per, l4_mean, l4_median, l4_std, l4_tgi, l4_vari, l4_skew,l4_kurt,
                                             l5_cnt, l5_per, l5_mean, l5_median, l5_std, l5_tgi, l5_vari, l5_skew,l5_kurt,
                                             GC_cnt, GC_per, GC_mean, GC_median, GC_std, GC_tgi, GC_vari, GC_skew,GC_kurt,
                                             US_cnt, US_per, US_mean, US_median, US_std, US_tgi, US_vari, US_skew,US_kurt,
                                             MS_cnt, MS_per, MS_mean, MS_median, MS_std, MS_tgi, MS_vari, MS_skew,MS_kurt,
                                             OS_cnt, OS_per, OS_mean, OS_median, OS_std, OS_tgi, OS_vari, OS_skew,OS_kurt,
                                             standard_metrics))
    
    
    #Remove entropy metric
    drop <- c("zentropy")
    all_metrics = all_metrics[,!(names(all_metrics) %in% drop)]
    
    #Rename all columns
    cNames <- names(all_metrics)
    cNames2 <- paste0("h_", cNames)
    names(all_metrics) <- cNames2
    
    return(all_metrics) 
  }
  
  # Get header info (Jeff Atkins script)
  ptx.header<-function(input_file){
    list(
      name = input_file,
      col_res = as.numeric(read.csv(input_file, sep = "", skip = 0,nrows = 2, 
                                    header = FALSE)[1,]),
      row_res = as.numeric(read.csv(input_file, sep = "", skip = 0,nrows = 2, 
                                    header = FALSE)[2,]),
      scan_center = as.matrix(read.csv(input_file, sep = "", skip = 2,nrows = 1, 
                                       header = FALSE)),
      reg = as.matrix(read.csv(input_file, sep = "", skip = 3,nrows = 3, 
                               header = FALSE)),
      trans_matrix = as.matrix(read.csv(input_file, sep = "", skip = 6,nrows = 4, 
                                        header = FALSE))
    )
  }

  # get proportion of gaps
  perGapFunc <- function(dat){
    data_gap <- dat %>% filter(x==0 & y==0 & z==0)
    per_gap <- (nrow(data_gap)/nrow(dat))*100
    return(data.frame(per_gap))
  }
  
  # Calculate distance from plot center
  getDist<-function(pts){
    pts$pnt_id <- row.names(dat)
    pts$count <- 1
    
    pts<- pts %>% mutate(returned = dplyr::if_else(x==0 & y==0 & z==0, 0, 1))
    pts$dist <- sqrt(pts$x^2 + pts$y^2 + pts$z^2)
    return(pts)    
  }
  
  # Create multiple fields to denote if a return reached each radii (0 = did not pass through, 1 = passed through)
  binRad <- function(pts, radii){
    for(ra in radii){
      pts<-pts %>% mutate("r_{ra}" := case_when(dist < ra ~0, dist >= ra ~ 1))
    }
    return(pts)
  }
  
  # Function to convert data to a raster grid using terra
  toRast <- function(pts, header_info, agg_factor, radii){
    #Convert points to a dataframe
    pts2 <- as.data.frame(pts)
    # Get number of columns from header info
    col_cnt <- header_info$col_res
    # Get number of rows from header info
    row_cnt <- header_info$row_res
    # Create matrix of just returns
    as_mat_returns <- matrix(data=pts2$returned, nrow=row_cnt, ncol=col_cnt, byrow=FALSE)
    # Create matrix of all pulses
    as_mat_pulses <- matrix(data=pts2$count, nrow=row_cnt, ncol=col_cnt, byrow=FALSE)
    # Generate blank raster for all returns
    all_returns <- rast(ncols = col_cnt, nrows = row_cnt, nlyrs=1, xmin = -180, xmax = 180, ymin = -60, ymax = 90)
    # Fill blank raster with values
    values(all_returns) <- as_mat_returns
    # Generate blank raster for all pulses
    all_pulses <- rast(ncols = col_cnt, nrows = row_cnt, nlyrs=1, xmin = -180, xmax = 180, ymin = -60, ymax = 90)
    # Fill blank raster with values
    values(all_pulses) <- as_mat_pulses
    # Aggregate by a factor to count number of returns in aggregating unit
    returns_agg <- aggregate(all_returns, fact=agg_factor, fun="sum")
    pulses_agg <- aggregate(all_pulses, fact=agg_factor, fun="sum")
    # Stack both raster grids into a single object using terra
    raster_stack <- c(pulses_agg, returns_agg)
    # Iterate through radii to generate return counts in aggregating unit
    for(ra in radii){
      # Generate empty raster
      rast1 <- rast(ncols = col_cnt, nrows = row_cnt, nlyrs=1, xmin = -180, xmax = 180, ymin = -60, ymax = 90)
      # Write values to matrix
      as_mat <- matrix(data=pts2[,paste0("r_", as.character(ra))], nrow=row_cnt, ncol=col_cnt, byrow=FALSE)
      # Fill raster with values
      values(rast1) <- as_mat
      # Add output to stack at end of each loop
      raster_stack <- c(raster_stack, aggregate(rast1, fact=agg_factor, fun="sum"))
    }
    # Rename bands
    names <- c("all_pulses", "all_returns")
    names2 <- paste0("r_", as.character(radii))
    names <- append(names, names2)
    names(raster_stack) <- names
    #Return raster stack
    
    # raster_stack2 <- rotate(raster_stack)
    # re-orient raster stack
    raster_stack3 <- flip(raster_stack, direction="vertical")
    
    return(raster_stack3)
  }
  
  # Function to calculate proportions of pulses in each voxel
  toProportions <- function(rasterIn, radii){
    # Extract all pulses count
    pulses <- rasterIn["all_pulses"]
    # Extract all returns count
    returns <- rasterIn["all_returns"]
    # List all processing radii
    radii2 <- paste0("r_", as.character(radii))
    # Find all canopy gaps or areas of pulses but no returns
    gaps <- returns <= 0
    for(ra in 3:42){ 
      # Process for most inner bin
      if(ra == 3){ 
        # Returns in voxel equals all returns - returns passing through
        gridIn <- returns - rasterIn[[ra]]
        # Code to NA if no pulses passing through voxel
        null1 <- app(pulses, function(x){x[x<=0]<-NA; return(x)})
        # Create mask to remove areas with no returns
        null2 <- null1 >= 0 
        # Proportion of pulses returning form voxel equals returns from voxel divided by total pulses 
        out <- gridIn/pulses 
        # Multiply by mask to code all areas with no pulses to NA
        out <- out*null2 
        # Add to output stack
        out2 <- c(gaps, out)
        # Process for second bin
      }else if(ra == 4){
        # Save results raster stack to new object
        first <- out2
        # Save result from first bin to new object
        first2 <- out
        # Code to NA if all pulses were already returned (areas of occlusion)
        first3 <- app(first2, function(x){x[x==1]<-NA; return(x)})
        # Make mask
        mask3 <- first3 >= 0
        # Subtract returns from prior bin
        gridPre <- pulses - gridIn
        # Save returns from prior bin to new object
        gridInPre <- gridIn
        # Count in bin equals number of returns minus returns that pass minus returns from prior bin
        gridIn <- returns - rasterIn[[ra]] - gridInPre
        # Running total of returns
        total_re <- gridIn + gridInPre
        # Proportion of pulses passing through voxel that returned from it
        out <- ((gridIn)/gridPre)
        # Multiply by mask for occlusion
        out <- out*mask3 
        # Add to raster stack
        out3 <- c(first, out)
      }else{
        # Save raster stack to new object
        preceed <- out3
        # Save result from prior bin
        prior <- out
        # Code to NA if all pulses were already returned (areas of occlusion)
        prior2 <- app(prior, function(x){x[x==1]<-NA; return(x)})
        # Make mask
        mask3 <- prior2 >= 0
        # Get all pulses passing through voxel as pulses minus those already returned
        gridPre <- pulses - total_re
        # Get pulses returned from voxel
        gridIn <- returns - rasterIn[[ra]] - total_re
        # Update cumulative number of returns
        total_re <- total_re + gridIn
        # Proportion of pulses passing through voxel that returned from it
        out <- ((gridIn)/gridPre)
        # Multiply by mask for occlusion
        out <- out*mask3 
        # Add to raster stack
        out3 <- c(preceed, out) 
      }
    }
    # Rename bands
    names <- c("gaps")
    names <- append(names, radii2)
    names(out3) <- names
    return(out3) 
  }
  
  # Summarize by height bins from spherical processing
  summarizeSpheres <- function(rast, dtm, radii, spacing, clipRadius, h1, h2, h3, h4, binMax){
    # Convert to terra raster
    rast2 <- rast
    # Code all NA to 11
    rast2[is.na(rast2)] <- 11
    # Convert raster to a dataframe
    # asPnts <- as.data.frame(rasterToPoints(rast2))
    asPnts <- as.data.frame(as.points(rast2))
    locxy<- crds(rast2, df=FALSE, na.rm=TRUE, na.all=TRUE)
    asPnts <- cbind(locxy, asPnts)
    # Extract and combine needed columns
    asPnts2 <- asPnts[,c(1:3)]
    # Set value to add to get voxel center
    asPnts2$r <- binRes/2
    # Loop through all radii to get separate record for each radii
    names(asPnts2) <- c("t", "p", "data", "r")
    for(i in 4:(length(radii)+3)){
      asPntsx <- asPnts[,c(1:2, i)]
      asPntsx$r <- (i-3) - binRes/2
      names(asPntsx) <- c("t", "p", "data", "r")
      asPnts2 <- bind_rows(asPnts2, asPntsx)
    }
    
    # Convert from spherical to Cartesian coordinates
    toCart <- as.data.frame(sph2cart(theta=asPnts2$t/57.2958, 
                                     phi=asPnts2$p/57.2958, 
                                     r=asPnts2$r))
    
    toCart$x <- toCart$x*-1
    toCart$y <- toCart$y*-1
    
    # Merge result to original points
    asPnts2 <- bind_cols(asPnts2, toCart)
    # At point ID as attribute
    asPnts2$id <- as.numeric(row.names(asPnts2))
    
    # Create empty, uniform raster in Cartesian space. 
    xseq <- seq(-10-clipRadius, 10+clipRadius, by=spacing) 
    yseq <- seq(-10-clipRadius, 10+clipRadius, by=spacing) 
    zseq <- seq(-10, binMax, by=spacing) 
    regPnts <- expand.grid(xseq, yseq, zseq)
    names(regPnts) <- c("centerX", "centerY", "centerZ")
    # Find spherical voxel that is closest to each point in Cartesian space. 
    withSphere <- mcNNindex(as.matrix(asPnts2[5:7]), as.matrix(regPnts), k=1)
    regPnts$id <- withSphere
    
    # Clip out only Cartesian points in cylinder extent. 
    regPnts <- mutate(regPnts, dist = sqrt(centerX^2+centerY^2))
    regPntsClip <- regPnts %>% filter(dist <= clipRadius)
    # Join data from closest spherical voxel to each Cartesian point
    regPnts3 <- dplyr::left_join(regPntsClip, asPnts2, by="id")
    
    # Write out to LAS file to temporary file
    regPnts4 <- regPnts3[,c("x", "y", "z", "data")]
    regPnts4$data <-as.integer(regPnts4$data*1000) 
    names(regPnts4) <- c("X", "Y", "Z", "Intensity")
    
    write.las(tempPath2, lhead, regPnts4)
    cyl_las <- readLAS(tempPath2)
    
    
    # Normalize Cartesian points to the dtm
    cyl_norm = normalize_height(cyl_las, dtm)
    # Remote below ground points
    cyl_norm_filter <- filter_poi(cyl_norm, Z > 0)
    the_nulls <- filter_poi(cyl_norm_filter, Intensity == 11000)
    the_not_nulls <- filter_poi(cyl_norm_filter, Intensity != 11000)
    
    
    # Convert to a dataframe
    lng <- data.frame(X = cyl_norm_filter@data$X, 
                      Y = cyl_norm_filter@data$Y, 
                      Z= cyl_norm_filter@data$Z, 
                      data=cyl_norm_filter@data$Intensity)
    
    
    # Separate data by height bin        
    l1 <- lng %>% filter(Z > 0 & Z <= h1) 
    l2 <- lng %>% filter(Z > h1 & Z <= h2) 
    l3 <- lng %>% filter(Z > h2 & Z <= h3) 
    l4 <- lng %>% filter(Z > h3 & Z <= h4) 
    l5 <- lng %>% filter(Z > h4 & Z <= binMax) 
    
    # Calculate metrics by height bin 
    # % Occluded, %With 0, %With returns, mean with proportion for areas with returns
    lng_na_per <- (nrow(lng %>% filter(data == 11000))/nrow(lng))*100
    lng_zero_per <- (nrow(lng %>% filter(data == 0))/nrow(lng))*100
    lng_with_prop <- lng %>% filter(data != 0 | data != 11000)
    lng_prop_mn <- mean(lng_with_prop$data/100) 
    lng_prop_sd <- sd(lng_with_prop$data/100) 
    lng_prop_sk <- skewness(lng_with_prop$data/100) 
    lng_prop_ku <- kurtosis(lng_with_prop$data/100) 
    
    l1_na_per <- (nrow(l1 %>% filter(data == 11000))/nrow(l1))*100
    l1_zero_per <- (nrow(l1 %>% filter(data == 0))/nrow(l1))*100
    l1_with_prop <- l1 %>% filter(data != 0 | data != 11000)
    l1_prop_mn <- mean(l1_with_prop$data/100) 
    l1_prop_sd <- sd(l1_with_prop$data/100) 
    l1_prop_sk <- skewness(l1_with_prop$data/100) 
    l1_prop_ku <- kurtosis(l1_with_prop$data/100) 
    
    l2_na_per <- (nrow(l2 %>% filter(data == 11000))/nrow(l2))*100
    l2_zero_per <- (nrow(l2 %>% filter(data == 0))/nrow(l2))*100
    l2_with_prop <- l2 %>% filter(data != 0 | data != 11000)
    l2_prop_mn <- mean(l2_with_prop$data/100) 
    l2_prop_sd <- sd(l2_with_prop$data/100) 
    l2_prop_sk <- skewness(l2_with_prop$data/100) 
    l2_prop_ku <- kurtosis(l2_with_prop$data/100)
    
    l3_na_per <- (nrow(l3 %>% filter(data == 11000))/nrow(l3))*100
    l3_zero_per <- (nrow(l3 %>% filter(data == 0))/nrow(l3))*100
    l3_with_prop <- l3 %>% filter(data != 0 | data != 11000)
    l3_prop_mn <- mean(l3_with_prop$data/100) 
    l3_prop_sd <- sd(l3_with_prop$data/100) 
    l3_prop_sk <- skewness(l3_with_prop$data/100) 
    l3_prop_ku <- kurtosis(l3_with_prop$data/100)
    
    l4_na_per <- (nrow(l4 %>% filter(data == 11000))/nrow(l4))*100
    l4_zero_per <- (nrow(l4 %>% filter(data == 0))/nrow(l4))*100
    l4_with_prop <- l4 %>% filter(data != 0 | data != 11000)
    l4_prop_mn <- mean(l4_with_prop$data/100) 
    l4_prop_sd <- sd(l4_with_prop$data/100) 
    l4_prop_sk <- skewness(l4_with_prop$data/100) 
    l4_prop_ku <- kurtosis(l4_with_prop$data/100)
    
    l5_na_per <- (nrow(l5 %>% filter(data == 11000))/nrow(l5))*100
    l5_zero_per <- (nrow(l5 %>% filter(data == 0))/nrow(l5))*100
    l5_with_prop <- l5 %>% filter(data != 0 | data != 11000)
    l5_prop_mn <- mean(l5_with_prop$data/100) 
    l5_prop_sd <- sd(l5_with_prop$data/100) 
    l5_prop_sk <- skewness(l5_with_prop$data/100) 
    l5_prop_ku <- kurtosis(l5_with_prop$data/100)
    
    
    #Merge to a dataframe
    sphere_metrics <- data.frame(cbind(lng_na_per,lng_zero_per, lng_prop_mn,lng_prop_sd,lng_prop_sk,lng_prop_ku,
                                       l1_na_per,l1_zero_per,l1_prop_mn,l1_prop_sd,l1_prop_sk,l1_prop_ku,
                                       l2_na_per,l2_zero_per,l2_prop_mn,l2_prop_sd,l2_prop_sk,l2_prop_ku,
                                       l3_na_per,l3_zero_per,l3_prop_mn,l3_prop_sd,l3_prop_sk,l3_prop_ku,
                                       l4_na_per,l4_zero_per,l4_prop_mn,l4_prop_sd,l4_prop_sk,l4_prop_ku,
                                       l5_na_per,l5_zero_per,l5_prop_mn,l5_prop_sd,l5_prop_sk,l5_prop_ku))
    cNames <- names(sphere_metrics)
    cNames2 <- paste0("s_", cNames)
    names(sphere_metrics) <- cNames2
    
    return(sphere_metrics)
  }
  
  # leafR functions updated for terra compatibility
    # Points by Z slice creates a dataframe of height point densities by height bin
  pointsByZSlice = function(Z, maxZ){
    heightSlices = as.integer(Z) # Round down
    zSlice = data.table::data.table(Z=Z, heightSlices=heightSlices) # Create a data.table (Z, slices))
    sliceCount = stats::aggregate(list(V1=Z), list(heightSlices=heightSlices), length) # Count number of returns by slice
    
    colRange = 0:maxZ
    addToList = setdiff(colRange, sliceCount$heightSlices)
    n = length(addToList)
    if (n > 0) {
      bindDt = data.frame(heightSlices = addToList, V1=integer(n))
      sliceCount = rbind(sliceCount, bindDt)
      # Order by height
      sliceCount = sliceCount[order(sliceCount$heightSlices),]
    }
    
    colNames = as.character(sliceCount$heightSlices)
    colNames[1] = "ground_0_1m"
    colNames[-1] = paste0("pulses_", colNames[-1], "_", sliceCount$heightSlices[-1]+1, "m")
    metrics = list()
    metrics[colNames] = sliceCount$V1
    
    return(metrics)
  }

  # LAI/LAD prep altered from leafR, voxelizes zslice data
  ladVoxels = function(las, grain.size = 1, k = 1){
    #empty list object that will be fueling with binneds data.frames
    LAD_VOXELS = list()
    Z = NA
    
    # convert normalized las cloud
    las@data$Z[las@data$Z < 0] = 0
    
    maxZ = floor(max(las@data$Z))
    
    func = formula(paste0("~pointsByZSlice(Z, ", maxZ, ")"))

    t.binneds     =lidR::pixel_metrics(las, func, res = grain.size,
                                       start = c(min(las@data$X), max(las@data$Y)))
    t.binneds    = data.frame(terra::crds(t.binneds, na.rm = FALSE, na.all = FALSE), terra::values(t.binneds))
    names(t.binneds)[1:2] = c("X", "Y")
    
    #select ground returns
    ground.returns = t.binneds[, grep("ground", names(t.binneds))]
    
    # select columns vegetation above 1m:
    if(nrow(t.binneds) != 1){ #this if is necessary when grain size is the whole plot
      pulses.profile.dz1 = t.binneds[, c(grep("pulses", names(t.binneds)))]
    }else{
      pulses.profile.dz1 = data.frame(matrix(as.numeric(as.character(t.binneds[, c(grep("pulses", names(t.binneds)))])), ncol = length(grep("pulses", names(t.binneds)))))
      names(pulses.profile.dz1) = names(t.binneds)[c(grep("pulses", names(t.binneds)))]
    }
    
    # invert data.frames for the sky be first
    pulses.profile.dz1 = pulses.profile.dz1[,length(pulses.profile.dz1):1] #invert columns
    
    #add grounds returns (0-1m)
    pulses.profile.dz1 = cbind(pulses.profile.dz1, ground.returns)
    rm(ground.returns)
    
    # total matriz and cumsum.matrix:
    total.pulses.matrix.dz1 = matrix(apply(pulses.profile.dz1, 1, sum), ncol = length(pulses.profile.dz1), nrow = nrow(pulses.profile.dz1))
    cumsum.matrix.dz1 = matrix(apply(pulses.profile.dz1, 1, cumsum), ncol = length(pulses.profile.dz1), nrow = nrow(pulses.profile.dz1), byrow = TRUE)
    
    rm(pulses.profile.dz1)
    
    #Pulses out for each voxel
    pulse.out.dz1 = total.pulses.matrix.dz1 - cumsum.matrix.dz1
    
    # The pulses.out of voxel 1 is the pulses.in of voxel 2 and so on...
    # Therefore, pulse.in is pulse.out without the last line and adding in the
    # first line the total pulses:
    if(nrow(t.binneds) != 1){ #if used when grain size of the whole plot
      pulse.in.dz1 <- cbind(total.pulses.matrix.dz1[,1], pulse.out.dz1[,-c(ncol(pulse.out.dz1))])
    }else{
      pulse.in.dz1 <- c(total.pulses.matrix.dz1[,1], pulse.out.dz1[,-c(ncol(pulse.out.dz1))])
    } #enf if
    
    rm(total.pulses.matrix.dz1, cumsum.matrix.dz1)
    
    # MacArthur-Horn eqquation
    # LAD = ln(S_bottom/S_top)*(1/(dz*K))
    #k value for LAD equation
    dz = 1
    
    LAD.dz1 = log(pulse.in.dz1/pulse.out.dz1) * 1/k * 1/dz
    
    rm(pulse.in.dz1, pulse.out.dz1)
    
    # Remove infinite and NaN values
    LAD.dz1[is.infinite(LAD.dz1)] <- NA; LAD.dz1[is.nan(LAD.dz1)] <- NA;
    
    # Remove the first 1 meter close to the ground (and the ground too)
    LAD.dz1 = LAD.dz1[, -c(ncol(LAD.dz1))]
    
    # Fuel list object
    LAD_VOXELS[["LAD"]] = LAD.dz1
    LAD_VOXELS[["coordenates"]] = t.binneds[,c("X", "Y")]
    
    rm(LAD.dz1, t.binneds)
    
    return(LAD_VOXELS)
  }
  
  #l adprofile more prep for leafR calculations
  
  lad.profile = function(VOXELS_LAD){
    
    t.lad.profile = apply(VOXELS_LAD$LAD, 2, mean, na.rm = TRUE)
    
    max_height = ncol(VOXELS_LAD[[1]]) + .5
    
    t.lad.profile = data.frame(height = seq(1.5, max_height), lad = t.lad.profile[length(t.lad.profile):1])
    
    return(t.lad.profile)
    
  }
  
  
  # leaf area index function
  lai = function(lad_profile, min = 1, max = 65){
    lai = sum(lad_profile$lad[(min):(max)], na.rm = TRUE)
    return(lai)
  }
  
  # Foliage height diversity function
  FHiD = function(lad_profile, evenness = FALSE, LAD.threshold = -1){
    
    # Applying threshold
    if(LAD.threshold == -1) LAD.threshold <- 1 / length(lad_profile$height)
    lad_profile <- lad_profile[lad_profile$lad >= LAD.threshold,]
    
    lad_profile$lad <- lad_profile$lad / 100
    
    # Calculating FHD
    if(evenness){
      FHD = - sum( lad_profile$lad * log(lad_profile$lad) ) / log( length(lad_profile$height) )
    }else{
      FHD = - sum( lad_profile$lad * log(lad_profile$lad) )
    } 
    
    return(FHD)
  }
  
  #Gini-Simpson foliage function
  GS = function(lad_profile, evenness = FALSE, LAD.threshold = -1){
    
    # applying threshold
    if(LAD.threshold == -1) LAD.threshold <- 1 / length(lad_profile$height)
    lad_profile <- lad_profile[lad_profile$lad >= LAD.threshold,]
    
    lad_profile$lad <- lad_profile$lad / 100
    
    # Calculating FHD
    if(evenness){
      GS = ( 1 - sum( lad_profile$lad^2 ) ) / ( 1 - ( 1 / length(lad_profile$height) ) )
    }else{
      GS = 1 - sum( lad_profile$lad^2 )
    } 
    
    return(GS)
    
  } 
  
  # Leaf area height volume calculation function
  LAHiV = function(lad_profile, LAI.weighting = FALSE, height.weighting = FALSE){
    
    # LAI.weighting
    if(LAI.weighting){
      LAHV = sum(lad_profile$height*lad_profile$lad)/sum(lad_profile$lad)
    }else{
      LAHV = sum(lad_profile$height*lad_profile$lad)
    } #end if else
    
     #height.weighting
    if(height.weighting){
      LAHV = LAHV/max(lad_profile$height)
    } #enf if
    
    return(LAHV)
    
  }
}  
####################################################################################################################### 
  
#=================Use Functions======================================================================
  
####################################################################################################################### 
  
  # Start the for loop
  for (file in files)
  {  
    # Define input PTX file
    inputPtx <- file
    
    # Remove the suffix to rename the files
    filename <-  substring(inputPtx, 0, characters)
    
    # Convert PTX to LAS and clip to radius
    ptxToLas(inputPtx, tempPath, lhead, clipRadius)
    
    # Classify and normalize point cloud
    
    # Prep TLS Data for segmentation and metric calculation
    las <- readLAS(tempPath)
    mycsf <- csf(TRUE, 0.5, .25, time_step = .65)
    lasCls <- classify_ground(las, mycsf)
    dtm = rasterize_terrain(lasCls, res = .25, pkg = "terra")
    writeRaster(dtm, filename = file.path(outDir, "dtm", paste0(filename, "_dtm.tif")), overwrite=TRUE)
    #writeLAS(las, file.path(outDir, "las", paste0(filename, ".las")))
    
    lasCls <- nnFilter2(lasCls)
    las_norm = tlsNormalize2(lasCls,dtm, min_res = 0.5, keep_ground = T)
    las_norm <- filter_poi(las_norm, Z >= 0L)
    thin = tlsSample(las_norm, smp.voxelize(0.02))
    rm(las)
    gc()
    
    # Calculate metrics by height bin
    metricsOut <- heightBinMetrics(las_norm, circleRad=clipRadius, b1, b2, b3, b4, bMax)
    
    # Use spherical voxels to characterize areas of gaps and occlusions. 
    dat <- fread(file.path(inDir, inputPtx), skip = 10, header = FALSE)
    colnames(dat)[1:4] <- c("x","y","z", "Intensity")
    per_gap <- perGapFunc(dat)
    header_info <- ptx.header(file.path(inDir, inputPtx))
    metricsOut <- bind_cols(metricsOut, per_gap)
    dat2 <- getDist(dat)
    rm(dat)
    gc()
    
    radii <- seq(1, binMax, by=binRes)
    dat3 <- binRad(dat2, radii)
    rm(dat2)
    gc()
    
    rast_stack <- toRast(dat3, header_info, agg_factor=10, radii)
    rm(dat3)
    gc()
    
    prop_out <- toProportions(rast_stack, radii)
    rm(rast_stack)
    gc()
    
    sph_summary <- summarizeSpheres(prop_out, dtm, radii, spacing, clipRadius, h1, h2, h3, h4, binMax)
    metricsOut <- bind_cols(metricsOut, sph_summary)
    
    # Create metrics of voxelized point cloud
    voxmetrics = toMetrics(thin, "vox_")
    metricsOut <- bind_cols(metricsOut, voxmetrics)
    
    # Create Canopy Height Model
    chm <- rasterize_canopy(thin, res = .5, p2r(0.7))  
    writeRaster(chm, filename = file.path(outDir, "chm", paste0(filename, "_chm.tif")), overwrite=TRUE)

    # Tree Segmentation  
    all_metrics = fastPointMetrics.available()
    my_metrics = all_metrics[c(16, 11)]
    Stems = fastPointMetrics(thin, ptm.knn(25), my_metrics)
    
    tlsfilter <- filter_poi(Stems, Verticality > 80, Verticality < 95)
    tlsfilter <- filter_poi(tlsfilter, Eigentropy < .03)
    map = treeMap(tlsfilter, map.hough(min_h = 2, max_h= 4, min_votes = 1), merge = 0)
    rm(Stems)
    gc()
    
    # Classify stem points only if trees were found in "map". Otherwise, create a blank tree inventory and add a "Stem" attribute for 
    if (object.size(map) > 20000L)
    {
      las_norm = treePoints(las_norm, map, trp.crop())
      
      las_norm = stemPoints(las_norm, stm.hough(
        h_step = 0.2,
        h_base = c(0.05, 2.05),
        min_votes = 1
      ))
      
      # get dbh and height
      inv = tlsInventory(las_norm, d_method = shapeFit(shape = 'circle', algorithm = 'ransac'))
      
      inv <- inv %>% mutate(DBH = (inv$Radius * 39.37) * 2)
      inv <- inv %>% mutate(BasalA = (DBH * DBH) * 0.005454)
    } else {
      inv  = data.frame(TreeID = c(0L),
                        X = c(0L),
                        Y = c(0L),
                        Radius = c(0L),
                        Error = c(0L),
                        H = c(0L),
                        h_radius = c(0L),
                        BasalA = c(0L),
                        DBH = c(0L))
      
      
      
      las_norm <- add_lasattribute(las_norm, x, "Stem", "tree stem point")
    }
    rm(map)
    gc()
    
    if (max(thin$Z) > 2L)
    {
      ladVox <- ladVoxels(thin)
      ladPro = lad.profile(ladVox)
      LAI <- lai(ladPro)
      ULAI <- lai(ladPro, min = 1, max = 3)
      MLAI <- lai(ladPro, min = 3, max = 9)
      OLAI <- lai(ladPro, min = 9, max = binMax)
      FHD <- FHiD(ladPro)
      GiSimp <- GS(ladPro)
      LAHV <-  LAHiV(ladPro, LAI.weighting = TRUE, height.weighting = TRUE)
    } else { 
      LAI = c(0L)
      ULAI = c(0L)
      MLAI = c(0L)
      OLAI = c(0L)
      FHD = c(0L)
      GiSimp = c(0L)
      LAHV = c(0L)
    }
    
    
    ground<- filter_poi(lasCls, Classification== 2L)
    dtmn= tlsSample(ground, smp.voxelize(1))
    dtm2 = rasterize_density(dtmn, res = 1)
    unoccl <-values(dtm2)
    NAs<- sum(unoccl==0)
    nonocarea<-900-NAs
    rm(ground, dtmn, dtm2, thin, lasCls)
    gc()
    
    # fuel ecology tree metrics
    TBA = sum(inv$BasalA)
    Basalarea = TBA/(0.000247105*nonocarea)
    MeanTH <- mean(inv$H)
    MDBH <- mean(inv$DBH)
    TreesN <- length(inv$TreeID)
    MaxTH <- max(inv$H)
    SDHT <- sd(inv$H)

    
    # extract stem points and create a point cloud of non-stem points
    las_norm@data[Stem == T, Classification := 20]
    justfuels<- filter_poi(las_norm, Classification < 18)
    # treefuels<- filter_poi(justfuels, Z > 1L) 
    thin2 <- tlsSample(justfuels, smp.voxelize(0.1))
    GCvol<- length(filter_poi(thin2, Z > 0 & Z <= 1)$Z)/1000
    USvol<- length(US <- filter_poi(thin2, Z > 1 & Z <= 3)$Z)/1000
    MSvol<- length(filter_poi(thin2, Z > 3 & Z <= 9)$Z)/1000
    OSvol<- length(filter_poi(thin2, Z > 9 & Z <= bMax)$Z)/1000
    treefuels<- filter_poi(thin2, Z > 1L)
    CBH<- quantile(treefuels$Z, 0.25) 
    rm(las_norm, thin2)
    gc()
    
    fem <- data.frame(TBA, Basalarea, MeanTH, MDBH, TreesN, MaxTH, SDHT, CBH, LAI, ULAI, MLAI, OLAI, FHD, GiSimp, LAHV, GCvol, USvol, MSvol, OSvol)
    metricsOut <- bind_cols(metricsOut, fem)
    
    write.csv(inv, file.path(outDir, "Inventory", paste0(filename, "_inv.csv")))
  

    # Height Segmentation  
    veght0to3m <- filter_poi(justfuels, Z > 0L, Z < 3L)
    fuelmetrics = toMetrics(veght0to3m, "fuel0_3")
    metricsOut <- bind_cols(metricsOut, fuelmetrics)
    rm(justfuels)
    gc()
    # a3pearce: outputs just before shrub classification the 0-3m cloud las
    # writeLAS(veght0to3m, file.path(outDir, "Shrubs", paste0(filename, "_01_input_0to3m.las")))
      
    # Call the nearest neighbor metrics remove missed tree points
    my_metrics2 = all_metrics[c(6, 7, 17, 16, 22)]
    veght0to3m = fastPointMetrics(veght0to3m, ptm.knn(25), my_metrics2)
    veght0to3m <- filter_poi(veght0to3m, Verticality < 80)
    veght0to3m <- filter_poi(veght0to3m, ZRange < 0.3)
    # Added by a3pearce: after initial shrub filters las
    # writeLAS(veght0to3m, file.path(outDir, "Shrubs", paste0(filename, "_02_after_geom_filter.las")))
      
    # Run algorithm to classify shrubs
    vegrast<- grid_canopy(veght0to3m, res = .5, p2r(0.2)) 
    col2 <- pastel.colors(200)
    ker <- matrix(1,3,3)
    vegrast <- focal(vegrast, w = ker, fun = mean, na.rm = TRUE)
    ## ADDED BY a3pearce - LMF, lowered hmin from 1.3 to 0.5 to capture shrubs closer to the ground
    # ttops <- find_trees(vegrast, lmf(3, hmin = 1.3, shape = c("circular")))
    ttops <- find_trees(vegrast, lmf(1.5, hmin = 0.25, shape = c("circular")))
    
    # run conditional handling function(try()) to ignore errors if there are no shrubs detected with the algorithm
    veght2 <- try(segment_trees(veght0to3m, silva2016(vegrast, ttops)))

    ## ADDED BY a3pearce - min point filter
    ## removes tree branches (high max_z, large z_range) and floating segments (high min_z)
    min_shrub_points = 100
    valid_shrubs <- veght2@data %>%
      group_by(treeID) %>%
      summarize(
        min_z = min(Z),
        max_z = max(Z),
        z_range = max(Z) - min(Z),
        npts    = dplyr::n()
      ) %>%
      filter(
        npts >= min_shrub_points,
        min_z < 0.5,
        max_z < 2.5,
        z_range < 2.0 # not a shrub
      ) %>%
      pull(treeID)
    
    veght2@data[treeID %in% valid_shrubs & treeID < 200, Classification := 21]
    shrubs <- filter_poi(veght2, Z > 0L, Classification == 21L)
    shrubmetrics = toMetrics(shrubs, "shrubs_")
    metricsOut <- bind_cols(metricsOut, shrubmetrics)
    rm(veght0to3m)
    gc()

    # Added by a3pearce: check out tree segmentation, get shrubs, get non-shrubs las
    # writeLAS(veght2, file.path(outDir, "Shrubs", paste0(filename, "_03_segmented.las")))
    # writeLAS(shrubs, file.path(outDir, "Shrubs", paste0(filename, "_04_shrubs.las")))
    
    if
    (object.size(shrubs) > 20000L)
    {
      shrublist <- crown_metrics(shrubs, func = .stdtreemetrics, geom = "convex")
      stlocdf <- st_centroid(shrublist)
      STLOC = data.frame(st_coordinates(stlocdf[,1]))
      shrubim <- st_drop_geometry(stlocdf)
      shrubinv <- cbind(shrubim,STLOC)
      shrubArea = sum(shrublist$convhull_area)
      scaledShrubArea = shrubArea/(0.000247105*nonocarea)
      MeanSH <- mean(shrublist$Z)
      MeanSA <- mean(shrublist$convhull_area)
      ShrubsN <- length(shrublist$treeID)
      MaxSH <- max(shrublist$Z)
      SDSHT <- sd(shrublist$Z)
      shdist <- st_distance(shrublist)
      MeanSD <- mean(shdist)
      MaxSD <- max(shdist)
      MinSD <- min(shdist)
      SDSD <- sd(shdist)
      shm <- grid_canopy(shrubs, res = 1, p2r(0.7)) 
      terra::writeRaster(shm, filename = paste0(outDir, "/Fuels/", substr(inputPtx, 1, nchar(inputPtx)-4), ".tif"), overwrite=TRUE)
    } else {
      ShrubsN = c(0L)
      shrubArea = c(0L)
      scaledShrubArea = c(0L) 
      MeanSH = c(0L) 
      MeanSA = c(0L) 
      MaxSH = c(0L)
      SDSHT = c(0L)
      MeanSD = c(0L)
      MaxSD = c(0L)
      MinSD = c(0L)
      SDSD = c(0L)
      shrubinv  = data.frame(treeID = c(0L),
                             Z = c(0L),
                             npoints = c(0L),
                             convhull_area = c(0L),
                             X = c(0L),
                             Y = c(0L))
      
    }
    
    write.csv(shrubinv, paste0(outDir, "/Shrubs/", substr(inputPtx, 1, nchar(inputPtx)-4), ".csv"))
    SFM <- data.frame(ShrubsN, shrubArea, scaledShrubArea, MeanSH, MeanSA, MaxSH, SDSHT, MeanSD, MaxSD, MinSD, SDSD)
    metricsOut <- bind_cols(metricsOut, SFM) 
    
    # Fine Fuels Layer
    fuelht<- filter_poi(veght2, Z > 0L, Classification < 21L, Z < 3L)
    finemetrics = toMetrics(fuelht, "fine_")
    metricsOut <- bind_cols(metricsOut, finemetrics)
    rm(veght2)
    gc()
    # Added by a3pearce: write out mid fuels not classified as ground to las
    # writeLAS(fuelht, file.path(outDir, "Shrubs", paste0(filename, "_05_finefuels.las")))
    
    # Filter 1-10hr fuels
    lin2<- filter_poi(fuelht, Linearity > 0.8)
    hr0_10_metrics = toMetrics(lin2, "hr0_10_")
    metricsOut <- bind_cols(metricsOut, hr0_10_metrics)
    
    # Filter 100-1000hr fuels
    CWD<- filter_poi(fuelht, Linearity < 0.5)
    CWD<- filter_poi(CWD, Planarity > 0.47)
    CWD <- filter_poi(CWD, EigenRatio2d < 0.4) 
    CWD <- filter_poi(CWD, Verticality < 68)
    CWD <- nnFilter3(CWD)
    hr100_1000_metrics = toMetrics(CWD, "hr100_1000_")
    metricsOut <- bind_cols(metricsOut, hr100_1000_metrics)
    
    # Replace NAs with 0 and write out final table
    metricsOut <- metricsOut %>% replace(is.na(.), 0)
    write.csv(metricsOut, file.path(outDir, "metrics", paste0(filename, "_metrics.csv")))
  }

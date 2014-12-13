"""
Config file that defines some
of the configuration parameters
"""

# The number of topics (defines also the number of features for k-means clustering
TOPIC_NBR= 80

BTM_CALL = "/home/rkempter/btm-v0.3/batch/btm"

# The number of location clusters
LOCATION_CLUSTER_NBR = 21


# Geographical clustering
#########################

# Number of cells in x
SIZE_X = 200

# Number of cells in y
SIZE_Y = 200

# DBSCAN EPS Parameter (diameter around a point)
EPS = 12

# DBSCAN MIN PTS Parameter
MIN_PTS = 40


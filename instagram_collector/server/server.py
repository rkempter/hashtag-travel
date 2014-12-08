from operator import itemgetter
def get_geo_json():
    """
    Generate a geo json (has to be moved to the flask application)
    """
    geo_json = {
        "type": "FeatureCollection",
        "features": []
    }
    df_result = pd.read_sql(media_query, conn)
    grouped_cluster = df_result.groupby([
        'cluster_id', 'cluster_name', 'center', 'radius'
    ])

    for name, group in grouped_cluster:
        cluster_id, cluster_name, center, radius = name
        center = loads(center)
        group_values = group[['id', 'image_url', 'lat', 'lng']].values

        geo_json['features'].append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    # lon, lat
                    "coordinates": [center.x, center.y],
                },
                "properties": {
                    "id": cluster_id,
                    "name": cluster_name,
                    "radius": radius,
                #    "media": [{"id": media[0],
                #               "image_url": media[1],
                #               "lat": media[2],
                #               "lng": media[3]} for media in group_values],
                }
            })

    return json.dumps(geo_json)
"""
Author: Renato Kempter, 2014
Collector for Instagram images containing geo location information.
"""
from contextlib import closing
import multiprocessing
import logging

from collections import namedtuple
from instagram import subscriptions, client
from flask import Flask, request, g, redirect, Response
from instagram_collector.config import (CLIENT_SECRET, REDIRECT_URI, CLIENT_ID,
                                        DATABASE, DB_HOST, DB_PASSWORD, DB_USER,
                                        THRESHOLD, RETURN_URI, ACCESS_TOKEN)

import MySQLdb

logging.basicConfig(filename='example.log',level=logging.DEBUG)

# The web application
app = Flask(__name__)


# The geographical locations that we want to subscribe to
LOCATIONS = [
    { 'lat': 48.916889692, 'long': 2.344497138, 'radius': 5000 },
    { 'lat': 48.883487422, 'long': 2.232917243, 'radius': 5000 },
    { 'lat': 48.857065773, 'long': 2.355140143, 'radius': 5000 },
    { 'lat': 48.816615905, 'long': 2.274115973, 'radius': 5000 },
    { 'lat': 48.822493333, 'long': 2.162536078, 'radius': 5000 },
]

def start():
    """
    Define callback for geographical post messages. Connect to the instagram api and
    get an access token.
    """
    unauthenticated_api = client.InstagramAPI(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI
    )
    return redirect(unauthenticated_api.get_authorize_url(scope=['basic']))

def subscribe_location():
    """
    Subscribes to all locations.
    """
    api = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    for location in LOCATIONS:
        api.create_subscription(
            object='geography',
            lat=location['lat'], lng=location['long'], radius=location['radius'],
            aspect='media',
            callback_url='http://grpupc1.epfl.ch/instagram/realtime_callback')

def unsubscribe_locations():
    """
    Unsubscribe from all locations
    """
    api = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    subscriptions = api.list_subscriptions()
    for subscription in map(lambda x: x['id'], subscriptions['data']):
        api.delete_subscriptions(id=subscription)

def connect_db():
    """
    Connect to the local mysql database
    """
    conn = MySQLdb.connect(user=DB_USER, passwd=DB_PASSWORD, db=DATABASE)
    return conn

def init_db():
    """
    Initialize the database based on the schema proposed
    """
    with closing(connect_db()) as db:
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().execute(f.read())
        db.commit()

def process_geo_location(update):
    """
    Process a list of updates and add them to subscriptions.
    """
    logging.getLogger(__name__).info("Processing an instagram update")
    insert_query = """INSERT IGNORE INTO media_events (`id`,
        `user_name`, `user_id`, `tags`, `location_name`,
        `location_lat`, `location_lng`, `filter`,
        `created_time`, `image_url`, `media_url`,
        `text`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    api = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    geo_id = update['object_id']
    medias, next = api.geography_recent_media(geography_id=geo_id, count=1)

    def tag_format(tags):
        """ Format tag for database """
        tags = ",".join(map(lambda tag: tag.name, tags))
        return tags

    logging.getLogger(__name__).info("Processing object with location data")
    media_tuples = map(lambda media_el: (media_el.id, media_el.user.username, media_el.user.id,
                           (tag_format(media_el.tags) if hasattr(media_el, 'tags') else ""),
                           (media_el.location.name if hasattr(media_el.location, 'name') else ""),
                            media_el.location.point.latitude, media_el.location.point.longitude,
                            media_el.filter, media_el.created_time.strftime("%Y-%m-%d %H:%M:%S"),
                            media_el.get_standard_resolution_url(),
                            media_el.link,
                            (media_el.caption if hasattr(media_el, 'caption') else "")), medias)


    db = connect_db()
    try:
        cursor = db.cursor()
        cursor.executemany(insert_query, media_tuples)
        db.commit()
    except Exception as e:
        logging.getLogger(__name__).error(insert_query)
        logging.getLogger(__name__).info(media_tuples)
        logging.getLogger(__name__).error("Database error: ")
        logging.getLogger(__name__).error(e)
    finally:
        db.close()

@app.route('/redirect')
def on_callback():
    """
    This gets the access token for the instagram api
    """
    print "redirect"
    code = request.values.get("code")
    if not code:
        return 'Missing code'
    try:
        instagram_client = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI)
        access_token, instagram_user = instagram_client.exchange_code_for_access_token(code)
        print "access token= " + access_token
        if not access_token:
            return Response('Could not get access token')

    except Exception, e:
        logging.getLogger(__name__).error(e)
        return Response("Didn't get the access token.")

    return Response("ok")


@app.route('/realtime_callback', methods=["GET", "POST"])
def on_realtime_callback():
    """
    When creating a real time subscription, need to return a challenge
    """
    print request
    mode = request.values.get('hub.mode')
    challenge = request.values.get('hub.challenge')
    verify_token = request.values.get('hub.verify_token')
    if challenge:
        return Response(challenge)
    else:
        reactor = subscriptions.SubscriptionsReactor()
        reactor.register_callback(subscriptions.SubscriptionType.GEOGRAPHY, process_geo_location)

        x_hub_signature = request.headers.get('X-Hub-Signature')
        raw_response = request.data
        try:
            reactor.process(CLIENT_SECRET, raw_response, x_hub_signature)
        except subscriptions.SubscriptionVerifyError:
            logging.error('Instagram signature mismatch')

    return Response('Parsed instagram')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8130, debug=True)

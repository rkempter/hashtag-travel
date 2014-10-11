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

# The subscription_counter takes care of memorizing the 'next' and 'last' id of
# collected medias
subscriptions_dict = {}

# The web application
app = Flask(__name__)

# The shared dictionary
manager = multiprocessing.Manager()
shared_dict = manager.dict()

# The queue of tasks with pull information
queue = multiprocessing.Queue()

# The geographical locations that we want to subscribe to
LOCATIONS = [
    { 'lat': 48.916889692, 'long': 2.344497138, 'radius': 5000 },
    { 'lat': 48.883487422, 'long': 2.232917243, 'radius': 5000 },
    { 'lat': 48.857065773, 'long': 2.355140143, 'radius': 5000 },
    { 'lat': 48.816615905, 'long': 2.274115973, 'radius': 5000 },
    { 'lat': 48.822493333, 'long': 2.162536078, 'radius': 5000 },
]

def queue_handling():
    """
    Handles new objects that arrive in the queue
    """
    pass

@app.route('/init')
def start():
    """
    Define callback for geographical post messages. Connect to the instagram api and
    get an access token.
    """
    g.reactor = subscriptions.SubscriptionsReactor()
    g.reactor.register_callback(subscriptions.SubscriptionType.GEOGRAPHY, process_geo_location)
    g.unauthenticated_api = client.InstagramAPI(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI
    )
    return redirect(g.unauthenticated_api.get_authorize_url(scope=['basic']))

@app.route('/subscribe')
def subscribe_location():
    """
    Subscribes to all locations and store the subscription id.
    """
    api = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    for location in LOCATIONS:
        response = api.create_subscription(
            object='geography', lat=location['lat'], lng=location['long'],
            radius=location['radius'], aspect='media',
            callback_url=RETURN_URI)
        print response
        for el in response.data:
            if 'subscription' in el['type']:
                subscriptions_dict[el['object_id']] = {'counter': 0}

    return Response('Subscribed')

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

def process_geo_location(updates):
    """
    Process a list of updates and add them to subscriptions.
    @Todo: This can be done using multiprocessing as well.
    """
    print updates
    #for update in updates:
    #    subscription_id = update['subscription_id']
    #    subscription_val = subscriptions_dict[subscription_id]
    #    lock.acquire()
    #    try:
    #        if 'next' not in subscription_val:
    #            subscription_val['next'] = update['object_id']
    #        subscription_val['last'] = update['object_id']
    #        subscription_val['counter'] += 1
    #        if subscription_val['counter'] > THRESHOLD:
    #            retrieve_recent_geo(subscription_val)
    #    finally:
    #        lock.release()

@app.route('/list')
def list():
    """
    List all current subscriptions
    """
    api = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    subscriptions = api.list_subscriptions()

    return Response("You have the following subscriptions: " + subscriptions)

@app.route('/unsubscribe')
def unsubscribe():
    """
    Unsubscribe from all subscriptions
    """
    api = client.InstagramAPI(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    try:
        subscriptions = api.list_subscriptions()
        for subscription in subscriptions:
            api.delete_subscriptions(id=subscription)
    except Exception as error:
        print error

    return Response("Unsubscribed from all subscriptions")

def retrieve_recent_geo(subscription_val):
    """
    Do batch retrieval of many different instagram medias
    """
    insert_query = """INSERT IGNORE INTO media_events (
                        'user_name', 'user_id', 'tags', 'location_name', 'location_lat',
                        'location_lng', 'filter', 'created_time', 'image_url', 'media_url',
                        'text') VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    recent_media = {}
    api = client.InstagramAPI(access_token=ACCESS_TOKEN)
    while True:
        new_recent_media, next = api.geography_recent_media(id=id, min_id=subscriptions_dict[id]['next'], count=subscription_val['counter'])
        if new_recent_media['pagination']['next_max_id'] > subscription_val['last']:
            break;
        else:
            recent_media.update(new_recent_media)

        subscription_val['next'] = next
        subscription_val['counter'] = 0

        insert_media = map(lambda x: (
            x['caption']['from'], x['caption']['id'], x['tags'], x['location']['name'],
            x['location']['latitude'], x['location']['longitude'], x['filter'], x['created_time'],
            x['images']['standart_resolution'], x['link'], x['caption']['text']), recent_media)

        db = getattr(g, 'db', None)
        if db:
            cursor = db.cursor()
            cursor.executemany(insert_query, insert_media)
            db.commit()
        else:
            raise Exception("The database is not defined.")

@app.before_request
def before_request():
    """
    Connect to the database
    """
    g._database = connect_db()

@app.teardown_appcontext
def teardown_db(exception):
    """
    Teardown the database after a request
    """
    db = getattr(g, '_database', None)
    if db is not None:
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

    return Response("Fine.")

@app.route('/realtime_callback')
def on_realtime_callback():
    """
    When creating a real time subscription, need to return a challenge
    """
    print "Realtime callback"
    mode = request.values.get("hub.mode")
    challenge = request.values.get("hub.challenge")
    verify_token = request.values.get("ub.verify_token")
    if request.method == 'POST':
        x_hub_signature = request.headers.get('X-Hub-Signature')
        raw_response = request.data
        try:
            g.reactor.process(CLIENT_SECRET, raw_response, x_hub_signature)
        except subscriptions.SubscriptionVerifyError as e:
            print "Signature mismatch"
    elif challenge:
        return Response(challenge)

    return Response('Parsed instagram')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8130, debug=True)

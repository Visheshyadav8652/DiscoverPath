import os

import joblib
from flask import Flask, render_template, request, redirect, flash, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from googletrans import Translator
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic

app = Flask(__name__)

df = pd.read_csv('data/Top Indian Places to Visit1.csv')
df2 = pd.read_csv('data/travel9.csv')
df3 = pd.read_csv('data/travel10.csv')
# df3 = pd.read_csv('data/Travel_places2.csv')

# Add a new column for Image URLs (initially empty)
df2['Image_URL'] = ''

# Sample image URLs (assuming they are stored locally in 'static/images/')
sample_images = {
    'Delhi': 'static/img/delhi.jpg',
    'Goa': 'static/img/goa.jpg',
    'Kerala': 'static/img/kerala.jpg',
    'Rajasthan': 'static/img/rajasthan.jpg',
    'Maharashtra': 'static/img/mumbai.jpg',
    'Chhattisgarh': 'static/img/telangana.jpg',
    'Madhya Pradesh': 'static/img/pexels.jpg',
    'Uttar Pradesh': 'static/img/uttarpradesh.jpg',
    'Bihar': 'static/img/pexels.jpg',
    'Jharkhand': 'static/img/jharkhand.jpg',
    'Odisha': 'static/img/odisha.jpg',
    'Sikkim': 'static/img/sikkim.jpg',
    'West Bengal': 'static/img/bengal.jpg',
    'Arunachal Pradesh': 'static/img/arunachal.jpg'

}

# Default image URL for states without a specific image
default_image = 'static/img/himachal.jpg'

# Populate the Image_URL column based on the State
df2['Image_URL'] = df['State'].map(sample_images).fillna(default_image)

# API section

app = Flask(__name__)

GEOAPIFY_API_KEY = 'ac05609ec8404db5bfbff65c1a0ab934'  # Replace with your Geoapify API key


def get_place_info(destination):
    # Get the coordinates of the destination using Geoapify's geocoding API
    print("get places called")
    geocode_url = f"https://api.geoapify.com/v1/geocode/search?text={destination}&apiKey={GEOAPIFY_API_KEY}"
    response = requests.get(geocode_url).json()
    # print(response)
    if response['features']:
        feature = response['features'][0]
        coordinates = response['features'][0]['geometry']['coordinates']
        longitude, latitude = coordinates[0], coordinates[1]
        place_id = feature['properties']['place_id']
        # print(f"Place ID: {place_id}")
        # Get places of interest around the destination
        places_url = f"https://api.geoapify.com/v2/places?categories=tourism.sights&filter=place:{place_id}&limit=10&apiKey={GEOAPIFY_API_KEY}"
        # hotels_url = f"https://api.geoapify.com/v2/places?categories=accomodation.hotel&filter=place:{place_id}&limit=10&apiKey={GEOAPIFY_API_KEY}"

        places_response = requests.get(places_url).json()
        return places_response['features']
    else:
        return []


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'  # SQLite example
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Needed for flash messages
db = SQLAlchemy(app)


# Database model for user registration
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(64), nullable=False)
    lastname = db.Column(db.String(64))
    email = db.Column(db.String(125), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f'{self.id} - {self.firstname}'


# Itinieries database
# Database model for user itineraries
class Itinerary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    destination = db.Column(db.String(100), nullable=False)
    how_many = db.Column(db.Integer, nullable=False)
    arrival_date = db.Column(db.DateTime, nullable=False)
    departure_date = db.Column(db.DateTime, nullable=False)
    # places = db.Column(db.JSON, nullable=False)  # Use PickleType for storing Python objects

    user = db.relationship('User', backref=db.backref('itineraries', lazy=True))

    def __repr__(self) -> str:
        return f'Itinerary({self.id}, {self.destination}, {self.user_id})'


# itineries database ends here

# @app.route('/clusters', methods=['GET', 'POST'])
# def clusters():
#     cluster_states = []
#     input_state = None
#
#     # Assume df2 has the required clustering data
#     if request.method == 'POST':
#         input_state = request.form.get('state_name')
#
#         # Get the cluster for the input state from df2
#         if input_state in df2['State'].values:
#             cluster = df2.loc[df2['State'] == input_state, 'loc_clusters'].iloc[0]
#
#             # Get all the states in the same cluster from df2
#             cluster_states = df2.loc[df2['loc_clusters'] == cluster, 'State'].unique().tolist()
#
#             # Remove the input state from the list
#             if input_state in cluster_states:
#                 cluster_states.remove(input_state)
#
#     states = df2['State'].unique().tolist()  # Get states from df2
#     return render_template('clusters.html', states=states, input_state=input_state, cluster_states=cluster_states)

# @app.route('/clusters', methods=['GET', 'POST'])
# def clusters():
#     cluster_cities_with_index = []
#     input_city = None
#
#     # Assume df2 has the required clustering data
#     if request.method == 'POST':
#         input_city = request.form.get('city_name')
#
#         # Get the cluster for the input city from df2
#         if input_city in df2['City'].values:
#             cluster = df2.loc[df2['City'] == input_city, 'loc_clusters'].iloc[0]
#
#             # Get all the cities in the same cluster along with their indices
#             cluster_data = df2.loc[df2['loc_clusters'] == cluster, ['City']].reset_index()
#
#             # Convert to a list of tuples (index, city)
#             cluster_cities_with_index = list(cluster_data.itertuples(index=False, name=None))
#
#             # Remove the input city from the list
#             cluster_cities_with_index = [city_info for city_info in cluster_cities_with_index if city_info[1] != input_city]
#
#     cities = df2['City'].unique().tolist()  # Get all city names from df2
#     return render_template('clusters.html', cities=cities, input_city=input_city, cluster_cities_with_index=cluster_cities_with_index)

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    cluster_cities_with_index = []
    input_city = None

    # Assume df2 has the required clustering data
    if request.method == 'POST':
        input_city = request.form.get('city_name')

        # Get the cluster for the input city from df2
        if input_city in df2['City'].values:
            cluster = df2.loc[df2['City'] == input_city, 'loc_clusters'].iloc[0]

            # Get all the cities in the same cluster along with their indices and images
            cluster_data = df2.loc[df2['loc_clusters'] == cluster, ['City', 'img', 'Name']].reset_index()

            # Convert to a list of tuples (index, city, img)
            cluster_cities_with_index = list(cluster_data.itertuples(index=False, name=None))

            # Remove the input city from the list
            cluster_cities_with_index = [city_info for city_info in cluster_cities_with_index if
                                         city_info[1] != input_city]

    cities = df2['City'].unique().tolist()  # Get all city names from df2
    return render_template('clusters.html', cities=cities, input_city=input_city,
                           cluster_cities_with_index=cluster_cities_with_index)


# Preprocess the data for recommendations
df_for_similarity = df[['Name', 'Google review rating', 'Number of google review in lakhs']].copy()

# Handle missing values (replace with 0 for numerical features if needed)
for col in df_for_similarity.columns:
    if pd.api.types.is_numeric_dtype(df_for_similarity[col]):
        df_for_similarity[col].fillna(0, inplace=True)

# Scale the numerical features
scaler = MinMaxScaler()
df_for_similarity[['Google review rating', 'Number of google review in lakhs']] = scaler.fit_transform(
    df_for_similarity[['Google review rating', 'Number of google review in lakhs']]
)

# Create the cosine similarity matrix
cosine_sim_reviews = cosine_similarity(
    df_for_similarity[['Google review rating', 'Number of google review in lakhs']]
)


# def get_recommendations_by_reviews_and_rating(place_name, cosine_sim=cosine_sim_reviews, df=df):
#     """
#     Recommends similar places based on Google review rating and number of reviews.
#     """
#     idx = df.index[df['Name'] == place_name].tolist()[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:6]  # Get top 5 similar places (excluding itself)
#     place_indices = [i[0] for i in sim_scores]
#     return df[['Name', 'Google review rating']].iloc[place_indices].to_dict(orient='records')
#
#
# @app.route('/recommendations', methods=['GET', 'POST'])
# def recommendations():
#     recommended_places = []
#     place_name = None
#     if request.method == 'POST':
#         place_name = request.form.get('place_name')
#         if place_name in df['Name'].values:
#             recommended_places = get_recommendations_by_reviews_and_rating(place_name)
#         else:
#             flash("Place not found in the database", "error")
#
#     return render_template('recommendations.html', place_name=place_name, recommended_places=recommended_places)

def get_recommendations_by_reviews_and_rating(place_name, cosine_sim=cosine_sim_reviews, df=df2):
    """
    Recommends similar places based on Google review rating and number of reviews.
    """
    # Find the index of the place in the dataframe
    idx = df2.index[df2['Name'] == place_name].tolist()[0]

    # Get similarity scores for all places with the selected place
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar places (excluding itself)

    # Get the indices of the recommended places
    place_indices = [i[0] for i in sim_scores]

    # Return a list of dictionaries with place details, including index and img URL
    return df2.loc[place_indices, ['Name', 'Google review rating', 'img']].reset_index().to_dict(orient='records')


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    recommended_places = []
    place_name = None

    if request.method == 'POST':
        place_name = request.form.get('place_name')
        if place_name in df['Name'].values:
            recommended_places = get_recommendations_by_reviews_and_rating(place_name)
        else:
            flash("Place not found in the database", "error")

    return render_template('recommendations.html', place_name=place_name, recommended_places=recommended_places)


# Home route
@app.route('/')
def home():
    states = df['State'].unique().tolist()
    return render_template('index.html', states=states)


# @app.route('/', methods=['GET', 'POST'])
# def index2():
#     if request.method == 'POST':
#         destination = request.form.get('destination')
#         how_many = request.form.get('how_many')
#         arrival_date = request.form.get('arrival_date')
#         departure_date = request.form.get('departure_date')
#
#         # Fetch place information
#         places = get_place_info(destination)
#         return render_template('itinerary.html', destination=destination, how_many=how_many, arrival_date=arrival_date, departure_date=departure_date, places=places)
#
#     return render_template('index2.html')

@app.route('/london')
def london():
    return render_template('london.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/india')
def india():
    # Convert the dataframe to a list of dictionaries for easy manipulation in Jinja
    places = df.to_dict(orient='records')
    return render_template('india.html', places=places)


# # Route to display states by zone
# @app.route('/states_by_zone')
# def states_by_zone():
#     states_by_zone = df.groupby('Zone')['State'].unique().reset_index()
#     states_by_zone.columns = ['Zone', 'States']
#     states_by_zone_dict = states_by_zone.to_dict(orient='records')
#     return render_template('states_by_zone.html', states_by_zone=states_by_zone_dict)

@app.route('/states_by_zone')
def states_by_zone():
    # Group states by zone, ensuring that 'State' and 'Image_URL' are unique
    states_by_zone = df2.groupby(['Zone', 'State']).first().reset_index()
    states_by_zone_dict = states_by_zone[['Zone', 'State', 'Image_URL']].to_dict(orient='records')
    return render_template('states_by_zone.html', states_by_zone=states_by_zone_dict)


# Route to display places by state
@app.route('/places_by_state')
def places_by_state():
    places_by_state = df.groupby('State')['Name'].apply(list).reset_index()

    places_by_state.columns = ['State', 'Places']
    places_by_state_dict = places_by_state.to_dict(orient='records')
    return render_template('places_by_state.html', places_by_state=places_by_state_dict)


# Route to show places for a specific state
# @app.route('/places/<state>')
# def show_places(state):
#     state_description = df2[df2['State'] == state]['State Description'].iloc[0]
#     places = df[df['State'] == state]['Name'].tolist()
#     return render_template('show_places.html', state=state, places=places, description=state_description)

# @app.route('/places/<state>', methods=['GET', 'POST'])
# def show_places(state):
#     state_description = df2[df2['State'] == state]['State Description'].iloc[0]
#     # places = df[df['State'] == state]['Name'].tolist()
#
#     # Retrieve the Image_URL for the selected state
#     image_url = df2[df2['State'] == state]['Image_URL'].iloc[0]
#     print(f"Image URL for state {state}: {image_url}")
#
#     state_places = df[df['State'] == state][['Name', 'Type', 'Establishment Year', 'Google review rating','Significance','Best Time to visit']].to_dict(
#         orient='records')
#
#     # Pass the image_url to the template
#     return render_template('show_places.html', state=state, places=state_places, description=state_description,
#                            image_url=image_url,state_places=state_places)

@app.route('/places/<state>', methods=['GET', 'POST'])
def show_places(state):
    state_description = df2[df2['State'] == state]['State Description'].iloc[0]
    # Include 'Sno' to have a unique identifier for each place
    state_places = df[df['State'] == state][
        ['Sno', 'Name', 'Type', 'Establishment Year', 'Google review rating', 'Significance',
         'Best Time to visit']].to_dict(orient='records')
    image_url = df2[df2['State'] == state]['Image_URL'].iloc[0]
    return render_template('show_places.html', state=state, places=state_places, description=state_description,
                           image_url=image_url)


# @app.route('/add_to_itinerary/<int:place_id>')
# def add_to_itinerary(place_id):
#     if 'user_id' not in session:
#         flash('Please log in to add places to your itinerary', 'error')
#         return redirect(url_for('login'))
#
#     # Get the place by place_id
#     place = df[df['Sno'] == place_id].to_dict(orient='records')
#     if not place:
#         flash("Place not found", "error")
#         return redirect(url_for('home'))
#     place = place[0]
#
#     user_id = session['user_id']
#
#     # Get the user's itineraries, ordered by creation date descending
#     itineraries = Itinerary.query.filter_by(user_id=user_id).order_by(Itinerary.id.desc()).all()
#
#     if itineraries:
#         # Get the most recent itinerary
#         itinerary = itineraries[0]
#         places = itinerary.places if itinerary.places else []
#     else:
#         # No itineraries exist for the user
#         flash('No existing itinerary found. Please create an itinerary first.', 'error')
#         return redirect(url_for('dashboard'))
#
#     # Check if the place is already in the itinerary
#     if place_id not in [p['Sno'] for p in places]:
#         places.append(place)
#         itinerary.places = places
#         db.session.commit()
#         flash(f"{place['Name']} has been added to your itinerary", "success")
#     else:
#         flash(f"{place['Name']} is already in your itinerary", "info")
#
#     return redirect(url_for('show_places', state=place['State']))

@app.route('/add_to_itinerary/<int:place_id>', methods=['GET', 'POST'])
def add_to_itinerary(place_id):
    if 'user_id' not in session:
        flash("You need to log in to add places to your itinerary.", "error")
        return redirect(url_for('login'))  # Redirect to login if user is not authenticated

    user_id = session['user_id']  # Get user ID from session

    # Retrieve the place information based on the place_id
    place = df[df['Sno'] == place_id].to_dict(orient='records')

    if not place:
        flash("Place not found!", "error")
        return redirect(url_for('show_places'))  # Handle if the place is not found

    # Since place is a list of dictionaries, access the first element
    place = place[0]  # Now place is a dictionary with the correct attributes

    # Create a new itinerary entry
    new_itinerary = Itinerary(
        user_id=user_id,
        destination=place['Name'],
        how_many=1,  # You can adjust how many based on your logic
        arrival_date=datetime.utcnow(),  # Example date, adjust as necessary
        departure_date=datetime.utcnow() + timedelta(days=1),  # Example date, adjust as necessary
        # places=[place]  # Store the place information as a JSON
    )

    # Add to the database and commit
    db.session.add(new_itinerary)
    db.session.commit()

    flash(f"{place['Name']} has been added to your itinerary!", "success")
    return redirect(url_for('show_places', state=place['State']))  # Redirect back to places or desired page


# Route to show states for a specific zone
@app.route('/zones/<zone>')
def show_states(zone):
    states = df[df['Zone'] == zone]['State'].unique().tolist()
    return render_template('show_states.html', zone=zone, states=states)


@app.route('/place/<int:sno>')
def place_details(sno):
    # Find the place by 'Sno' from the CSV data
    place = df[df['Sno'] == sno].to_dict(orient='records')
    if not place:
        flash("Place not found", "error")
        return redirect(url_for('home'))
    return render_template('place_details.html', place=place[0])


# Registration route

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Retrieve form data
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Simple validation
        if not first_name or not email or not password:
            if not first_name:
                flash('First name is required', 'err_first_name')
            if not email:
                flash('Email is required', 'err_email')
            if not password:
                flash('Password is required', 'err_password')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match', 'err_password')
            return redirect(url_for('register'))

        # Check if user already exists
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already registered', 'err_email')
            return redirect(url_for('register'))

        # Hash the password using default pbkdf2:sha256
        hashed_password = generate_password_hash(password)

        # Create new user and add to database
        new_user = User(firstname=first_name, lastname=last_name, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful!', 'success')
        return redirect(url_for('home'))

    return render_template('register.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Fetch the user by email
        user = User.query.filter_by(email=email).first()

        # Check if user exists and password matches
        if user and check_password_hash(user.password_hash, password):
            # Log the user in (store user information in session)
            session['user_id'] = user.id
            session['user_name'] = user.firstname
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('index.html')


# # Dashboard route (protected page)
# @app.route('/dashboard')
# def dashboard():
#     if 'user_id' not in session:
#         flash('Please log in to access this page', 'error')
#         return redirect(url_for('login'))
#
#     return f"Welcome to your dashboard, {session['user_name']}!"


# Dashboard route (protected page)
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        destination = request.form.get('destination')
        how_many = request.form.get('how_many')
        arrival_date = request.form.get('arrival_date')
        departure_date = request.form.get('departure_date')

        # Fetch place information
        places = get_place_info(destination)

        # Fetch an image URL for the destination
        image_url = get_google_images(destination)

        # Fetch an image URL for each place in the 'places' list
        for place in places:
            place_name = place['properties']['name']
            place['image_url'] = get_google_images(place_name)

        # Create a new itinerary and save it to the database
        new_itinerary = Itinerary(
            user_id=session['user_id'],
            destination=destination,
            how_many=int(how_many),
            arrival_date=datetime.strptime(arrival_date, '%Y-%m-%d'),
            departure_date=datetime.strptime(departure_date, '%Y-%m-%d'),
            # places=json.dumps(places)
        )
        db.session.add(new_itinerary)
        db.session.commit()

        return render_template('itinerary.html', destination=destination, how_many=how_many, arrival_date=arrival_date,
                               departure_date=departure_date, places=places, image_url=image_url)

    # Pass the user's first name to the template
    return render_template('dashboard.html', user_name=session.get('user_name'))


import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode


def get_google_images(query):
    # Format the search query by replacing spaces with '+'
    query = query.replace(' ', '+')
    google_search_url = f"https://in.images.search.yahoo.com/search/images?p={query}"

    # Headers to simulate a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        # Send a GET request to Google
        response = requests.get(google_search_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image tags
        image_tags = soup.find_all('img')

        # Extract image URLs (Skip the first one as it may be a logo)
        image_urls = [img['src'] for img in image_tags if 'src' in img.attrs]

        # Return the first valid image URL after skipping the first
        return image_urls[1] if len(image_urls) > 1 else None
    except Exception as e:
        print(f"Error fetching image URL for {query}: {e}")
        return None


@app.route('/my_itineraries')
def my_itineraries():
    if 'user_id' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))

    itineraries = Itinerary.query.filter_by(user_id=session['user_id']).all()

    # For each itinerary, fetch the image for the destination
    for itinerary in itineraries:
        # Use get_google_images to fetch image for each itinerary's destination
        image_url = get_google_images(itinerary.destination)
        itinerary.image_url = image_url  # Dynamically add image_url to each itinerary object

    return render_template('my_itineraries.html', itineraries=itineraries)


# on the basis of users data

def get_popular_destinations(limit=10):
    all_destinations = []

    # Retrieve all itineraries
    itineraries = Itinerary.query.all()

    # Loop through each itinerary and extract the destination
    for itinerary in itineraries:
        # Append the destination name, ensuring it's properly capitalized
        all_destinations.append(itinerary.destination.title())

    # Use Counter to find the most common destinations
    destination_counter = Counter(all_destinations)
    popular_destinations = destination_counter.most_common(limit)

    # Get images for the popular destinations
    for i in range(len(popular_destinations)):
        destination_name = popular_destinations[i][0]
        # Fetch the image for each destination
        image_url = get_google_images(destination_name)
        popular_destinations[i] = {
            'name': destination_name,
            'count': popular_destinations[i][1],
            'image_url': image_url
        }

    return popular_destinations

    # Return a list of dictionaries with destination name and count
    # return [{'name': name, 'count': count} for name, count in popular_destinations]


@app.route('/popular_recommendations')
def popular_recommendations():
    # Get the popular destinations using the new function
    popular_destinations = get_popular_destinations()
    return render_template('popular_recommendations.html', popular_destinations=popular_destinations)


# it ends here

# similarity 4 way

# Step 2: Preprocess data (convert to lowercase for case-insensitive comparison)
df3['Zone'] = df3['Zone'].str.lower()
df3['State'] = df3['State'].str.lower()
df3['City'] = df3['City'].str.lower()
df3['Name'] = df3['Name'].str.lower()
df3['Type'] = df3['Type'].str.lower()
df3['Significance'] = df3['Significance'].str.lower()
df3['Season'] = df3['Season'].str.lower()

# Step 2: Encode categorical features using LabelEncoder
label_enc = LabelEncoder()
df3['Zone_enc'] = label_enc.fit_transform(df3['Zone'])
df3['State_enc'] = label_enc.fit_transform(df3['State'])
df3['City_enc'] = label_enc.fit_transform(df3['City'])
df3['Type_enc'] = label_enc.fit_transform(df3['Type'])
df3['Season_enc'] = label_enc.fit_transform(df3['Season'])

# Step 3: Calculate text-based similarity for 'Name' and 'Significance' columns using TF-IDF
tfidf = TfidfVectorizer()
tfidf_name = tfidf.fit_transform(df3['Name'])
tfidf_significance = tfidf.fit_transform(df3['Significance'])

name_sim = cosine_similarity(tfidf_name)
significance_sim = cosine_similarity(tfidf_significance)

# Step 4: Calculate similarity scores for categorical features (Zone, State, City, Type, Season)
zone_sim = (df3['Zone_enc'].values.reshape(-1, 1) == df3['Zone_enc'].values.reshape(1, -1)).astype(float)
state_sim = (df3['State_enc'].values.reshape(-1, 1) == df3['State_enc'].values.reshape(1, -1)).astype(float)
city_sim = (df3['City_enc'].values.reshape(-1, 1) == df3['City_enc'].values.reshape(1, -1)).astype(float)
type_sim = (df3['Type_enc'].values.reshape(-1, 1) == df3['Type_enc'].values.reshape(1, -1)).astype(float)
season_sim = (df3['Season_enc'].values.reshape(-1, 1) == df3['Season_enc'].values.reshape(1, -1)).astype(float)

# Step 5: Combine similarity scores using predefined weights
alpha = [0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2]  # Weights assigned to each feature
overall_similarity = (
    alpha[0] * zone_sim +
    alpha[1] * state_sim +
    alpha[2] * city_sim +
    alpha[3] * type_sim +
    alpha[4] * name_sim +
    alpha[5] * significance_sim +
    alpha[6] * season_sim
)

# Step 6: Define a function to find similar places based on user input
def find_similar_places(input_text):
    input_text = input_text.lower()

    # Detect which column the input matches and find the corresponding index
    if (df3['Zone'] == input_text).any():
        query_index = df3[df3['Zone'] == input_text].index[0]
    elif (df3['State'] == input_text).any():
        query_index = df3[df3['State'] == input_text].index[0]
    elif (df3['City'] == input_text).any():
        query_index = df3[df3['City'] == input_text].index[0]
    elif (df3['Name'] == input_text).any():
        query_index = df3[df3['Name'] == input_text].index[0]
    elif (df3['Significance'] == input_text).any():
        query_index = df3[df3['Significance'] == input_text].index[0]
    elif (df3['Season'] == input_text).any():
        query_index = df3[df3['Season'] == input_text].index[0]
    else:
        return "No match found."

    # Calculate similarity and return top results
    similarity_scores = overall_similarity[query_index]
    top_indices = similarity_scores.argsort()[::-1][1:6]  # Exclude the query itself
    similar_places = df2.iloc[top_indices][['Name','img']]
    return similar_places.to_dict(orient='records')
# similarity 4 way ends

# @app.route('/similarity', methods=['GET', 'POST'])
# def similarity():
#     if request.method == 'POST':
#         input_text = request.form['input_text']
#         results = find_similar_places(input_text)
#         if isinstance(results, str):  # Handle case where no match is found
#             return render_template('similarity.html', result=results)
#         else:
#             return render_template('similarity.html', result=results.to_html(index=False))
#     else:
#         return render_template('similarity.html')

@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    if request.method == 'POST':
        input_text = request.form['input_text']
        results = find_similar_places(input_text)

        if isinstance(results, str):  # Handle case where no match is found
            return render_template('similarity.html', result=results)
        else:
            # Combine names and images into a single list of dictionaries
            places = [{'name': place['Name'], 'img': place['img']} for place in results]
            return render_template('similarity.html', places=places)
    else:
        return render_template('similarity.html')


# route optimizer
# Load your dataset
df4 = pd.read_csv('data/travel8.2.csv')


# Function to calculate the distance between two sets of latitude and longitude
def calculate_distance(lat1, lon1, lat2, lon2):
    start_coords = (lat1, lon1)
    end_coords = (lat2, lon2)
    return geodesic(start_coords, end_coords).kilometers


@app.route('/route')
def route():
    return render_template('route.html')


# Option 1: Recommend the nearest place based on user's current location
# @app.route('/recommend_nearest_place', methods=['POST'])
# def recommend_nearest_place():
#     data = request.get_json()
#     current_location = data['location']
#     visited_places = data['visited_places']  # List of visited places
#
#     # Get the user's current location from the dataset
#     user_location = df4[df4['Name'] == current_location]
#
#     if user_location.empty:
#         return jsonify({'error': 'Place not found in the dataset!'})
#
#     user_lat = user_location.iloc[0]['latitude']
#     user_lon = user_location.iloc[0]['longitude']
#
#     # Calculate distances from the current location to all other places, excluding visited places
#     distances = []
#     for index, row in df4.iterrows():
#         place_name = row['Name']
#         lat = row['latitude']
#         lon = row['longitude']
#
#         # Skip the current location and already visited places
#         if place_name != current_location and place_name not in visited_places:
#             distance = calculate_distance(user_lat, user_lon, lat, lon)
#             distances.append((place_name, distance))
#
#     # Find the nearest unvisited place
#     if distances:
#         nearest_place = min(distances, key=lambda x: x[1])  # Nearest place by distance
#         return jsonify({
#             'nearest_place': nearest_place[0],
#             'distance': round(nearest_place[1], 2)
#         })
#     else:
#         return jsonify({'message': 'No other unvisited places found in the dataset.'})
#
#
# # Option 2: Recommend a route based on user's input
# @app.route('/recommend_route', methods=['POST'])
# def recommend_route():
#     data = request.get_json()
#     current_location = data['location']
#     places = data['places']
#
#     # Find the user's current location in the dataset
#     user_location = df4[df4['Name'] == current_location]
#     if user_location.empty:
#         return jsonify({'error': 'Place not found in the dataset!'})
#
#     user_lat = user_location.iloc[0]['latitude']
#     user_lon = user_location.iloc[0]['longitude']
#
#     # Initialize the route with the current location
#     route = [current_location]
#     current_lat, current_lon = user_lat, user_lon
#
#     while places:
#         nearest_place = None
#         min_distance = float('inf')
#
#         # Iterate over the places to find the nearest one
#         for place in places:
#             location_data = df4[df4['Name'] == place]
#
#             if location_data.empty:
#                 # Skip places that are not in the dataset
#                 continue
#
#             lat = location_data.iloc[0]['latitude']
#             lon = location_data.iloc[0]['longitude']
#
#             # Calculate the distance to the place
#             distance = calculate_distance(current_lat, current_lon, lat, lon)
#             # print(f"Distance from {route[-1]} to {place}: {distance} km")  # Debugging line
#
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_place = place
#
#         if nearest_place:
#             route.append(nearest_place)
#             places.remove(nearest_place)
#
#             # Update current location coordinates
#             current_location_data = df4[df4['Name'] == nearest_place]
#             current_lat = current_location_data.iloc[0]['latitude']
#             current_lon = current_location_data.iloc[0]['longitude']
#         else:
#             # If no valid places are found, break the loop
#             break
#
#     final_route = ' -> '.join(route)
#     return jsonify({'route': final_route})

# Option 1: Recommend the nearest place based on user's current location, along with index and img
@app.route('/recommend_nearest_place', methods=['POST'])
def recommend_nearest_place():
    data = request.get_json()
    current_location = data['location']
    visited_places = data['visited_places']  # List of visited places

    # Get the user's current location from the dataset
    user_location = df4[df4['Name'] == current_location]

    if user_location.empty:
        return jsonify({'error': 'Place not found in the dataset!'})

    user_lat = user_location.iloc[0]['latitude']
    user_lon = user_location.iloc[0]['longitude']

    # Calculate distances from the current location to all other places, excluding visited places
    distances = []
    for index, row in df4.iterrows():
        place_name = row['Name']
        lat = row['latitude']
        lon = row['longitude']

        # Skip the current location and already visited places
        if place_name != current_location and place_name not in visited_places:
            distance = calculate_distance(user_lat, user_lon, lat, lon)
            distances.append((index, place_name, distance))

    # Find the nearest unvisited place
    if distances:
        nearest_place = min(distances, key=lambda x: x[2])  # Nearest place by distance
        nearest_index = nearest_place[0]
        nearest_name = nearest_place[1]
        nearest_distance = round(nearest_place[2], 2)

        # Fetch the image from the 'img' column using the index
        nearest_img = df4.loc[nearest_index, 'img']

        return jsonify({
            'nearest_place': nearest_name,
            'distance': nearest_distance,
            'index': nearest_index,
            'image': nearest_img
        })
    else:
        return jsonify({'message': 'No other unvisited places found in the dataset.'})

# Option 2: Recommend a route with place names, distances, indexes, and images
@app.route('/recommend_route', methods=['POST'])
def recommend_route():
    data = request.get_json()
    current_location = data['location']
    places = data['places']

    user_location = df4[df4['Name'] == current_location]
    if user_location.empty:
        return jsonify({'error': 'Place not found in the dataset!'})

    user_lat = user_location.iloc[0]['latitude']
    user_lon = user_location.iloc[0]['longitude']

    # Initialize the route list with the current location
    route = []

    # Add the current location details (name, index, image) as the first stop
    current_place_info = {
        'place_name': current_location,
        'index': int(user_location.index[0]),  # Convert index to standard int
        'image': user_location.iloc[0]['img']  # Assuming 'img' column holds image URLs
    }
    route.append(current_place_info)

    # Start finding the route from the current location
    current_lat, current_lon = user_lat, user_lon

    while places:
        # Find the nearest place from the current location
        nearest_place = None
        min_distance = float('inf')

        for place in places:
            location_data = df4[df4['Name'] == place]
            if not location_data.empty:
                lat = location_data.iloc[0]['latitude']
                lon = location_data.iloc[0]['longitude']
                distance = calculate_distance(current_lat, current_lon, lat, lon)

                if distance < min_distance:
                    min_distance = distance
                    nearest_place = place

        # Add the nearest place details (name, index, image) to the route
        if nearest_place:
            location_data = df4[df4['Name'] == nearest_place]
            place_info = {
                'place_name': nearest_place,
                'index': int(location_data.index[0]),  # Convert index to standard int
                'image': location_data.iloc[0]['img']  # Assuming 'img' column holds image URLs
            }
            route.append(place_info)
            places.remove(nearest_place)

            # Update current location to the nearest place's coordinates
            current_lat = location_data.iloc[0]['latitude']
            current_lon = location_data.iloc[0]['longitude']

    # Create a string with arrows indicating the order of the route
    route_with_arrows = " → ".join([place['place_name'] for place in route])

    return jsonify({
        'route': route,
        'route_with_arrows': route_with_arrows
    })

def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

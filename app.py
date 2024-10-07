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

app = Flask(__name__)

df = pd.read_csv('data/Top Indian Places to Visit1.csv')
df2 = pd.read_csv('data/Travel_places3.csv')
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
    print(response)
    if response['features']:
        feature = response['features'][0]
        coordinates = response['features'][0]['geometry']['coordinates']
        longitude, latitude = coordinates[0], coordinates[1]
        place_id = feature['properties']['place_id']
        # print(f"Place ID: {place_id}")
        # Get places of interest around the destination
        places_url = f"https://api.geoapify.com/v2/places?categories=tourism.sights&filter=place:{place_id}&limit=10&apiKey={GEOAPIFY_API_KEY}"
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

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    cluster_states = []
    input_state = None

    # Assume df2 has the required clustering data
    if request.method == 'POST':
        input_state = request.form.get('state_name')

        # Get the cluster for the input state from df2
        if input_state in df2['State'].values:
            cluster = df2.loc[df2['State'] == input_state, 'loc_clusters'].iloc[0]

            # Get all the states in the same cluster from df2
            cluster_states = df2.loc[df2['loc_clusters'] == cluster, 'State'].unique().tolist()

            # Remove the input state from the list
            if input_state in cluster_states:
                cluster_states.remove(input_state)

    states = df2['State'].unique().tolist()  # Get states from df2
    return render_template('clusters.html', states=states, input_state=input_state, cluster_states=cluster_states)


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


def get_recommendations_by_reviews_and_rating(place_name, cosine_sim=cosine_sim_reviews, df=df):
    """
    Recommends similar places based on Google review rating and number of reviews.
    """
    idx = df.index[df['Name'] == place_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar places (excluding itself)
    place_indices = [i[0] for i in sim_scores]
    return df[['Name', 'Google review rating']].iloc[place_indices].to_dict(orient='records')


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
                               departure_date=departure_date, places=places)

    # Pass the user's first name to the template
    return render_template('dashboard.html', user_name=session.get('user_name'))


@app.route('/my_itineraries')
def my_itineraries():
    if 'user_id' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))

    itineraries = Itinerary.query.filter_by(user_id=session['user_id']).all()
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

    # Return a list of dictionaries with destination name and count
    return [{'name': name, 'count': count} for name, count in popular_destinations]

@app.route('/popular_recommendations')
def popular_recommendations():
    # Get the popular destinations using the new function
    popular_destinations = get_popular_destinations()
    return render_template('popular_recommendations.html', popular_destinations=popular_destinations)


# it ends here

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

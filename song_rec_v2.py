import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load the datasets
artist_path = r"\MillionSongSubset.csv"
artists = pd.read_csv(artist_path)
songs = pd.read_csv('https://raw.githubusercontent.com/AGeoCoder/Million-Song-Dataset-HDF5-to-CSV/master/SongCSV.csv')

# clean the artist dataset by dropping rows with missing values
artists = artists.dropna()
    
# reset the index of the dataframe
artists = artists.reset_index(drop=True)

# define the columns to be used for the recommendation
artist_columns = ['ArtistFamiliarity', 'Hotness']
song_columns = ['Danceability', 'Duration', 'KeySignature', 'KeySignatureConfidence', 'Tempo', 'TimeSignature', 'TimeSignatureConfidence']

# define a function to get the artist vector
def get_artist_vector(artist):
    # get the row for the specified artist
    row = artists[artists['ArtistName'] == artist]
    # extract the columns to be used for the recommendation
    artist_vector = row[artist_columns].values
    return artist_vector

# define a function to get the song vector
def get_song_vector(title):
    # get the row for the specified song title
    row = songs[songs['Title'] == title]
    # extract the columns to be used for the recommendation
    song_vector = row[song_columns].values
    return song_vector

# define a function to get the top artist recommendations
def get_artist_recommendations(artist, num_recommendations=5):
    # get the artist vector for the specified artist
    artist_vector = get_artist_vector(artist)
    if artist_vector.size == 0:
        # handle the case where the specified artist is not in the dataset
        return None
    # calculate the cosine similarity between the artist vector and all other artists
    similarities = cosine_similarity(artist_vector, artists[artist_columns])
    # get the indices of the top recommendations
    indices = similarities.argsort()[0][-num_recommendations-1:-1][::-1]
    # get the top recommendations
    recommendations = artists.loc[indices, ['ArtistName', 'ArtistFamiliarity', 'Hotness']]
    return recommendations

# define a function to get the top song recommendations
def get_song_recommendations(title, num_recommendations=5):
    # get the song vector for the specified song title
    song_vector = get_song_vector(title)
    # calculate the cosine similarity between the song vector and all other songs
    similarities = cosine_similarity(song_vector, songs[song_columns])
    # get the indices of the top recommendations
    indices = similarities.argsort()[0][-num_recommendations-1:-1][::-1]
    # get the top recommendations
    recommendations = songs.loc[indices, ['Title', 'ArtistName', 'Year']]
    return recommendations

# prompt the user to choose between artist or song recommendation
choice = input("Do you want to get artist or song recommendations? (A/S) ")
if choice == 'A':
    # prompt the user to input an artist name
    query = input("Enter an artist name: ")
    # get the top artist recommendations for the specified artist
    recommendations = get_artist_recommendations(query)
    if recommendations is not None:
        # display the top artist recommendations
        print("\nTop Artist Recommendations for {}: \n".format(query))
        print(recommendations.to_string(index=False))
    else:
        print("No recommendations found for {}".format(query))
elif choice == 'S':
    # prompt the user to input a song title
    title = input("Enter a song title: ")
    # get the top recommendations for the specified song
    recommendations = get_song_recommendations(title)
    # display the top recommendations
    print("\nTop Recommendations for {}: \n".format(title))
    print(recommendations.to_string(index=False))
else:
    print("Invalid choice.")
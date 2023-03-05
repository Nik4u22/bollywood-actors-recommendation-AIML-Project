# Import important libraries
import pandas as pd
import glob

def load_actors_image_dataset():
    
    actors_dataset = pd.DataFrame(columns=['Name','Image_Path'])
    
    actors_list = pd.read_csv('Actors_list.txt', sep=' ', header=None)
    #print(actors_list)
    # Give images folder path
    actors_image_dataset_path = str("/Users/nik4u/Downloads/archive/Bollywood Actor Images/Bollywood Actor Images")

    # Access each subfolder with the actor name and get image names
    for actor in actors_list[0]:
        downloads_path = str(actors_image_dataset_path+"/"+actor)
        #print(downloads_path)
        files = glob.glob(downloads_path+'/*')
        for f in files:
            actor_details = {'Name': actor, 'Image_Path': f}
            actors_dataset = actors_dataset.append(actor_details, ignore_index = True)

    return actors_dataset
